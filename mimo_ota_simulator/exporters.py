
import numpy as np
import os
from typing import Dict, List, Optional
import csv
import zipfile
import io

from .data_models import CalibrationConfig, PropsimExportConfig

class PropsimASCIIExporter:
    """
    Exports MIMO OTA Simulation results to Keysight Propsim F64 standard .ASC format.
    
    COMPLIANCE NOTE (Based on Propsim User Reference, Section 20.1):
    - File format is strictly 'Delay Re Im' triplets. No Time column.
    - Header must follow specific keywords (CIRs, Taps/CIR, CIR_Update_Rate).
    - One file per logical link (Tx -> Probe).
    
    To manage the multiple files generated for an OTA setup, this exporter
    can optionally package them into a single .zip archive.
    """
    
    def __init__(self, export_config: PropsimExportConfig, calibration_config: Optional[CalibrationConfig] = None):
        self.config = export_config
        self.calibration = calibration_config
        # Mode A 链路预算 (scale / digital_boost / required_attenuation_db),
        # 由 _calculate_scaling_factor 填充, 供调用方读取 (audit batch 2, A6)
        self.last_mode_a_info: Optional[Dict] = None

    def export(self, sim_result: Dict, output_path: str = None, direction: str = "dl") -> str:
        """
        Main export function. 
        Generates individual .asc files for each Probe-Tx link.
        Returns the path to the directory (or zip) containing the files.
        """
        if output_path is None:
            output_path = self.config.filename
            
        # Determine base name and directory
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        if not base_dir: base_dir = "."
        
        if direction not in ("dl", "ul"):
            raise ValueError(f"Unknown export direction: {direction}")

        if direction == "ul":
            ray_components = sim_result.get('ray_components_ul')
            impulse_response = sim_result.get('channel_impulse_response_ul')
        else:
            ray_components = sim_result.get('ray_components')
            impulse_response = sim_result.get('channel_impulse_response')
        if impulse_response is None:
            raise ValueError(f"Missing impulse response for direction '{direction}'.")
        
        # Prepare Data Tensors (Time, Probes, Tx, Taps)
        # We process everything into a unified structure before writing.
        # This simplifies the specific logic for Static vs Dynamic.
        
        apply_calibration = direction != "ul"
        if ray_components:
            print("[Propsim Exporter] Generating Wideband Dynamic Doppler data...")
            # Returns: (NumSamples, NumProbes, NumTx, NumTaps) complex, and (NumTaps) delays
            data_tensor, delays = self._compute_dynamic_tensor(
                sim_result, impulse_response, ray_components, apply_calibration=apply_calibration
            )
        else:
            print("[Propsim Exporter] Generating Wideband Static data...")
            data_tensor, delays = self._compute_static_tensor(
                sim_result, impulse_response, apply_calibration=apply_calibration
            )
            
        carrier_frequency_hz = sim_result.get('center_frequency_hz', 3.5e9)

        # Write Files (One per Link / Logical Channel)
        # F64 Architecture: 1 Channel = 1 Input -> 1 Output path.
        # For 2x2 MIMO with 16 Dual-Pol probes: 2 Inputs * 32 Outputs = 64 Files.
        
        num_samples, num_effective_probes, num_tx, num_taps = data_tensor.shape
        generated_files = []
        
        print(f"[Propsim Exporter] Writing {num_effective_probes * num_tx} ASC files (Samples={num_samples}, Taps={num_taps})...")
        
        for tx in range(num_tx):
            for p in range(num_effective_probes):
                # Construct filename: Base_InputX_OutputY.asc
                # This maps directly to F64 "Channel" definition.
                # Input: Tx antenna index (1-based)
                # Output: Physical Probe Port index (1-based, 1..32 for 16 dual-pol probes)
                fname = f"{base_name}_In{tx+1}_Out{p+1}.asc"
                fpath = os.path.join(base_dir, fname)
                
                # Extract Link Data: (Samples, Taps)
                link_data = data_tensor[:, p, tx, :] # Complex
                
                self._write_single_asc_file(fpath, link_data, delays, carrier_frequency_hz)
                generated_files.append(fpath)
                
        return generated_files

    def _generate_asc_content(self, data: np.ndarray, delays: np.ndarray, carrier_frequency_hz: float) -> str:
        """Generates the content of a standard .ASC file as a string."""
        num_samples, num_taps = data.shape
        out = io.StringIO()
        
        # --- Standard Header ---
        out.write("***** Header *****\n")
        out.write(f"{num_samples} CIRs\n")
        out.write(f"{num_taps} Taps/CIR\n")
        out.write(f"{self.config.sample_rate_hz:.1f} CIR_Update_Rate\n")
        out.write(f"{carrier_frequency_hz:.1f} Carrier_Frequency\n")
        out.write("***** Tap data *****\n")
        
        delay_strs = [f"{d:.9e}" for d in delays]
        
        for t in range(num_samples):
            row = []
            for k in range(num_taps):
                val = data[t, k]
                row.append(delay_strs[k])
                row.append(f"{val.real:.6e}")
                row.append(f"{val.imag:.6e}")
            out.write("\t".join(row) + "\n")
            
        return out.getvalue()

    def _write_single_asc_file(self, fpath: str, data: np.ndarray, delays: np.ndarray, carrier_frequency_hz: float):
        """
        Writes a single standard .ASC file to disk.
        """
        content = self._generate_asc_content(data, delays, carrier_frequency_hz)
        with open(fpath, 'w', newline='') as f:
            f.write(content)

    def export_to_zip_memory(self, sim_result: Dict, base_name: str = "propsim_export", direction: str = "dl") -> str:
        """
        Exports the generated .asc files directly into an in-memory ZIP archive
        and returns it as a Base64 encoded string. This purely relies on memory buffers
        and completely avoids local disk IO. Extremely useful for Microservices.
        """
        import base64
        
        if direction not in ("dl", "ul"):
            raise ValueError(f"Unknown export direction: {direction}")

        if direction == "ul":
            ray_components = sim_result.get('ray_components_ul')
            impulse_response = sim_result.get('channel_impulse_response_ul')
        else:
            ray_components = sim_result.get('ray_components')
            impulse_response = sim_result.get('channel_impulse_response')
            
        if impulse_response is None:
            raise ValueError(f"Missing impulse response for direction '{direction}'.")
            
        apply_calibration = direction != "ul"
        if ray_components:
            print("[Propsim Exporter] Generating Dynamic data to Memory ZIP...")
            data_tensor, delays = self._compute_dynamic_tensor(
                sim_result, impulse_response, ray_components, apply_calibration=apply_calibration
            )
        else:
            print("[Propsim Exporter] Generating Static data to Memory ZIP...")
            data_tensor, delays = self._compute_static_tensor(
                sim_result, impulse_response, apply_calibration=apply_calibration
            )
            
        carrier_frequency_hz = sim_result.get('center_frequency_hz', 3.5e9)
        num_samples, num_effective_probes, num_tx, num_taps = data_tensor.shape
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for tx in range(num_tx):
                for p in range(num_effective_probes):
                    fname = f"{base_name}_In{tx+1}_Out{p+1}.asc"
                    link_data = data_tensor[:, p, tx, :] # Complex
                    content = self._generate_asc_content(link_data, delays, carrier_frequency_hz)
                    zip_file.writestr(fname, content)
                    
        return base64.b64encode(zip_buffer.getvalue()).decode('utf-8')

    def _compute_static_tensor(self, sim_result: Dict, impulse_response: List[Dict], apply_calibration: bool = True):
        """
        Returns (Samples, Probes, Tx, Taps) and (Taps) delays.
        Values are constant over time.
        """
        # Sort taps by delay
        sorted_taps = sorted(impulse_response, key=lambda x: x['delay'])
        delays = np.array([t['delay'] for t in sorted_taps])
        
        # Get dimensions
        first_coeff = sorted_taps[0]['coeff'] # (Probes, Tx)
        num_probes, num_tx = first_coeff.shape
        num_taps = len(sorted_taps)
        num_samples = int(self.config.duration_s * self.config.sample_rate_hz)
        
        # Build Static Tensor (1, P, Tx, K)
        static_frame = np.zeros((1, num_probes, num_tx, num_taps), dtype=complex)
        
        # Apply Cal & Scaling
        scaling_factor = self._calculate_scaling_factor(sim_result, sorted_taps)
        cal_vector = self._get_calibration_vector(num_probes, apply_calibration=apply_calibration) # (P, 1)
        
        for k, tap in enumerate(sorted_taps):
            # coeff is (P, T)
            # Apply scaling
            val = tap['coeff'] * scaling_factor
            # Apply cal (broadcast over T)
            val *= cal_vector
            static_frame[0, :, :, k] = val
            
        # Repeat for time
        # (Samples, P, Tx, K)
        full_tensor = np.tile(static_frame, (num_samples, 1, 1, 1))
        
        return full_tensor, delays

    def _compute_dynamic_tensor(self, sim_result: Dict, impulse_response: List[Dict], ray_components: List[Dict], apply_calibration: bool = True):
        """
        Returns (Samples, Probes, Tx, Taps) and (Taps) delays.
        Values evolve with Doppler.
        """
        sorted_taps = sorted(impulse_response, key=lambda x: x['delay'])
        delays = np.array([t['delay'] for t in sorted_taps])
        num_taps = len(sorted_taps)
        
        # Map cluster_index to rays
        cluster_rays_map = {}
        for rc in ray_components:
            c_idx = rc['cluster_index']
            if c_idx not in cluster_rays_map: cluster_rays_map[c_idx] = []
            cluster_rays_map[c_idx].append(rc)
            
        # Get dims
        sample_w = ray_components[0]['W_base']
        num_probes, num_tx = sample_w.shape
        num_samples = int(self.config.duration_s * self.config.sample_rate_hz)
        dt = 1.0 / self.config.sample_rate_hz
        
        # Pre-calc Scaling & Cal
        scaling_factor = self._calculate_scaling_factor(sim_result, sorted_taps)
        cal_vector = self._get_calibration_vector(num_probes, apply_calibration=apply_calibration)
        
        # Pre-compute Effective Rays per Tap
        tap_engines = []
        for tap in sorted_taps:
            c_idx = tap['cluster_index']
            rays = cluster_rays_map.get(c_idx, [])
            
            eff_list = []
            for r in rays:
                phi = r['phi_init']
                W = r['W_base']
                # phi_init may be:
                #  - scalar (ray mode): broadcasts naturally
                #  - (N_probes,) vector (strict_pfs): axis-specific broadcast
                #
                # Dispatch precedence:
                #  1. Scalar phi: broadcast natively, axis irrelevant
                #  2. Vector phi WITH 'phi_axis' tag (internal synthesizer +
                #     reciprocity builder always set this): trust the tag,
                #     validate shape, apply.
                #  3. Vector phi WITHOUT 'phi_axis' tag (external callers /
                #     older saved results following the documented schema):
                #     fall back to shape inference — works for non-square
                #     W_base where phi.shape[0] matches exactly one axis.
                #     Only fail when shape is genuinely ambiguous (square W
                #     and phi matches both axes).
                if np.ndim(phi) == 0:
                    phi_factor = np.exp(1j * phi)
                else:
                    axis = r.get('phi_axis')
                    if axis is None:
                        # Untagged: infer from shape if unambiguous
                        match_row = (phi.shape[0] == W.shape[0])
                        match_col = (phi.shape[0] == W.shape[1])
                        if match_row and match_col:
                            raise ValueError(
                                f"Ambiguous phi_init axis: W_base is square "
                                f"{W.shape} and phi.shape[0]={phi.shape[0]} matches "
                                f"both axes. Tag the ray_component with "
                                f"phi_axis='row' (DL) or 'col' (UL) to disambiguate."
                            )
                        if match_row:
                            phi_factor = np.exp(1j * phi)[:, None]
                        elif match_col:
                            phi_factor = np.exp(1j * phi)[None, :]
                        else:
                            raise ValueError(
                                f"phi_init shape {phi.shape} cannot broadcast against "
                                f"W_base shape {W.shape} on either axis."
                            )
                    elif axis == 'row':
                        if phi.shape[0] != W.shape[0]:
                            raise ValueError(
                                f"phi_axis='row' but phi.shape[0]={phi.shape[0]} != "
                                f"W.shape[0]={W.shape[0]}"
                            )
                        phi_factor = np.exp(1j * phi)[:, None]
                    elif axis == 'col':
                        if phi.shape[0] != W.shape[1]:
                            raise ValueError(
                                f"phi_axis='col' but phi.shape[0]={phi.shape[0]} != "
                                f"W.shape[1]={W.shape[1]}"
                            )
                        phi_factor = np.exp(1j * phi)[None, :]
                    else:
                        raise ValueError(f"unknown phi_axis={axis!r}; expected 'row' or 'col'")
                w = W * phi_factor * scaling_factor * cal_vector
                eff_list.append({'W': w, 'w_doppler': 2*np.pi*r['doppler_hz']})
            
            if eff_list:
                W_stack = np.stack([x['W'] for x in eff_list]) # (R, P, Tx)
                Omegas = np.array([x['w_doppler'] for x in eff_list])[:, np.newaxis] # (R, 1)
                tap_engines.append({'active': True, 'W': W_stack, 'Omegas': Omegas})
                if c_idx == 0:
                    print(f"[DEBUG EXPORT] Tap {c_idx} Active Rays: {len(eff_list)}. Max Omega: {np.max(np.abs(Omegas)):.4f}")
            else:
                # Static fallback for empty tap
                static_val = tap['coeff'] * scaling_factor * cal_vector
                tap_engines.append({'active': False, 'val': static_val})

        # Generate Time Series (Chunked)
        full_tensor = np.zeros((num_samples, num_probes, num_tx, num_taps), dtype=complex)
        chunk_size = 1000
        
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            size = end_idx - start_idx
            t_chunk = (start_idx + np.arange(size)) * dt
            
            for k, engine in enumerate(tap_engines):
                if engine['active']:
                    # Dynamic sum
                    # (R, P, T) * (R, C) -> (C, P, T)
                    # Einsum: rpt, rc -> cpt
                    # Verify shapes:
                    # W: (R, P, T) -> rpt
                    # phase: (R, C) -> rc
                    phase = np.exp(1j * engine['Omegas'] * t_chunk[np.newaxis, :])
                    full_tensor[start_idx:end_idx, :, :, k] = np.einsum('rpt,rc->cpt', engine['W'], phase)
                else:
                    # Static copy
                    full_tensor[start_idx:end_idx, :, :, k] = engine['val'][np.newaxis, :, :]
                    
        return full_tensor, delays

    def _calculate_scaling_factor(self, sim_result, sorted_taps):
        """Mode B: 数字域施加 pathloss; Mode A: 归一化到峰值, 绝对电平交给
        模拟链路。

        Mode A 链路预算 (audit batch 2, A6 — target_output_power_dbm 此前
        被接受但从未读取): 导出波形峰值归一 (数字增益 digital_boost dB),
        F64 口以 config.target_output_power_dbm 播放。以 0 dBm 等效发射参考,
        DUT 端复现绝对信道所需外部衰减:

            req_att = target_output_power_dbm + pathloss_db + digital_boost

        (target_output_power_dbm=0 时退化为旧公式 pathloss+boost, 旧公式即
        隐含假设 F64 输出 0 dBm。) 结果存入 self.last_mode_a_info 供调用方/
        UI 读取, 不再只打 stdout。
        """
        target_path_loss_db = sim_result.get('pathloss_db', 100.0)
        if self.config.mode == 'B':
            return 10**(-target_path_loss_db / 20.0)
        elif self.config.mode == 'A':
            # Normalize to global peak (across all taps/probes/tx)
            # Estimate peak from static weights to be fast
            max_val = 0.0
            for tap in sorted_taps:
                max_val = max(max_val, np.max(np.abs(tap['coeff'])))

            if max_val > 1e-9:
                scale = 1.0 / max_val
                digital_boost = 20 * np.log10(scale)
                f64_out_dbm = self.config.target_output_power_dbm
                req_att = f64_out_dbm + target_path_loss_db + digital_boost
                self.last_mode_a_info = {
                    'scale': scale,
                    'digital_boost_db': digital_boost,
                    'f64_output_power_dbm': f64_out_dbm,
                    'pathloss_db': target_path_loss_db,
                    'required_attenuation_db': req_att,
                }
                print(f"[Mode A] Scale: {scale:.2f}, F64 out: {f64_out_dbm:.1f} dBm, "
                      f"Req ATT: {req_att:.2f} dB (ref 0 dBm Tx)")
                return scale
        return 1.0

    def _get_calibration_vector(self, num_probes, apply_calibration: bool = True):
        cal = np.ones((num_probes, 1), dtype=complex)
        if self.calibration and apply_calibration:
            for p in range(num_probes):
                cal[p, 0] = self.calibration.get_correction_factor(p + 1)
        return cal

class MFSWRawExporter(PropsimASCIIExporter):
    """
    MIMO-First Standard Waveform (MFSW) Exporter.
    Outputs purely 'dry' baseband I/Q data and delays, completely decoupled
    from any vendor-specific headers (like Keysight's ***** Header *****).
    
    Format:
    Generates a Tab-Separated string without any header rows. 
    Each row is a CIR snapshot. Columns are triples: [Delay (ns)] [Real] [Imag].
    This raw format is then zipped and sent to the Orchestrator's format adapters.
    """
    
    def _generate_asc_content(self, data: np.ndarray, delays: np.ndarray, carrier_frequency_hz: float) -> str:
        """
        Overrides the PropsimASCIIExporter generator.
        Removes all Keysight F64 proprietary string properties.
        """
        num_samples, num_taps = data.shape
        out = io.StringIO()
        
        delay_strs = [f"{d:.9e}" for d in delays]
        
        for t in range(num_samples):
            row = []
            for k in range(num_taps):
                val = data[t, k]
                row.append(delay_strs[k])
                row.append(f"{val.real:.6e}")
                row.append(f"{val.imag:.6e}")
            out.write("\t".join(row) + "\n")
            
        return out.getvalue()
        
    def export_to_zip_memory(self, sim_result: Dict, base_name: str = "mfsw_export", direction: str = "dl") -> str:
        """
        Overrides the zip memory method to use the .mfsw extension instead of .asc.
        """
        import base64
        
        if direction not in ("dl", "ul"):
            raise ValueError(f"Unknown export direction: {direction}")

        if direction == "ul":
            ray_components = sim_result.get('ray_components_ul')
            impulse_response = sim_result.get('channel_impulse_response_ul')
        else:
            ray_components = sim_result.get('ray_components')
            impulse_response = sim_result.get('channel_impulse_response')
            
        if impulse_response is None:
            raise ValueError(f"Missing impulse response for direction '{direction}'.")
            
        apply_calibration = direction != "ul"
        if ray_components:
            data_tensor, delays = self._compute_dynamic_tensor(
                sim_result, impulse_response, ray_components, apply_calibration=apply_calibration
            )
        else:
            data_tensor, delays = self._compute_static_tensor(
                sim_result, impulse_response, apply_calibration=apply_calibration
            )
            
        carrier_frequency_hz = sim_result.get('center_frequency_hz', 3.5e9)
        num_samples, num_effective_probes, num_tx, num_taps = data_tensor.shape
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            # Inject metadata to root to help Orchestrator Formatters
            metadata = f"Samples: {num_samples}\nTaps: {num_taps}\nCarrier_Hz: {carrier_frequency_hz}\nSampleRate_Hz: {self.config.sample_rate_hz}\n"
            zip_file.writestr("mfsw_metadata.txt", metadata)
            
            for tx in range(num_tx):
                for p in range(num_effective_probes):
                    fname = f"{base_name}_In{tx+1}_Out{p+1}.mfsw"
                    link_data = data_tensor[:, p, tx, :]
                    content = self._generate_asc_content(link_data, delays, carrier_frequency_hz)
                    zip_file.writestr(fname, content)
                    
        return base64.b64encode(zip_buffer.getvalue()).decode('utf-8')
