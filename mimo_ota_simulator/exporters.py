
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

    def export(self, sim_result: Dict, output_path: str = None) -> str:
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
        
        ray_components = sim_result.get('ray_components')
        impulse_response = sim_result.get('channel_impulse_response')
        
        # Prepare Data Tensors (Time, Probes, Tx, Taps)
        # We process everything into a unified structure before writing.
        # This simplifies the specific logic for Static vs Dynamic.
        
        if ray_components:
            print("[Propsim Exporter] Generating Wideband Dynamic Doppler data...")
            # Returns: (NumSamples, NumProbes, NumTx, NumTaps) complex, and (NumTaps) delays
            data_tensor, delays = self._compute_dynamic_tensor(sim_result, impulse_response, ray_components)
        else:
            print("[Propsim Exporter] Generating Wideband Static data...")
            data_tensor, delays = self._compute_static_tensor(sim_result, impulse_response)
            
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
                
                self._write_single_asc_file(fpath, link_data, delays)
                generated_files.append(fpath)
                
        return generated_files

    def _write_single_asc_file(self, fpath: str, data: np.ndarray, delays: np.ndarray):
        """
        Writes a single standard .ASC file.
        data: (Samples, Taps) complex
        delays: (Taps) float seconds
        """
        num_samples, num_taps = data.shape
        
        with open(fpath, 'w', newline='') as f:
            # --- Standard Header ---
            f.write("***** Header *****\n")
            f.write(f"{num_samples} CIRs\n")
            f.write(f"{num_taps} Taps/CIR\n")
            f.write(f"{self.config.sample_rate_hz:.1f} CIR_Update_Rate\n")
            # Optional: Carrier Frequency (useful for F64 to know)
            # We assume it's passed or default
            f.write("3500000000.0 Carrier_Frequency\n") # Placeholder or from config
            f.write("***** Tap data *****\n")
            # Note: No column header line in standard ASC data block
            
            # --- Data Body ---
            # Format: Delay Re Im Delay Re Im ...
            # Optimization: Pre-format delays since they are constant
            delay_strs = [f"{d:.9e}" for d in delays]
            
            for t in range(num_samples):
                row = []
                for k in range(num_taps):
                    val = data[t, k]
                    # Delay
                    row.append(delay_strs[k])
                    # Re
                    row.append(f"{val.real:.6e}")
                    # Im
                    row.append(f"{val.imag:.6e}")
                
                f.write("\t".join(row) + "\n")

    def _compute_static_tensor(self, sim_result: Dict, impulse_response: List[Dict]):
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
        cal_vector = self._get_calibration_vector(num_probes) # (P, 1)
        
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

    def _compute_dynamic_tensor(self, sim_result: Dict, impulse_response: List[Dict], ray_components: List[Dict]):
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
        cal_vector = self._get_calibration_vector(num_probes)
        
        # Pre-compute Effective Rays per Tap
        tap_engines = []
        for tap in sorted_taps:
            c_idx = tap['cluster_index']
            rays = cluster_rays_map.get(c_idx, [])
            
            eff_list = []
            for r in rays:
                w = r['W_base'] * np.exp(1j * r['phi_init']) * scaling_factor * cal_vector
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
                # Log info...
                digital_boost = 20 * np.log10(scale)
                req_att = target_path_loss_db + digital_boost
                print(f"[Mode A] Scale: {scale:.2f}, Req ATT: {req_att:.2f} dB")
                return scale
        return 1.0

    def _get_calibration_vector(self, num_probes):
        cal = np.ones((num_probes, 1), dtype=complex)
        if self.calibration:
            for p in range(num_probes):
                cal[p, 0] = self.calibration.get_correction_factor(p + 1)
        return cal
