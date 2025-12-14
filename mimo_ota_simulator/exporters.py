
import numpy as np
import os
from typing import Dict, List, Optional
import csv

from .data_models import CalibrationConfig, PropsimExportConfig

class PropsimASCIIExporter:
    """
    Exports MIMO OTA Simulation results to Keysight Propsim F64 ASCII format.
    Handles Calibration (Pre-distortion) and Amplitude Control Modes (A/B).
    """
    
    def __init__(self, export_config: PropsimExportConfig, calibration_config: Optional[CalibrationConfig] = None):
        self.config = export_config
        self.calibration = calibration_config

    def export(self, sim_result: Dict, output_path: str = None) -> str:
        """
        Main export function.
        
        Args:
            sim_result: Dictionary returned by MIMO_OTA_Simulator.run()
                        Must contain 'transmit_mapping_matrix_A' (The Probe Weights).
            output_path: Optional override for output filename.
            
        Returns:
            Path to the generated file.
        """
        # 1. Get Base Weights (Snapshot)
        # Shape: (Num_Probes, Num_Tx)
        # F64 expects impulse responses. In our narrowband assumption, 
        # these complex weights ARE the single-tap impulse response coefficients.
        weights_snapshot = sim_result['transmit_mapping_matrix_A'] 
        num_probes, num_tx = weights_snapshot.shape
        
        # 2. Time Expansion
        # Generate time vector
        num_samples = int(self.config.duration_s * self.config.sample_rate_hz)
        time_vector = np.linspace(0, self.config.duration_s, num_samples, endpoint=False)
        
        # For now, we assume STATIC channel (frozen snapshot) extended over time.
        # Future: Apply Doppler rotation here based on ray angles.
        # weights_series shape: (Num_Samples, Num_Probes, Num_Tx)
        weights_series = np.tile(weights_snapshot[np.newaxis, :, :], (num_samples, 1, 1))
        
        # 3. Apply Calibration (Pre-distortion)
        if self.calibration:
            for p_idx in range(num_probes):
                # Port ID is 1-based usually
                correction = self.calibration.get_correction_factor(port_id=p_idx + 1)
                weights_series[:, p_idx, :] *= correction

        # 4. Amplitude Control Logic
        # Calculate System Path Loss that needs to be compensated
        # This is the "Software" part of the link budget.
        target_path_loss_db = sim_result.get('pathloss_db', 100.0) # Default if missing
        
        # Current weights might already include some scaling from the simulator?
        # Let's check simulator.py: 
        # A_cluster = sqrt(power) * w * g_tx. 
        # 'power' in simulator is typically normalized (sum = 1) or relative to 0dB?
        # TR 38.901 model usually outputs pathloss separately. 
        # The 'clusters' power usually sums to 1 (linear) if normalized, 
        # OR sums to K-factor scaled power.
        # BUT MIMO_OTA_Simulator does NOT apply pathloss_db to 'transmit_mapping_matrix_A'.
        # It calculates relative weights.
        # So we MUST apply the large scale pathloss here.
        
        system_gain_linear = 10**(-target_path_loss_db / 20.0) # Voltage gain
        
        if self.config.mode == 'B':
            # Mode B: Internal Scaling
            # Apply full pathloss to digital data
            print(f"[Mode B] Applying Pathloss {target_path_loss_db:.2f} dB to digital waveform.")
            weights_series *= system_gain_linear
            
        elif self.config.mode == 'A':
            # Mode A: External ATT
            # We keep digital data HOT (Full Scale). 
            # We assume F64 output is at self.config.target_output_power_dbm (e.g. -10 dBm)
            # The receiver expects Pin = Ptx - PL.
            # So we rely on external ATT to drop the power.
            # We DO NOT multiply by system_gain_linear.
            # Instead we calculate the required attenuation and print it.
            
            # Note: We should normalize weights so max peak is 1.0 (Full Scale DAC)
            max_val = np.max(np.abs(weights_series))
            if max_val > 1e-9:
                scaling_factor = 1.0 / max_val
                weights_series *= scaling_factor
                # The 'gain' we added by normalizing needs to be removed by ATT too.
                # Added Gain = 20 * log10(scaling_factor)
                # Original needed gain = -Pathloss
                # Total required ATT = Pathloss + 20*log10(scaling_factor)
                
                # Wait, let's look at power budget:
                # P_out_f64 (fixed) = -10 dBm
                # P_rx_target = -10 - Pathloss
                # Actual P_out_digital = P_out_f64 (since we normalized to 1.0 ~= 0dBFS)
                # We need P_rx = P_out_digital - L_external
                # So L_external (ATT) = Pathloss
                # PLUS the scaling we did? 
                # Yes. If original signal was 0.001 (-60dB) and we boosted to 1.0 (0dB),
                # we added 60dB gain. We need to attenuate that back.
                
                digital_boost_db = 20 * np.log10(scaling_factor)
                required_att_db = target_path_loss_db + digital_boost_db
                
                print(f"[Mode A] Generated Full-Scale Waveform.")
                print(f"[Mode A] Target PL: {target_path_loss_db:.2f} dB")
                print(f"[Mode A] Digital Boost applied: {digital_boost_db:.2f} dB")
                print(f"[Mode A] REQUIRED EXTERNAL ATTENUATION: {required_att_db:.2f} dB")
                
                # Save this info to a sidecar file later?
        
        # 5. Write to File
        if output_path is None:
            output_path = self.config.filename
            
        self._write_ascii_file(output_path, time_vector, weights_series)
        return output_path

    def _write_ascii_file(self, filename: str, time_vec: np.ndarray, weights: np.ndarray):
        """
        Writes data in columns: Time, Re_1, Im_1, Re_2, Im_2 ...
        weights shape: (Time, Probes, Tx)
        We flatten Probes/Tx. Order usually: (Probe1_Tx1, Probe2_Tx1, ..., Probe1_Tx2...)
        Or usually F64 takes (OutputPort1, OutputPort2...) where OutputPort maps to Probe.
        If MIMO, F64 usually handles Tx mapping internally if using GCM, 
        BUT for "Custom Impulse Response", we define the channel h_ij.
        
        Format typically:
        Time    Port1_Real  Port1_Imag  Port2_Real  Port2_Imag ...
        
        Wait, for MIMO (Multiple Tx), a single file usually defines ONE impulse response 
        matrix if it's a "MIMO Channel" file, OR we output separate files for each Tx?
        
        Keysight Propsim ASCII format usually supports multiple links in columns.
        Let's assume columns are:
        Time, h_11_re, h_11_im, h_21_re, h_21_im ... (Rx_Tx order)
        """
        
        num_samples, num_probes, num_tx = weights.shape
        
        with open(filename, 'w', newline='') as f:
            # Header (Optional, often skipped or specific comments)
            f.write("! Keysight Propsim F64 ASCII Channel Data\n")
            f.write(f"! SampleRate: {self.config.sample_rate_hz}\n")
            f.write("! Columns: Time, (Re, Im) for each Probe-Tx pair\n")
            
            writer = csv.writer(f, delimiter='\t') # Tab delimited is common for instruments
            
            for t_idx in range(num_samples):
                row = [f"{time_vec[t_idx]:.9f}"]
                
                # Loop Tx then Probes? Or Probes then Tx?
                # Usually link by link.
                # Let's assume Probe (Rx) is inner loop, Tx is outer? 
                # Or just flat list of all Links.
                # Let's write standard: Rx1_Tx1, Rx2_Tx1, ..., Rx1_Tx2...
                for tx in range(num_tx):
                    for rx in range(num_probes):
                        val = weights[t_idx, rx, tx]
                        row.append(f"{val.real:.6e}")
                        row.append(f"{val.imag:.6e}")
                
                writer.writerow(row)
                
        print(f"Written {num_samples} samples to {filename}")
