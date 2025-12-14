
import numpy as np
import os
from typing import Dict, List, Optional
import csv

from .data_models import CalibrationConfig, PropsimExportConfig

class PropsimASCIIExporter:
    """
    Exports MIMO OTA Simulation results to Keysight Propsim F64 ASCII format.
    Handles Calibration (Pre-distortion) and Amplitude Control Modes (A/B).
    
    Supports:
    1. Static Snapshot (Cluster/Pinv modes) - Low computational load.
    2. Doppler-Evolved Time Series (Ray mode) - High fidelity dynamic phase.
    """
    
    def __init__(self, export_config: PropsimExportConfig, calibration_config: Optional[CalibrationConfig] = None):
        self.config = export_config
        self.calibration = calibration_config

    def export(self, sim_result: Dict, output_path: str = None) -> str:
        if output_path is None:
            output_path = self.config.filename
            
        # Check if we have Ray Components for Doppler reconstruction
        ray_components = sim_result.get('ray_components')
        
        if ray_components:
            print("[Propsim Exporter] Ray components found. Generating Doppler-evolved time series...")
            self._export_dynamic_doppler(sim_result, output_path)
        else:
            print("[Propsim Exporter] No ray components. Generating static snapshot series...")
            self._export_static_snapshot(sim_result, output_path)
            
        return output_path

    def _export_static_snapshot(self, sim_result: Dict, filename: str):
        """Original optimized static writer."""
        weights_snapshot = sim_result['transmit_mapping_matrix_A'].copy()
        num_probes, num_tx = weights_snapshot.shape
        
        # Apply Calibration
        if self.calibration:
            for p_idx in range(num_probes):
                correction = self.calibration.get_correction_factor(port_id=p_idx + 1)
                weights_snapshot[p_idx, :] *= correction

        # Apply Amplitude Control
        self._apply_amplitude_control(weights_snapshot, sim_result)
        
        # Write
        self._write_ascii_file_streaming(filename, weights_snapshot, is_static=True)

    def _export_dynamic_doppler(self, sim_result: Dict, filename: str):
        """Reconstructs time series from rays: Sum(W_base * exp(j*phi_init) * exp(j*2pi*fd*t))."""
        ray_components = sim_result['ray_components']
        jitter_vec = sim_result.get('probe_phase_jitter_vector') # Shape (N_probes, 1) or None
        
        # We need to determine the array shape from the first component
        sample_w = ray_components[0]['W_base'] # (N_probes, N_tx)
        num_probes, num_tx = sample_w.shape
        
        # Pre-calculate Calibration Vector
        cal_vector = np.ones((num_probes, 1), dtype=complex)
        if self.calibration:
            for p_idx in range(num_probes):
                cal_vector[p_idx, 0] = self.calibration.get_correction_factor(port_id=p_idx + 1)
        
        # Pre-calculate Amplitude Scaling Factor
        # This is tricky because we construct the signal point-by-point.
        # We need to know the PEAK magnitude to normalize for Mode A.
        # But we don't want to generate the whole array in memory.
        # Strategy:
        # 1. Generate a small subset (e.g. 100 points) to estimate Peak? 
        #    Risk: Peak might happen elsewhere.
        # 2. Or just apply pathloss directly (Mode B) which is safe.
        # 3. For Mode A, we assume the 'Static Snapshot' peak is representative?
        #    Usually yes. Let's use the static snapshot (sum at t=0) to calculate scaling.
        
        # Reconstruct t=0 snapshot to find scaling factor
        snapshot_t0 = np.zeros_like(sample_w)
        for rc in ray_components:
            # A_ray = W_base * exp(j * phi_init)
            snapshot_t0 += rc['W_base'] * np.exp(1j * rc['phi_init'])
        
        if jitter_vec is not None:
            snapshot_t0 *= jitter_vec
            
        scaling_factor = 1.0
        target_path_loss_db = sim_result.get('pathloss_db', 100.0)
        
        if self.config.mode == 'B':
            scaling_factor = 10**(-target_path_loss_db / 20.0)
            print(f"[Mode B] Scaling Factor: {scaling_factor:.4e} (-{target_path_loss_db} dB)")
        elif self.config.mode == 'A':
            max_val = np.max(np.abs(snapshot_t0))
            if max_val > 1e-9:
                scaling_factor = 1.0 / max_val
                digital_boost_db = 20 * np.log10(scaling_factor)
                required_att_db = target_path_loss_db + digital_boost_db
                print(f"[Mode A] Est. Digital Boost: {digital_boost_db:.2f} dB")
                print(f"[Mode A] REQUIRED EXTERNAL ATTENUATION: {required_att_db:.2f} dB")

        # Now we write efficiently
        num_samples = int(self.config.duration_s * self.config.sample_rate_hz)
        dt = 1.0 / self.config.sample_rate_hz
        
        # Optimization: Pre-compute (W_base * exp(j*phi_init) * cal * jitter * scale) for each ray
        # Then only multiply by exp(j*2pi*fd*t) in the loop.
        # Let's call this 'W_effective_static'
        
        effective_rays = []
        for rc in ray_components:
            # W_eff = W_base * exp(j*phi) * cal * jitter * scale
            w_eff = rc['W_base'] * np.exp(1j * rc['phi_init']) * scaling_factor
            # Apply probe-specific corrections (broadcast over Tx dim)
            w_eff *= cal_vector # (N_p, 1) * (N_p, N_tx) -> OK
            if jitter_vec is not None:
                w_eff *= jitter_vec
            
            effective_rays.append({
                'W_eff': w_eff,
                'omega': 2 * np.pi * rc['doppler_hz']
            })
            
        print(f"Writing {num_samples} dynamic samples to {filename}...")
        
        with open(filename, 'w', newline='') as f:
            f.write("! Keysight Propsim F64 ASCII Channel Data (Dynamic Doppler)\n")
            f.write(f"! SampleRate: {self.config.sample_rate_hz}\n")
            
            # Optimization: Group rays by same Doppler? 
            # (Rays in same cluster have similar Doppler, but not identical due to AS).
            # No, just brute force sum. 
            # 20 rays * 20 clusters = 400 complex MACs per sample per link. 
            # 400 * 16 * 2 * 100k = ~1.2 GFLOPS. Python might be slow.
            # But we are I/O bound anyway.
            
            # Vectorized approach for speed:
            # We can process in chunks (e.g. 1000 samples) to use NumPy
            chunk_size = 1000
            
            # Stack all W_eff into tensor: (N_rays, N_probes, N_tx)
            W_stack = np.stack([r['W_eff'] for r in effective_rays]) # (N_rays, P, Tx)
            
            # Fix: Omegas should be (N_rays, 1) for broadcasting with (1, Chunk)
            Omegas = np.array([r['omega'] for r in effective_rays])[:, np.newaxis] # (N_rays, 1)
            
            for start_idx in range(0, num_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, num_samples)
                current_chunk_size = end_idx - start_idx
                
                # Time vector for this chunk
                t_chunk = (start_idx + np.arange(current_chunk_size)) * dt # (Chunk,)
                
                # Phase rotation matrix: (N_rays, Chunk)
                # exp(j * omega * t)
                # Omegas: (N_rays, 1)
                # t_chunk: (1, Chunk) (Broadcasting automatically happens here)
                phase_rot = np.exp(1j * Omegas * t_chunk[np.newaxis, :]) # (N_rays, Chunk)
                
                # We need output: (Chunk, Probes, Tx)
                # Result = Sum_over_rays( W_stack * phase_rot )
                # W_stack: (N_rays, P, Tx)
                # phase_rot: (N_rays, Chunk)
                # Einsum: 'rpt,rc->cpt'
                chunk_data = np.einsum('rpt,rc->cpt', W_stack, phase_rot)
                
                # Write Chunk
                for local_i in range(current_chunk_size):
                    row_str = [f"{t_chunk[local_i]:.9f}"]
                    # Flatten P, Tx
                    for tx in range(num_tx):
                        for rx in range(num_probes):
                            val = chunk_data[local_i, rx, tx]
                            row_str.append(f"{val.real:.6e}")
                            row_str.append(f"{val.imag:.6e}")
                    f.write("\t".join(row_str) + "\n")

        print(f"Export Complete: {filename}")

    def _apply_amplitude_control(self, weights: np.ndarray, sim_result: Dict):
        """Helper for static mode amplitude."""
        target_path_loss_db = sim_result.get('pathloss_db', 100.0)
        
        if self.config.mode == 'B':
            scaling = 10**(-target_path_loss_db / 20.0)
            weights *= scaling
            print(f"[Mode B] Applied Pathloss: {target_path_loss_db} dB")
        elif self.config.mode == 'A':
            max_val = np.max(np.abs(weights))
            if max_val > 1e-9:
                weights *= (1.0 / max_val)
                print(f"[Mode A] Normalized. Max peak: {max_val:.4e}")

    def _write_ascii_file_streaming(self, filename: str, weights_snapshot: np.ndarray, is_static: bool = False):
        """Legacy static writer."""
        # ... (Same as before, simplified) ...
        num_samples = int(self.config.duration_s * self.config.sample_rate_hz)
        dt = 1.0 / self.config.sample_rate_hz
        num_probes, num_tx = weights_snapshot.shape
        
        row_str_parts = []
        for tx in range(num_tx):
            for rx in range(num_probes):
                val = weights_snapshot[rx, tx]
                row_str_parts.append(f"{val.real:.6e}")
                row_str_parts.append(f"{val.imag:.6e}")
        static_row = "\t".join(row_str_parts)
        
        with open(filename, 'w', newline='') as f:
            f.write("! Keysight Propsim F64 ASCII Channel Data (Static)\n")
            f.write(f"! SampleRate: {self.config.sample_rate_hz}\n")
            for t_idx in range(num_samples):
                f.write(f"{t_idx * dt:.9f}\t{static_row}\n")
        print(f"Export Complete: {filename}")
