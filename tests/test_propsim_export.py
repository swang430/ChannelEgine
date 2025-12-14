
import unittest
import numpy as np
import os
import sys
import csv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig, CalibrationConfig, CalibrationEntry, PropsimExportConfig
from mimo_ota_simulator.exporters import PropsimASCIIExporter

class TestPropsimExport(unittest.TestCase):
    
    def setUp(self):
        # 1. Setup Basic Simulation
        self.chamber = ChamberConfig(
            num_probes=8,
            distribution='ring',
            radius_m=2.0,
            probe_antenna={"gain_dBi": 0.0, "pattern": "Isotropic"}
        )
        
        self.channel = TargetChannelConfig(
            model_name='UMa',
            cluster_model_name='CDL-C', # NLOS
            center_frequency_hz=3.5e9,
            tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
            rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
            force_condition='NLOS'
        )
        
        self.simulator = MIMO_OTA_Simulator()
        
        # Run once to get results
        self.sim_results = self.simulator.run(
            self.chamber, self.channel, synthesis_method='ray'
        )
        
        self.output_file = "test_propsim_output.asc"

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_export_mode_b_scaling(self):
        """Test Mode B (Internal Scaling) applies pathloss directly."""
        print("\n--- Test Mode B Export ---")
        
        export_config = PropsimExportConfig(
            filename=self.output_file,
            mode='B',
            duration_s=0.001, # Short duration
            sample_rate_hz=1000
        )
        
        exporter = PropsimASCIIExporter(export_config)
        exporter.export(self.sim_results)
        
        # Check file content
        data = self._read_ascii_file(self.output_file)
        
        # Check magnitude
        # Pathloss for CDL-C NLOS is typically high (> 100dB).
        # Values should be very small.
        avg_mag = np.mean(np.abs(data))
        print(f"Mode B Avg Magnitude: {avg_mag}")
        self.assertLess(avg_mag, 1e-3, "Mode B output should be small (pathloss applied)")

    def test_export_mode_a_normalization(self):
        """Test Mode A (External ATT) normalizes waveform."""
        print("\n--- Test Mode A Export ---")
        
        export_config = PropsimExportConfig(
            filename=self.output_file,
            mode='A',
            duration_s=0.001,
            sample_rate_hz=1000
        )
        
        exporter = PropsimASCIIExporter(export_config)
        exporter.export(self.sim_results)
        
        # Read as complex numbers to calculate true magnitude
        data_complex = self._read_ascii_file_complex(self.output_file)
        max_mag = np.max(np.abs(data_complex))
        print(f"Mode A Max Magnitude: {max_mag}")
        
        # Relax tolerance to 3 places due to ASCII .6e formatting loss
        self.assertAlmostEqual(max_mag, 1.0, places=3, msg="Mode A output should be normalized to 1.0")

    def test_calibration_predistortion(self):
        """Test calibration logic applies inverse phase."""
        print("\n--- Test Calibration ---")
        
        # Define a calibration that adds 90 deg phase shift to Port 1
        cal_entries = [
            CalibrationEntry(port_id=1, cable_phase_deg=90.0, cable_loss_db=0.0, probe_gain_dbi=0.0)
        ]
        # Other ports default to 0 phase (implicit in logic if not found? No, logic defaults to 1.0+0j)
        # Wait, my logic in CalibrationConfig.get_correction_factor defaults to 1.0 if not found.
        
        cal_config = CalibrationConfig(entries=cal_entries)
        
        export_config = PropsimExportConfig(
            filename=self.output_file,
            mode='A', # Normalized
            duration_s=0.001,
            sample_rate_hz=1000
        )
        
        exporter = PropsimASCIIExporter(export_config, calibration_config=cal_config)
        exporter.export(self.sim_results)
        
        # Get raw (uncalibrated) weight for Probe 1 (Index 0)
        raw_weights = self.sim_results['transmit_mapping_matrix_A']
        raw_val_p1 = raw_weights[0, 0] # Probe 1, Tx 1
        raw_phase = np.angle(raw_val_p1, deg=True)
        
        # Get exported (calibrated) value
        data_complex = self._read_ascii_file_complex(self.output_file) # Shape (Samples, Probes*Tx)
        # Port 1 is the first column pair (Real, Imag)
        exported_val_p1 = data_complex[0, 0] 
        exported_phase = np.angle(exported_val_p1, deg=True)
        
        print(f"Raw Phase P1: {raw_phase:.2f} deg")
        print(f"Exported Phase P1: {exported_phase:.2f} deg")
        print(f"Calibration: Cable adds +90 deg. Pre-distortion should add -90 deg.")
        
        expected_phase = raw_phase - 90.0
        
        # Normalize to -180..180
        diff = (exported_phase - expected_phase + 180) % 360 - 180
        self.assertAlmostEqual(diff, 0.0, delta=1.0, msg="Phase calibration incorrect")

    def test_dynamic_doppler_variance(self):
        """Test that Ray mode export produces time-varying data (Doppler)."""
        print("\n--- Test Dynamic Doppler ---")
        
        # Ensure UE has velocity to generate Doppler
        self.channel.ue_velocity = (10.0, 0, 0) # 10 m/s
        
        # Re-run simulator to get Doppler-ready results
        self.sim_results = self.simulator.run(
            self.chamber, self.channel, synthesis_method='ray'
        )
        
        export_config = PropsimExportConfig(
            filename=self.output_file,
            mode='B',
            duration_s=0.01, # 10ms, enough to see phase rotation
            sample_rate_hz=1000
        )
        
        exporter = PropsimASCIIExporter(export_config)
        exporter.export(self.sim_results)
        
        data_complex = self._read_ascii_file_complex(self.output_file)
        # data_complex shape: (Samples, Ports)
        
        # Check variance over time for the first port
        # If static, variance of real/imag parts should be 0.
        # If dynamic, > 0.
        
        # Check first port
        port0_series = data_complex[:, 0]
        std_real = np.std(np.real(port0_series))
        std_imag = np.std(np.imag(port0_series))
        
        print(f"Std Dev Real: {std_real:.6e}, Imag: {std_imag:.6e}")
        
        self.assertTrue(std_real > 1e-9 or std_imag > 1e-9, "Output should vary over time due to Doppler")

    def _read_ascii_file(self, filename):
        """Helper to read numeric data from ASCII file."""
        values = []
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row or row[0].startswith('!'): continue
                # Skip Time column (index 0), parse rest
                row_vals = [float(x) for x in row[1:]]
                values.extend(row_vals)
        return np.array(values)

    def _read_ascii_file_complex(self, filename):
        """Helper to read complex data from ASCII file. Returns (Samples, Ports)."""
        complex_rows = []
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row or row[0].startswith('!'): continue
                # Skip Time column
                vals = [float(x) for x in row[1:]]
                # Pairs Re, Im
                c_vals = []
                for i in range(0, len(vals), 2):
                    c_vals.append(complex(vals[i], vals[i+1]))
                complex_rows.append(c_vals)
        return np.array(complex_rows)

if __name__ == '__main__':
    unittest.main()
