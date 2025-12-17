import unittest
import numpy as np
import os
import sys
import glob

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
            cluster_model_name='CDL-C', # NLOS (Has many clusters)
            center_frequency_hz=3.5e9,
            tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
            rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
            force_condition='NLOS'
        )
        
        self.simulator = MIMO_OTA_Simulator()
        self.base_output_name = "test_propsim_output.asc"

    def tearDown(self):
        # Clean up all generated ASC files
        base_name = os.path.splitext(self.base_output_name)[0]
        files = glob.glob(f"{base_name}*.asc")
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

    def test_wideband_static_export(self):
        """Test Wideband Static Export (Cluster Mode)."""
        print("\n--- Test Wideband Static (Cluster Mode) ---")
        
        sim_results = self.simulator.run(self.chamber, self.channel, synthesis_method='cluster')
        
        export_config = PropsimExportConfig(
            filename=self.base_output_name,
            mode='B',
            duration_s=0.001,
            sample_rate_hz=1000
        )
        
        exporter = PropsimASCIIExporter(export_config)
        # This will generate multiple files: test_propsim_output_Tx1_Probe1.asc, etc.
        generated_files = exporter.export(sim_results)
        first_file = generated_files[0]
        
        print(f"Exporter returned {len(generated_files)} files. First: {first_file}")
        self.assertTrue(os.path.exists(first_file), "Main export file not found")
        
        # Check Total File Count
        # 1 Tx * 8 Probes (Single Pol) = 8 Files. 
        # (Note: In this test setup, dual_pol is default False in ChamberConfig unless specified. 
        # ChamberConfig default is dual_polarized=False? Let's check simulator.py defaults. 
        # Actually in test_propsim_export.py setup, we defined num_probes=8. 
        # If dual_pol is false, then 8 files. If true, 16.
        # Let's check sim_results shape.
        # But wait, the Exporter uses whatever is in 'transmit_mapping_matrix_A'.
        # In test_wideband_static_export, synthesis_method='cluster' uses ChamberConfig.
        
        # Let's just check the file we expect: _In1_Out1.asc
        expected_file = self.base_output_name.replace(".asc", "_In1_Out1.asc")
        self.assertTrue(os.path.exists(expected_file), f"Standard file not found: {expected_file}")
        
        header, data = self._read_standard_asc_file(expected_file)
        
        # Check Header
        num_taps = int(header.get('Taps/CIR', 0))
        num_cirs = int(header.get('CIRs', 0))
        print(f"Detected Header: {num_cirs} CIRs, {num_taps} Taps/CIR")
        
        self.assertGreater(num_taps, 1, "CDL-C should have multiple taps")
        self.assertEqual(num_cirs, 1, "Duration 0.001s at 1000Hz should give 1 sample (or very few)")
        
        # Check Data Dimensions
        # Standard Format: Delay Re Im Delay Re Im ...
        # Cols = 3 * NumTaps
        expected_cols = 3 * num_taps
        self.assertEqual(data.shape[1], expected_cols, "Column count mismatch (Standard Format)")

    def test_wideband_dynamic_export(self):
        """Test Wideband Dynamic Export (Ray Mode)."""
        print("\n--- Test Wideband Dynamic (Ray Mode) ---")
        
        # Enable velocity for Doppler
        self.channel.ue_velocity = (10.0, 0, 0)
        sim_results = self.simulator.run(self.chamber, self.channel, synthesis_method='ray')
        
        export_config = PropsimExportConfig(
            filename=self.base_output_name,
            mode='A',
            duration_s=0.01,
            sample_rate_hz=1000 # 10 samples
        )
        
        exporter = PropsimASCIIExporter(export_config)
        exporter.export(sim_results)
        
        expected_file = self.base_output_name.replace(".asc", "_In1_Out1.asc")
        header, data = self._read_standard_asc_file(expected_file)
        
        num_cirs = int(header.get('CIRs', 0))
        self.assertEqual(num_cirs, 10, "Should have 10 CIRs")
        
        # Check Variance (Dynamic)
        # Scan all Re/Im columns for any variance
        # Cols: Delay, Re, Im, Delay, Re, Im...
        # Indices: 1, 2, 4, 5, ...
        re_indices = range(1, data.shape[1], 3)
        im_indices = range(2, data.shape[1], 3)
        
        all_re = data[:, re_indices]
        all_im = data[:, im_indices]
        
        std_re_max = np.max(np.std(all_re, axis=0))
        std_im_max = np.max(np.std(all_im, axis=0))
        
        print(f"Max Tap Std Dev: Re={std_re_max:.4e}, Im={std_im_max:.4e}")
        
        self.assertTrue(std_re_max > 1e-9 or std_im_max > 1e-9, "At least one tap should be time-varying in dynamic mode")

    def _read_standard_asc_file(self, filename):
        """Parses Standard Keysight Propsim .ASC file."""
        header = {}
        data = []
        in_header = False
        in_data = False
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                if "***** Header *****" in line:
                    in_header = True
                    continue
                if "***** Tap data *****" in line:
                    in_header = False
                    in_data = True
                    continue
                    
                if in_header:
                    # Format: Value Key
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        val, key = parts
                        header[key.strip()] = val.strip()
                
                elif in_data:
                    # Tab separated values
                    parts = line.split('\t')
                    if parts:
                        try:
                            data.append([float(x) for x in parts])
                        except ValueError:
                            pass # Skip malformed lines
                            
        return header, np.array(data)

if __name__ == '__main__':
    unittest.main()
