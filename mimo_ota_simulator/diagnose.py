import inspect
import sys
import os

# Add the parent directory to the path to allow importing ChannelEgine
# This mimics the effect of `pip install -e` for this script
# Note: We assume this script is run from the 'mimo_ota_simulator' directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

print(f"--- Temporarily added to sys.path: {parent_dir} ---")

try:
    print("--- Attempting to import ChannelSimulator from channel_model_38901.simulator ---")
    from channel_model_38901.simulator import ChannelSimulator
    print("--- Import successful ---")
    
    print("\n--- Source code of ChannelSimulator.__init__ as seen by Python ---")
    source_code = inspect.getsource(ChannelSimulator.__init__)
    print(source_code)
    print("-------------------------------------------------------------")
    
except ImportError as e:
    print(f"\n--- FAILED TO IMPORT ---")
    print(f"Error: {e}")
    print(f"\nPython is searching for modules in these paths:")
    for p in sys.path:
        print(p)
        
except Exception as e:
    print(f"\n--- An unexpected error occurred ---")
    print(e)
