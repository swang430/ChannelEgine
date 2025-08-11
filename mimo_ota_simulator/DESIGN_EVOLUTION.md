# Design Evolution for ChannelEngine (V4.0 Integration)

**Version:** 2.0
**Date:** 2025-08-11

## 1. Goal

To enable the advanced, physically-grounded **Hierarchical Physical Channel Synthesis** method in the `MIMO_OTA_Simulator`. This requires the `ChannelEngine` (specifically the `ChannelSimulator.run()` method) to expose more detailed, per-cluster channel information.

## 2. Core Requirement: Expose Per-Cluster Channel Matrix

The fundamental limitation preventing the advanced synthesis method is that the `clusters` object currently returned by the engine is a list of dictionaries containing only scalar parameters (angle, power, etc.). The complex channel matrix for each cluster (`H_c`) is calculated as a temporary variable inside `_synthesize_channel` and then discarded after being added to the total channel matrix.

To proceed, the `MIMO_OTA_Simulator` **must** have access to this per-cluster channel matrix.

### 2.1. Proposed Data Structure

It is strongly recommended to define a formal Pydantic `BaseModel` for the cluster structure to ensure type safety and clarity. The `clusters` list returned by `ChannelSimulator.run()` should contain instances of this model.

**Proposed `EngineCluster` Pydantic Model:**

```python
# To be defined in channel_model_38901/common/data_models.py

import numpy as np
from pydantic import BaseModel

class EngineCluster(BaseModel):
    # --- Existing Scalar Parameters ---
    delay_s: float
    power_dB: float
    aod_deg: float
    aoa_deg: float
    zod_deg: float
    zoa_deg: float
    departure_azimuth_spread_deg: float
    arrival_azimuth_spread_deg: float
    # ... any other relevant scalar parameters ...

    # --- NEW REQUIRED FIELD ---
    # The full complex channel matrix for this single cluster.
    # This contains the combined effects of transmit array, receive array,
    # and multipath rays within the cluster.
    h_matrix: np.ndarray # Shape: (Nr, Nt)

    class Config:
        arbitrary_types_allowed = True
```

### 2.2. Implementation Guidance

1.  **Location of Change**: The ideal place to capture this information is inside the `_synthesize_channel` method in `channel_model_38901/channel_generator.py`.

2.  **Logic**: Inside the `for n, cluster in enumerate(clusters):` loop, you calculate the correlated channel for the cluster:
    ```python
    H_iid = (np.random.randn(num_rx, num_tx) + 1j * np.random.randn(num_rx, num_tx)) / np.sqrt(2)
    H_corr = L_rx @ H_iid @ L_tx.conj().T
    ```
    The `H_corr` variable (after being multiplied by the Doppler term and any other complex phase terms, but **before** being multiplied by the scalar power `np.sqrt(cluster['power'])`) is the `h_matrix` we need.

3.  **Data Population**: This `h_matrix` should be stored as a new field in the `EngineCluster` object for each cluster before it is returned by the `generate_channel_realization` method.

### 2.3. Rationale for `h_matrix`

Providing the full per-cluster `h_matrix` is the most powerful and flexible approach. With this, the `MIMO_OTA_Simulator` can then derive the necessary transmit-side information (`G_tx` matrix) required for the `A = W_map @ G_tx` formula. For example, it can:

-   In a simple SISO/SIMO case, use the single column of `h_matrix` as the transmit vector.
-   In a MISO/MIMO case, use techniques like Singular Value Decomposition (SVD) on `h_matrix` or simply use its first row to derive a representative transmit vector `h_tx`.

This approach provides the necessary data for the advanced simulation without forcing the `ChannelEngine` to make assumptions about how the data will be used.