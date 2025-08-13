# MIMO-OTA Simulator Integration Guide

This document provides guidance for developers of the `MIMO_OTA_Simulator` on how to use the core `channel-model-38901` library.

## 1. Core Concept: Hybrid Channel Models

As of version 1.0.0, the `channel-model-38901` library supports a **hybrid model** architecture. This is the recommended way to generate channels for conformance testing.

The key idea is to decouple the **Large-Scale Environment** (which determines pathloss and large-scale fading) from the **Small-Scale Cluster Model** (which determines the multipath cluster characteristics).

- **Large-Scale Environment**: Selected via the `scenario_name` parameter. This should be a stochastic model like `UMa`, `RMa`, etc. It is responsible for calculating pathloss and generating Large-Scale Parameters (LSPs) like shadow fading, delay spread, and angle spreads.

- **Small-Scale Cluster Model**: Selected via the new `cluster_model_name` parameter. This can be:
    - `Stochastic`: Clusters are generated randomly based on the LSPs from the large-scale environment.
    - `CDL-A`, `CDL-B`, `SCME`, etc.: The fixed, standardized cluster parameters (delays, powers, angles) from these models are loaded directly. 

The final channel is a combination of both parts, providing a realistic channel where, for example, a fixed CDL-A model is subject to the pathloss and shadow fading of a UMa environment.

## 2. API Usage Example

The `ChannelSimulator` class in the core library is the main entry point. Its constructor has been updated to support the hybrid model.

### Legacy Invocation (Not Recommended)

Previously, deterministic models were treated as standalone scenarios. This is no longer the recommended approach.

```python
# Old way: The CDL model was isolated from any large-scale fading.
# LSPs and pathloss would be zero.
simulator = ChannelSimulator(scenario_name='CDL-A', center_frequency_hz=3.5e9)
```

### New Hybrid Model Invocation (Recommended)

To generate a channel that combines a UMa environment with CDL-A clusters, you should now do the following:

```python
# New way: Explicitly define the environment and the cluster model.
simulator = ChannelSimulator(
    scenario_name='UMa', 
    center_frequency_hz=3.5e9,
    cluster_model_name='CDL-A'
)

# When run, this simulator will:
# 1. Calculate pathloss for the UMa scenario.
# 2. Generate UMa-specific LSPs (shadow fading, etc.).
# 3. Load the fixed cluster definitions for CDL-A.
# 4. Combine them into the final channel matrix.
results = simulator.run()
```

This approach provides a much more physically meaningful and flexible simulation.
