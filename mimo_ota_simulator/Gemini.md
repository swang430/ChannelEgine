# MIMO OTA 仿真器设计规格书 (Gemini.md)

## 1. 项目概述与目标

**项目目标**: 创建一个名为 `MIMO_OTA_Simulator` 的高性能Python应用。该应用将利用我们已开发完成的 `ChannelSimulator` 作为核心信道引擎，驱动一个MIMO OTA（Over-the-Air）测试暗室，用于对物理设备（DUT）进行可重复的、高保真度的性能测试。

**核心目的**: 根据用户输入的一组**暗室物理参数**和**目标信道参数**，计算出需要馈入到暗室中每个探头天线的激励信号，并提供分解后的核心矩阵，以供分析和验证。

## 2. 系统架构与核心关系

本应用的设计严格遵循“引擎与应用分离”的哲学。其核心是**将OTA信道合成问题，分解为三个关键矩阵的运算**。

*   `H_target (Nr x Nt)`: 我们想要在暗室中复现的**目标信道矩阵**。
*   `B (Nr x Np)`: **暗室传递矩阵 (Chamber Transfer Matrix)**。
*   `A (Np x Nt)`: **发射端映射矩阵 (Transmit Mapping Matrix)**。

**核心公式**: `H_effective (Nr x Nt) = B (Nr x Np) @ A (Np x Nt)`

**核心工作流程**: 

*   **阶段一：目标信道剖面生成 (由 `ChannelSimulator` 引擎完成)**
    1.  `MIMO_OTA_Simulator` 接收到用户的上层请求，例如：“在16探头的环形暗室中，复现一个CDL-C，4x2 MIMO，UE速度30km/h的信道”。
    2.  它将请求中的信道部分打包，直接调用 `ChannelSimulator` 的API。
    3.  `ChannelSimulator` 引擎运行完整的物理仿真，生成并返回一个描述目标信道特性、包含了**每个簇的`h_matrix`**的`EngineCluster`对象列表（即`ChannelProfile`的核心）。这个`EngineCluster`列表就是我们要复现的目标信道的“数字配方”。

*   **阶段二：空间映射与信号合成 (由 `MIMO_OTA_Simulator` 应用完成)**
    1.  `MIMO_OTA_Simulator` 获取到这个“数字配方” (`ChannelProfile`)。
    2.  它根据用户输入的**暗室参数** (`ChamberConfig`)，执行其独有的OTA空间映射算法，将`EngineCluster`列表中的每一个簇，分配到具体的物理探头上。
    3.  最终，它计算出每个探头应该馈入的信号，完成空间“渲染”。

## 3. 核心技术挑战与设计方案

### 3.1 簇到探头的映射 (Cluster-to-Probe Mapping)
*   **设计方案**: 采用“角度加权合成”策略。

### 3.2 探头复用与虚拟化 (Probe Reuse & Virtualization)
*   **设计方案**: 采用“功率筛选”策略。

### 3.3 校准准备 (Preparation for Calibration)
*   **设计方案**: 引入可加载的**校准矩阵 `B_cal`** 进行预失真处理。

### 3.4 信道模型接口 (Channel Model Interface)
*   **设计方案**: 设计一个通用的**`ChannelProfile`数据类**（其`clusters`字段为`List[EngineCluster]`）。我们将提供独立的**解析器 (Parser)** 来适配不同的信道模型，如`CDL_Parser`, `SCME_Parser`等。

### 3.5 仿真结果验证与输出
*   **设计方案**: 
    *   **必须输出：等效H矩阵 (`H_effective`)**。
    *   **不予实现：反推物理参数**。

## 4. V4.1: 最终系统设计与GUI

### 4.1 核心架构：分层式物理信道合成

我们最终采用“分层式”的物理合成模型，以精确地模拟真实世界的物理过程。

**第一步: 目标物理信道定义**
*   用户通过GUI定义一个高层级的“目标物理信道”（如3GPP UMa），信道引擎 (`ChannelEngine`) 负责生成一组带有角度扩展（AS）的**物理簇**。

**第二步: 物理到硬件的映射**
*   `MIMO_OTA_Simulator` 将物理簇映射为一组（最多24个）没有AS的**硬件簇**。此过程将模拟AS，并管理信道仿真器的硬件资源。

**第三步: 硬件到探头的映射**
*   `MIMO_OTA_Simulator` 将硬件簇的精确AOA，映射为一组探头权重，通过多个物理探头的加权激励，在空间中“虚拟”出该AOA。

### 4.2 最终数据模型

*   **`ChamberConfig`**:
    *   `num_probes: int`: 物理探头的数量。
    *   `dual_polarized: bool`: 探头是否为双极化。
    *   `distribution: Literal['ring']`: 探头分布。
    *   `radius_m: float`: 探头环的半径。
    *   `calibration_matrix_B: np.ndarray`: 暗室校准矩阵，形状为 `(Nr, Np_eff)`。

*   **`TargetChannelConfig`**:
    *   包含所有V2.0版本的详细配置，如场景、几何、天线阵列、信道状态等。

### 4.3 核心算法

*   **`_calculate_probe_weights_for_aoa`**: 根据AOA，使用环形空间中的线性插值算法，计算探头权重。
*   **`_get_g_tx_from_h_matrix`**: 从每个簇的 `h_matrix` 中，提取出发射端的信道分量。
*   **`_map_physical_to_hardware_clusters`**: 将物理簇映射为硬件簇，并根据功率进行排序和筛选。

### 4.4 GUI 功能

最终的GUI (`app.py`) 提供了对所有关键参数的完全控制：
*   **场景与几何**: 模型、频率、速度等。
*   **暗室配置**: 物理探头数、半径、双极化开关。
*   **天线配置**: Tx和Rx的独立配置。
*   **运行控制**:
    *   **合成方法**: 可在 `pinv` 和 `cluster` 之间选择。
    *   **最大硬件簇**: 控制用于合成的硬件簇数量。
    *   LOS/NLOS强制、随机种子等。
*   **可视化**:
    *   A矩阵的热力图。
    *   探头与簇的极坐标图。
    *   原始矩阵数据（幅度和相位）。

## 5. 测试与验证策略

一个未经严格验证的仿真器是不可信的。我们实施了一个**3层验证策略**来确保模型的正确性、稳定性和准确性。所有测试均已实现并通过。

### 5.1 单元测试 (Unit Testing)
*   **目标**: 从数学上确保最底层的、独立的算法单元的精确性。
*   **实现**: `tests/test_simulator_unit.py`
*   **方法**: 重点测试 `_calculate_probe_weights_for_aoa` 方法。测试用例覆盖了常规、边界和双极化情况，通过断言确保空间插值算法的输出与理论预期完全一致。

### 5.2 健全性检查 (Sanity Checks)
*   **目标**: 确保仿真器的核心 `run` 方法在与底层信道引擎交互时，参数传递正确，且输出的数据维度符合预期。
*   **实现**: `tests/test_mimo_ota_simulator.py`
*   **方法**: 利用 `pytest-mock` 库“模拟”（Mock）`channel_model_38901` 的 `ChannelGenerator`。测试不仅验证 `run` 方法返回矩阵的形状，还通过检查对模拟对象的调用参数，确保了从顶层配置到底层引擎的参数传递是准确无误的。

### 5.3 回归测试 (Regression Testing)
*   **目标**: 创建一个“黄金标准”输出文件。在代码发生变更后，可通过对比新的输出与黄金标准是否一致，来防止引入意料之外的错误。
*   **实现**: `create_golden_file.py` (生成脚本), `tests/test_mimo_ota_simulator.py` (测试用例), `tests/golden_results.npz` (数据文件)
*   **方法**: 使用固定的输入参数（包含随机种子）生成一个“黄金”结果文件。测试用例会使用相同的参数运行仿真，并断言其输出与黄金文件中的数据在允许的误差范围内完全一致，从而建立了一个可靠的代码“安全网”。 
