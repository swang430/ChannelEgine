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

## 4. API与数据流定义 (V1.2)

*   **输入对象**:
    *   `ChamberConfig(num_probes: int, distribution: Literal['ring', 'multi-ring'], radius_m: float, calibration_matrix_B: np.ndarray)`
    *   `TargetChannelConfig(model_name: str, mimo_config: str, ue_speed_kph: float)`

*   **核心类**: `class MIMO_OTA_Simulator`

*   **核心方法**: `run(chamber_config: ChamberConfig, channel_config: TargetChannelConfig) -> Dict`

*   **输出字典**: 
    *   `target_H_matrix: np.ndarray`
    *   `transmit_mapping_matrix: np.ndarray` (即`A`矩阵)
    *   `effective_H_matrix: np.ndarray` (即`B @ A`的结果)
    *   `probe_signals: Dict[int, np.ndarray]`

## 5. 分阶段测试计划 (新增)

我们将遵循测试驱动开发（TDD）的模式，创建一个新的测试文件 `tests/test_mimo_ota_simulator.py`，并按以下阶段进行开发与测试：

### 阶段一：数据模型与配置解析测试
*   **目标**: 确保所有输入（暗室配置、目标信道）和中间数据（信道剖面）都能被正确地创建和解析。
*   **测试用例 1.1 (暗室配置)**: 创建一个`ChamberConfig`对象，并验证其所有属性都被正确设置。
*   **测试用例 1.2 (CDL解析器)**: 创建一个简单的CDL-A模型定义文件作为测试资产。验证`CDL_Parser`能否正确读取该文件，并生成一个包含正确簇数量和参数的`ChannelProfile`对象。

### 阶段二：核心算法单元测试 (簇到探头的映射)
*   **目标**: 在隔离的环境下，对最核心的`A`矩阵（发射端映射矩阵）生成算法进行测试。
*   **测试用例 2.1 (最邻近匹配)**: 在一个简化的4探头暗室中，输入一个来自88度的单簇。断言算法生成的`A`矩阵，应将此簇的能量100%分配给位于90度的那个探头。
*   **测试用例 2.2 (插值合成)**: 在同一个4探头暗室中，输入一个来自45度的单簇。断言`A`矩阵应根据我们设定的插值逻辑，将能量大致平均地分配给0度和90度的两个探头。

### 阶段三：端到端集成与验证测试
*   **目标**: 对完整的`run()`方法和最终的`H_effective`计算进行测试。
*   **测试用例 3.1 (理想暗室验证)**: 在一个理想暗室中（校准矩阵`B`为单位阵），最终的`H_effective`应该等于我们计算出的`A`矩阵（在维度匹配时）。本测试将验证 `H_effective = I @ A`。
*   **测试用例 3.2 (与目标信道对比)**: 在理想暗室中，最终合成的`H_effective`应该非常接近引擎生成的`H_target`。本测试将断言 `H_effective` 和 `H_target` 之间的误差低于某个设定的阈值。