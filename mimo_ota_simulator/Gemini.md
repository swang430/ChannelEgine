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

我们的最终目标，就是通过精心计算`A`，使得最终合成的`H_effective`尽可能地接近`H_target`。

## 3. 核心技术挑战与设计方案

### 3.1 簇到探头的映射 (Cluster-to-Probe Mapping)
*   **设计方案**: 采用“角度加权合成”策略。

### 3.2 探头复用与虚拟化 (Probe Reuse & Virtualization)
*   **设计方案**: 采用“功率筛选”策略。

### 3.3 校准准备 (Preparation for Calibration)
*   **设计方案**: 引入可加载的**校准矩阵 `B_cal`** 进行预失真处理。

### 3.4 信道模型接口 (Channel Model Interface)
*   **设计方案**: 设计通用的**`ChannelProfile`数据类**，并为不同信道模型（CDL, SCME等）提供独立的**解析器 (Parser)**。

### 3.5 仿真结果验证与输出
*   **设计方案**: 
    *   **必须输出：等效H矩阵 (`H_effective`)**，用于与`H_target`对比，验证仿真精度。
    *   **不予实现：反推物理参数**，因其属于复杂的信道估计问题。

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

## 5. 分阶段测试计划 (V1)

(已完成)

## 6. 图形用户界面 (GUI) 设计方案 (V1)

(已完成)

## 7. V2.0: 全功能引擎前端设计方案

### 7.1 目标

将 `MIMO_OTA_Simulator` 及其GUI升级为一个功能全面的 `ChannelEgine` 前端，忠实、完整地暴露其所有核心配置能力，并为高级MIMO仿真提供支持。

### 7.2 功能特性映射

GUI和模拟器核心将支持以下三个层面的完整配置：

#### A. 场景与几何布局
- **支持的场景**: 提供包含以下所有选项的下拉菜单：`UMa`, `UMi-StreetCanyon`, `UMi-OpenArea`, `RMa`, `InH-Office`, `SMa`, `InF`。
- **基站与用户位置**: 提供三维向量输入框，用于精确设置BS和UE的 (x, y, z) 坐标。
- **用户速度**: 提供三维向量输入框，用于设置UE的速度矢量 (vx, vy, vz)。
- **中心频率**: (已实现)

#### B. 天线阵列配置 (MIMO)
为 **Tx（发射端）** 和 **Rx（接收端）** 提供完全独立的、对称的配置选项。
- **阵列类型**: 下拉选择 `ULA` (均匀线性阵) 或 `URA` (均匀矩形阵)。
- **阵列尺寸**: 数字输入框，分别设置**行数 (Rows)** 和 **列数 (Cols)**。
- **天线间距**: 浮点数输入框，分别设置**水平间距**和**垂直间距**（以波长倍数计）。

#### C. 信道状态控制
- **视距状态 (LOS/NLOS)**: 下拉菜单提供三个选项：`Default` (由模型自行判断), `Force LOS` (强制视距), `Force NLOS` (强制非视距)。
- **随机种子 (Random Seed)**: 数字输入框，用于固定随机数生成器的种子，以保证仿真结果的可复现性。

### 7.3 V2.0 实施计划

(已完成)

## 8. V4.0: 分层式物理信道合成架构

### 8.1 核心判断：转向物理簇合成

基于对AOA/AS映射和硬件资源复用的深入讨论，我们决定彻底放弃矩阵伪逆(`pinv`)的方案。`pinv`方法无法对信道的物理构成（如簇的角度、扩展等）进行精细控制。

因此，项目V4.0阶段的核心，是转向并深化**“角度加权合成”**方案，构建一个能完整反映真实物理过程的仿真器。

### 8.2 V4.0 系统架构

系统将采用“分层式”的物理合成模型，流程分为三步：

**第一步: 目标物理信道定义 (Target Channel Definition)**
*   **输入**: 用户通过GUI定义一个高层级的“目标物理信道”，例如一个标准的3GPP UMa信道，该信道包含`N`个物理簇，每个簇都拥有各自的中心到达角（AOA）和角度扩展（AS）等参数。

**第二步: 物理到硬件的映射 (Physical-to-Hardware Mapping)**
*   **核心任务**: **资源分配与AS模拟**。
*   **逻辑**: 
    1.  系统拥有一个资源池：信道仿真器提供的 **24个逻辑簇**。
    2.  系统遍历`N`个物理簇，读取每个簇的**角度扩展（AS）**参数。
    3.  基于AS的大小，系统决定调用**`M`个**硬件逻辑簇来共同模拟这一个物理簇的能量扩散（例如，使用高斯功率分布）。
*   **输出**: 一个由最多24个“**硬件簇**”构成的列表。列表中的每一个硬件簇，都只有一个**精确的AOA**（没有AS）和一个具体的**功率等级**。

**第三步: 硬件到探头的映射 (Hardware-to-Probe Mapping)**
*   **核心任务**: **计算A矩阵与AOA模拟**。
*   **逻辑**: 
    1.  系统遍历所有“硬件簇”。
    2.  基于每个硬件簇的精确AOA，计算出一组探头权重，通过**2个或多个**物理探头的加权激励，在空间中“虚拟”出这个精确的到达角度。
    3.  最终的 `A` 矩阵，是所有硬件簇的探头权重，根据各自的功率和相位进行叠加后的总和。

### 8.3 V4.0 实施计划

1.  **阶段一 (引擎改造)**: **(用户)** 在 `ChannelEgine` 中，按照 `DESIGN_EVOLUTION.md` 的描述，将每个簇的复数增益矩阵 `h_matrix` 暴露出来。这是实现新方案的**绝对前提**。
2.  **阶段二 (AOA映射核心)**: **(Gemini)** 在引擎改造完成后，实现**第三步**的核心功能：用多个探头虚拟化一个精确AOA。
3.  **阶段三 (AS映射与资源管理)**: **(Gemini)** 实现**第二步**的功能，完成AS到多个硬件簇的映射，以及对24个逻辑簇的资源管理。
4.  **阶段四 (GUI升级)**: **(Gemini)** 升级GUI，以支持新的AOA和AS参数输入和可视化。

