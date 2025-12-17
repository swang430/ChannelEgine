# 技术附录：从 CDL 参数到 F64 CIR 文件的完整数据流解析

本文档详细拆解了 `MIMO OTA Simulator` 如何将抽象的 3GPP CDL 信道模型参数，一步步转化为 Keysight Propsim F64 硬件可执行的 64 个物理通道文件。这一过程涵盖了从统计建模、空间映射到时域信号合成的全过程。

---

## 1. 数据流全景图 (Data Flow Overview)

整个生成过程可以概括为以下四个核心阶段：

| 阶段 | 输入 (Input) | 核心处理 (Process) | 输出 (Output) | 物理意义 |
| :--- | :--- | :--- | :--- | :--- |
| **L1. 参数定义** | 场景(UMa), 频率, 速度 | 查表 38.901, LSP 随机化 | **物理簇列表 (Clusters)** | 确定信号“从哪里来、多强、多晚” |
| **L2. 空间离散** | 簇角度, 探头几何 | 高斯加权 (Gaussian Weighting) | **探头权重 (Probe Weights)** | 将连续空间信号分配给离散物理探头 |
| **L3. 射线合成** | 权重, Tx天线, 速度 | 射线分裂, 多普勒相位旋转 | **时变信道矩阵 H(t, $\tau$)** | 生成包含时间动态的基带 I/Q 序列 |
| **L4. 硬件映射** | H(t, $\tau$) 张量 | 端口展开 (Unrolling) | **64个 .asc 文件** | 适配 F64 硬件的物理端口定义 |

---

## 2. 环节详解 (Detailed Breakdown)

### L1. 3GPP CDL 模型参数生成 (Model Generation)
这是仿真的起点，定义了信道的统计特性。

*   **操作**: 用户选择 `UMa` (场景) 和 `CDL-C` (簇模型)。
*   **内部处理**:
    1.  **LSP 生成**: 模拟器根据 UMa 的统计分布（对数正态分布），随机生成大尺度参数（LSP），如阴影衰落（Shadow Fading, SF）、K因子、角度扩展（ASD/ASA）。
    2.  **簇缩放**: 加载 CDL-C 标准定义的 23 个簇（Cluster）。根据上一步生成的 LSP，对这 23 个簇的功率、时延和角度进行整体缩放和随机化。
*   **输出数据**: 一个包含 23 个对象的列表，每个对象描述一个簇：
    *   `Delay`: 例如 300ns
    *   `Power`: 例如 -5dB
    *   `AoA`: 例如 45° (到达角)
    *   `AoD`: 例如 10° (出发角)

### L2. 空间映射与探头加权 (Spatial Mapping)
这是 MIMO-OTA 的核心步骤，将“角度”转换为“探头能量”。

*   **输入**: 上一步的 Cluster 列表，以及 16 个双极化探头的几何位置（Ring, Radius=1.5m）。
*   **内部处理**:
    1.  **射线分裂 (Ray Expansion)**: 为了保证多普勒谱的平滑度，每个簇被进一步分裂为 20 条子射线（Sub-rays），分布在簇中心角度附近。
    2.  **权重计算**: 对于每一条射线，计算其 AoA 与每一个探头物理角度的距离。
        *   使用 **PAS (Power Angular Spectrum)** 形状函数（通常为高斯函数）计算权重。
        *   *示例*: 如果一条射线来自 0°，位于 0° 的探头权重接近 1，位于 180° 的探头权重接近 0。
*   **输出数据**: 这里的“角度”信息消失了，转化为了 **Probe Weights 矩阵**。
    *   矩阵维度: `[20 Rays x 23 Clusters, 32 Probe Ports]`。

### L3. 动态信号合成 (Dynamic Synthesis)
这一步将静态的权重转化为随时间变化的波形。

*   **输入**: UE 移动速度 (如 30km/h)，Tx 天线配置 (2 Ant)，采样率 (如 1000Hz)，时长 (1s)。
*   **内部处理**:
    1.  **多普勒计算**: 对于每一条射线，计算其相对于 UE 速度矢量的多普勒频移 $f_d$。
        $$ f_d = \frac{v}{\lambda} \cos(\theta_{ray} - \theta_{v}) $$
    2.  **相位旋转**: 生成时间序列 $t = 0...1s$。计算每一时刻的相位旋转因子 $e^{j 2\pi f_d t}$。
    3.  **相干叠加**: 将所有 20 条子射线在每一个 Tx-Probe 链路上进行复数叠加。
        $$ h_{tx, probe}(t) = \sum_{ray} \sqrt{P_{ray}} \cdot W_{probe, ray} \cdot e^{j (2\pi f_{d, ray} t + \phi_{random})} $$
*   **输出数据**: 一个 4 维张量 (Tensor)。
    *   维度: `[Time Samples, 32 Probes, 2 Tx, Taps]`。
    *   这包含了所有链路在每一毫秒的 I/Q 值。

### L4. 物理文件生成 (File Generation)
最后一步是将内存中的数学张量转换为 F64 认识的文件格式。

*   **逻辑**: F64 的每一个 "Channel" 对应一条物理连线 (Input -> Output)。
*   **输入**: 2 个 Tx 输入，32 个 Probe 输出。
*   **映射规则**:
    *   **Loop 1 (Tx)**: 遍历 Tx1, Tx2
    *   **Loop 2 (Probe)**: 遍历 Probe Port 1 到 32
*   **生成动作**:
    1.  提取张量切片 `Tensor[:, probe_idx, tx_idx, :]`。
    2.  格式化为 ASC 三元组: `Delay \t Real \t Imag`。
    3.  写入 Header: `***** Header *****`, `CIR_Update_Rate`, 等。
*   **最终产物**: 64 个 `.asc` 文件。
    *   `Model_In1_Out1.asc` (Tx1 -> Probe 1 V-pol)
    *   `Model_In1_Out2.asc` (Tx1 -> Probe 1 H-pol)
    *   ...
    *   `Model_In2_Out32.asc` (Tx2 -> Probe 16 H-pol)

---

## 3. 关键变化点检查 (Sanity Check Points)

在验证过程中，您可以依据此流程检查以下特征：

1.  **CDL 到 Probe 的转化**: 
    *   由于 **L2 空间映射**，如果你在 App 中看到 Cluster 集中在 90°，那么生成的 `_OutX.asc` 文件中，只有对应 90° 的探头文件（如 Output 9, 10）会有强能量，其他文件应当接近静默（或只有底噪）。
    
2.  **速度 到 多普勒 的转化**:
    *   由于 **L3 动态合成**，文件中的 Real/Imag 数值会随行数（时间）呈现周期性震荡。震荡频率应严格对应多普勒频移。如果速度设为 0，Real/Imag 应保持恒定。

3.  **文件数量**:
    *   由于 **L4 物理映射**，对于 2x2 MIMO 输入和 16 双极化探头，必须正好生成 2x16x2 = **64** 个文件。少于此数量意味着映射逻辑缺失。
