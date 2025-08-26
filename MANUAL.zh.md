# 3GPP TR 38.901 信道模型仿真器 - 用户手册

## 1. 项目概述

本项目是一个基于Python的高性能信道模型仿真器，精确实现了3GPP TR 38.901 V19.0.0标准中定义的5G无线信道模型。它采用模块化设计，将核心信道引擎 (`channel_model_38901`) 与上层应用（如`mimo_ota_simulator`）分离，确保了代码的可重用性和可扩展性。

## 2. 功能特性

- **严格遵循标准**: 精确实现3GPP TR 38.901中定义的各主要场景，如 `UMa`, `UMi`, `RMa`, `InH` 等。
- **混合信道模型**: 允许将随机的大尺度环境（决定路径损耗和阴影衰落）与确定性的小尺度簇模型（如 `CDL-A`, `CDL-B`）相结合，生成物理意义更完备的信道。
- **确定性LSP**: 支持使用大尺度参数（LSP）的统计中位值，以消除随机性，便于进行可重复的对比测试。
- **简洁的API**: 通过 `ChannelSimulator` 外观类提供简单易用的编程接口。
- **图形用户界面 (GUI)**: 提供一个基于Streamlit的GUI (`gui.py`)，用于快速执行仿真和结果可视化。

## 3. 安装指南

1.  **克隆仓库**:
    ```bash
    git clone <repository_url>
    ```
2.  **创建并激活Python虚拟环境**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **安装依赖包**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **以可编辑模式安装本地包**:
    这一步是必需的，它能确保上层应用可以正确调用核心库。
    ```bash
    pip install -e ./channel_model_38901
    pip install -e ./mimo_ota_simulator
    ```

## 4. 快速入门

### 4.1 使用GUI

最简单的入门方式是运行项目提供的GUI。

```bash
python gui.py
```

在打开的网页浏览器窗口中，您可以选择不同的场景、配置参数并运行仿真，结果将以图表形式实时展示。

### 4.2 使用脚本

以下是一个基本的脚本示例，演示如何生成一个 `UMa` 场景的信道。

```python
from channel_model_38901.simulator import ChannelSimulator

# 1. 创建一个UMa场景的仿真器实例
simulator = ChannelSimulator(scenario_name='UMa', center_frequency_hz=3.5e9)

# 2. 运行仿真
results = simulator.run()

# 3. 打印结果
print(f"场景: {results['lsps'].scenario}")
print(f"信道条件: {'LOS' if results['is_los'] else 'NLOS'}")
print(f"路径损耗: {results['pathloss_db']:.2f} dB")
print(f"信道矩阵 H (形状: {results['channel_matrix'].shape}):\n{results['channel_matrix']}")
```

## 5. 用户手册

### 5.1 使用 `ChannelSimulator`

`ChannelSimulator` 是与库交互的主要接口。在创建实例时，您需要指定最核心的参数。

```python
# 构造函数
simulator = ChannelSimulator(
    scenario_name: str,              # 大尺度环境, e.g., 'UMa', 'RMa'
    center_frequency_hz: float,      # 中心频率 (Hz)
    cluster_model_name: str = None,  # 小尺度模型, e.g., 'CDL-A'
    use_median_lsps: bool = False    # 是否使用LSP中位值
)
```

### 5.2 混合信道模型

混合模型是本仿真器的核心特色。它允许您将一个大尺度环境的路径损耗和阴影衰落，应用到一个确定性的簇模型上。

例如，要生成一个受 `UMa` 环境影响的 `CDL-A` 信道：

```python
simulator = ChannelSimulator(
    scenario_name='UMa',           # 大尺度环境
    cluster_model_name='CDL-A',    # 小尺度簇模型
    center_frequency_hz=3.5e9
)
results = simulator.run()
```

此时，仿真器会：
1.  计算 `UMa` 场景的路径损耗。
2.  生成 `UMa` 场景的随机大尺度参数（阴影衰落、角度扩展等）。
3.  加载 `CDL-A` 预定义的、固定的簇参数。
4.  将三者结合，生成最终的信道矩阵。

### 5.3 使用确定性LSP

为了进行可复现的测试，您可以设置 `use_median_lsps=True`。这将使得所有大尺度参数（如时延扩展、角度扩展、阴影衰落）都取其统计分布的中位值，从而消除这部分的随机性。

```python
simulator = ChannelSimulator(
    scenario_name='RMa',
    center_frequency_hz=0.7e9,
    use_median_lsps=True  # 启用确定性LSP
)
results = simulator.run()
```

### 5.4 配置仿真参数

您可以在创建 `ChannelSimulator` 实例后，通过其提供的方法修改仿真参数。

```python
# 创建实例
simulator = ChannelSimulator(scenario_name='UMa', center_frequency_hz=3.5e9)

# 修改基站和UE的位置
bs_pos = np.array([[0, 0, 30]])
ue_pos = np.array([[150, 50, 1.5]])
simulator.set_bs_ue_positions(bs_pos, ue_pos)

# 修改UE的速度 (30 km/h, 沿Y轴正方向)
ue_vel = np.array([[0, 30 * 1000 / 3600, 0]])
simulator.set_velocities(ue_vel)

# 修改天线配置 (4x2 MIMO)
simulator.set_antenna_config(num_tx_rows=4, num_tx_cols=1, num_rx_rows=2, num_rx_cols=1)

# 重新运行仿真
results = simulator.run()
```

## 6. API 参考

### `ChannelSimulator`

#### `__init__(self, scenario_name, center_frequency_hz, cluster_model_name=None, use_median_lsps=False)`
初始化仿真器。

#### `run(self, force_condition=None, time_s=0.0, logger=print, use_median_lsps=None) -> Dict`
运行一次信道实现。
- **force_condition**: `'LOS'` 或 `'NLOS'`，强制指定信道条件。
- **time_s**: 时间点（秒），用于计算多普勒效应。
- **logger**: 日志记录函数。
- **use_median_lsps**: 单次运行覆盖LSP中位值设置。
- **返回**: 包含结果的字典。

#### `set_bs_ue_positions(self, bs_positions, ue_positions)`
设置基站和UE位置。

#### `set_velocities(self, ue_velocities)`
设置UE速度。

#### `set_antenna_config(self, num_tx_rows, num_tx_cols, num_rx_rows, num_rx_cols)`
设置天线阵列配置。

#### `set_scenario(self, scenario_name, cluster_model_name=None)`
切换仿真场景。

#### `set_median_lsp_usage(self, use_median_lsps)`
设置是否使用LSP中位值。

#### `get_config(self) -> ScenarioConfig`
返回当前仿真配置。

## 7. 输出数据结构

`run()` 方法返回一个字典，包含以下键：

- `channel_matrix` (`np.ndarray`): 最终的复数信道矩阵 `H`，已包含路径损耗和阴影衰落。
- `channel_matrix_normalized` (`np.ndarray`): 归一化后的信道矩阵，其功率总和为1。
- `clusters` (`List[EngineCluster]`): 描述信道多径簇的列表。每个 `EngineCluster` 对象包含时延、功率、角度等详细信息。
- `lsps` (`LargeScaleParameters`): 本次仿真生成的大尺度参数对象。
- `pathloss_db` (`float`): 计算出的路径损耗（dB）。
- `is_los` (`bool`): `True` 表示为LOS信道，`False` 表示为NLOS信道。

## 8. MIMO OTA 仿真器 (`mimo_ota_simulator`)

`mimo_ota_simulator` 是一个利用 `channel_model_38901` 核心库构建的上层应用，专门用于MIMO OTA（Over-the-Air）测试场景的仿真。

### 8.1 核心功能

它的核心任务是解决一个工程问题：如何在一个物理的、由有限探头天线组成的暗室中，精确地复现一个复杂的目标信道（如CDL-A模型）。

它通过“空间场合成法”实现这一目标，即将目标信道的每个多径簇，通过其周围的多个物理探头进行加权合成，从而在被测设备（DUT）所在的位置精确地复现该簇的来波方向和功率。

### 8.2 如何使用

该应用提供了一个独立的GUI，是进行MIMO OTA仿真的主要入口。

```bash
python mimo_ota_simulator/app.py
```

在打开的GUI中，您可以配置：

- **暗室参数**: 如探头数量、环形半径、是否双极化等。
- **目标信道**: 使用与核心库一致的参数，定义您想要复现的信道，例如 `UMa` + `CDL-C`。
- **天线配置**: 配置发射端（虚拟）和接收端（DUT）的天线阵列。

运行后，仿真器会计算出需要馈入到每个物理探头的激励信号（A矩阵），并提供丰富的可视化图表来分析合成效果，例如：

- **簇-探头功率贡献矩阵**: 清晰地展示每个目标簇的功率是如何分配到各个物理探头上的。
- **各探头相对功率分布**: 显示所有探头的最终功率分布。

## 9. 图形用户界面 (GUI)

`gui.py` 提供了一个方便的图形界面，用于交互式地运行仿真。

- **左侧边栏**: 包含所有可配置的参数，如场景、频率、天线配置、速度等。
- **主窗口**: 显示仿真结果，包括信道矩阵、路径损耗、簇信息以及各种可视化图表。

## 10. 致谢

本项目的开发得到了 [Google Gemini](https://gemini.google.com) 的大力支持。
