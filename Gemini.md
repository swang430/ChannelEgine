# Gemini.md: 3GPP TR 38.901 信道模型开发指令

## 1. 项目概述与核心原则

**项目目标**: 创建一个名为** **`ChannelModel38901` 的高性能Python库。该库将精确实现3GPP技术报告TR 38.901中定义的5G无线信道模型，用于学术研究和系统级仿真** **^1^。

**核心设计原则**:

1. **标准遵从性**: 所有实现必须严格遵循3GPP TR 38.901技术报告。
2. **为标准演进而设计 (未来适应性)**: 将算法逻辑与标准参数彻底分离。
3. **模块化与关注点分离**: 系统将被分解为逻辑独立的组件。
4. **性能**: 优先采用基于NumPy的向量化操作。
5. **易用性**: 通过**外观（Facade）设计模式**提供一个简洁的顶层API (`ChannelSimulator`)。
6. **健壮性与容错性**: 优雅地处理异常数据。

## 2. 高层系统架构

系统由以下主要组件构成：

* `ChannelSimulator`: 面向用户的顶层API（外观）。
* `ParameterLoader` & `ParameterTables`: 负责加载标准参数。
* `ScenarioConfig`: 持有用户定义的仿真配置。
* `ABC_*Model` (抽象基类): 定义核心算法的统一接口（策略）。
* `{Scenario}_*Model` (具体子类): 提供具体的场景算法实现。
* `ChannelGenerator`: 核心仿真引擎，执行12步信道生成流程。

## 3. 详细组件规格

### 3.1 数据模型与参数化

#### 3.1.1 参数JSON文件 (`parameters_tr38901_v19.0.0.json`)
* **目的**: 存储所有源自3GPP标准的参数，实现代码与数据的分离。

#### 3.1.2 数据类 (使用 `pydantic`)
* `class ScenarioConfig`: 持有用户仿真配置。
* `class AntennaArray`: 定义天线阵列的配置。
* `class LargeScaleParameters`: 封装一组大尺度参数。

### 3.2 核心模型接口与实现

#### 3.2.1 路径损耗模型
* `abstract class ABC_PathlossModel(ABC)`
* `class UMa_Pathloss(ABC_PathlossModel)` 等具体实现。

#### 3.2.2 大尺度衰落模型
* `abstract class ABC_FadingModel(ABC)`
* `class UMa_Fading(ABC_FadingModel)` 等具体实现。

#### 3.2.3 空间相关性模型
* `function get_spatial_correlation_matrix(...)`: 计算空间相关性的统一入口。

### 3.3 仿真引擎 (`ChannelGenerator`)
* **职责**: 严格按照TR 38.901第7.5节的12步流程生成信道。

## 4. 分阶段实施计划

* **阶段一至五**: 基础架构、核心模型、API封装与GUI (已完成)

## 4.1 设计演进策略：从传播核心到空间维度

*   **第一阶段（已完成）: 核心传播物理**
*   **第二阶段（已完成）: 完整空间维度**
*   **第三阶段（未来扩展）: 高级天线与仿真功能**
    *   **实现1**: 扩展天线阵列模型以支持URA。 (已完成)
    *   **实现2**: 实现多普勒效应。 (已完成)
    *   **实现3**: 支持多用户/多基站。

## 4.2 API 使用模式与实现说明

`ChannelSimulator` 的正确使用流程为：实例化 -> 手动配置 `simulator.config` -> 手动重新初始化 `simulator.generator` -> 运行。

## 5. 应用场景与使用哲学：随机仿真 vs. 标准化测试

本仿真器的一个核心设计思想是灵活性，使其能同时支持两种重要的应用场景：

1.  **统计性系统级仿真 (Statistical System-Level Simulation)**: 目的是评估通信系统在**整个场景（如UMa）**下的平均性能。

2.  **标准化一致性测试 (Standardized Conformance Testing)**: 目的是在一个**可重复的、具有代表性的**恶劣信道环境下，考核设备（如手机、基站）的物理层性能。

## 6. 测试与验证策略

一个未经严格验证的信道模型是不可信的。我们实施了一个**四层验证策略**来确保模型的正确性、稳定性和准确性。

### 6.1 单元测试 (Unit Testing)
*   **目标**: 从数学上确保最底层的、新增的核心算法单元的精确性。
*   **实现**: `tests/test_synthesis.py` 和 `tests/test_geometry.py`
*   **结论**: **核心算法模块已通过验证。**

### 6.2 健全性检查 (Sanity Checks)
*   **目标**: 确保仿真结果在宏观上符合基本的物理直觉。
*   **实现**: `tests/test_sanity_checks.py`
*   **结论**: **模型的宏观表现符合物理规律。**

### 6.3 统计验证 (Statistical Validation)
*   **目标**: 通过大规模蒙特卡洛仿真，确保模型的输出在统计特性上与3GPP标准一致。这是**最高级别的验证**。
*   **实现**: 为了处理大规模仿真产生的大量数据，验证流程被拆分为“生成-分析”两步：
    *   **数据生成脚本**: `tests/generate_lsp_data.py` 和 `tests/generate_h_matrix_data.py`。这些脚本的唯一职责是，在静默模式下运行上千次仿真，并将原始结果（LSP对象或H矩阵）保存到中间数据文件（`.pkl`）中。
    *   **数据分析脚本**: `tests/analyze_lsp_data.py` 和 `tests/analyze_h_matrix_data.py`。这些脚本负责读取中间数据文件，执行所有统计计算（均值、方差、相关性等），打印最终的对比报告，并生成可视化图表。
*   **结论**: **模型的随机过程和核心物理实现在统计意义上是精确的。**

### 6.4 回归测试 (Regression Testing)
*   **目标**: 确保未来的代码修改不会意外破坏已有功能。
*   **实现**: `tests/create_golden_files.py` 和 `tests/test_regression.py`
*   **结论**: **我们为当前稳定版的引擎建立了一个可靠的“代码安全网”。**

## 7. 编码规范与开发环境

* **语言**: Python 3.10+
* **核心库**: `numpy`, `scipy`, `pydantic`, `pytest`, `tqdm`
* **代码风格**: PEP 8