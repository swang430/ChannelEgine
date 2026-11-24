# P2-108 ChannelEgine 调用约定（2026-11-24）

工作树 `/private/tmp/p2108-ce`，新增模块 `mimo_ota_simulator.channel_comparison`。仅计算辅助量，所有返回 `formal_verdict=None`，`alignment_status=unverified`，不调用 HyperRT、不连接设备、不写回工程。协调者独占 git 发布。

## 数组接口（已实现，63 项针对性回归通过）

- `frequency_response(coefficients, delays_s, offsets_hz) -> complex[F]`：同长一维路径系数和秒制时延。H(Δf)=Σh exp(-j2πΔfτ)。系数已经包含载频相位，禁止再次乘载频，禁止逐端口减最早时延。
- `member_taps(coefficients, delays_s, membership, *, tap_budget=24)`：成员为非负整数，同长数组；各簇复系数相干和、每端口按路径功率加权时延。返回 `cluster_ids, coefficients, delays_s` NumPy 数组及 `requested_taps,used_taps`。超预算/零功率簇/未分组拒绝，不截断。
- `compare_responses(reference, methods, offsets_hz, port_pairs, *, amplitude_floor, time_basis=None, delay_zero_basis=None, simulated=False)`：一个源帧，各响应形状 `[port,frequency]`，methods 是名称到复数组映射。频点、极化、帧与来源由调用者预先核对。`port_pairs` 必须为唯一 Python 整数 `(a,b)` 且 `a<b`。
  - JSON 安全字典：`schema=channel-comparison/1, offsets_hz, port_pairs, amplitude_floor, methods, phase_valid_mask, matrix_valid_mask, coverage,time_basis,delay_zero_basis,alignment_status,simulated,formal_verdict`。
  - `methods[name].matrix_nmse[F] = Σport|Hmodel-Hraw|² / Σport|Hraw|²`；零参考能量返回 null。
  - `methods[name].phase_error_deg[pair][frequency]` 为绝对包裹角差，0..180°。**所有模型与原始参考**的该对两个端口都严格大于同一幅度下限，才进入共同相位掩码；否则 null，不能记成 0°。
  - `coverage`：`phase_valid,phase_total,phase_fraction,matrix_valid,matrix_total,matrix_fraction`。分母保留全部要求点。
  - time/delay basis 只记录说明，不能凭字符串自动升级对齐判定。模拟始终无正式结论。
- `power_summary(coefficients, *, input_power_w=None)`：返回 `carrier_coherent_gain=|Σh|²,tap_power_sum=Σ|h|²,input_power_w,absolute_received_power_w`。无输入参考则最后一项 null；有输入参考按载频相干增益乘输入功率（W）。系数不是 dBm，抽头功率和不是载频相干功率增益。
- `recover_asc_frames(data, *, rate_hz, source_frame_count, threshold_ns=1e-7)`：只用于本批参考的 `[sample,port,tap,3]` 数组，末维 delay_ns/re/im。按照历史脚本时延二阶差分配对拐点恢复 N-1 个数值参考，左右幅度和最短弧相位外推分别保留；无法唯一恢复时拒绝猜测。
  - 返回 NumPy `delays_s,coefficients,right_delays_s,right_coefficients` 形状 `[N-1,port,tap]`，以及 `sample_positions,times_s,unmatched_source_frames=[N-1],time_status=unverified,time_basis`。
  - 真实 38 源帧必须得到前 37 帧；末帧 37 单列，不以 ASC 末采样点冒充。
  - `amplitude_roundoff` 保留零附近负幅度修正计数、最大绝对修正及相对容差。只容许局部外推幅度尺度 1e-10 内的负数修正为零；实质负幅度拒绝。此次左右外推共 24 个修正，最大 3.671027558510337e-21。

## 接入职责

Meta-3D 负责安全解析 MPDB/ASC、来源摘要、成员哈希及源行顺序验证；验证 4×2 唯一端口、同载频同极化、24 抽头及共同源帧，再调用以上接口。建议默认频偏 `np.arange(-20,21)*2.5e6`（±50 MHz，41 点）、幅度下限 `1e-8`、端口对为同 TX 或同 RX 的 16 对。可提供逐点 CSV：frame/method/offset_hz/port_a/port_b/matrix_nmse/phase_error_deg/phase_valid；另列左右外推差异、carrier 相对误差和未配对源帧。

## 最终验证与实样结果

- 首次红测试：12 项新增测试均因缺少计算模块而失败。实现后通过；实样回归另捕获外推零幅度附近舍入负值，定位后作有界修正并输出修正量。
- `python3 -m pytest tests/test_channel_comparison.py tests/test_channel_comparison_sample.py tests/test_joint_cluster_response.py tests/test_imported_cluster_trajectory.py tests/test_port_evidence.py -q`：**63 passed in 2.45s**。
- `python3 scripts/check_channel_comparison_mutations.py`：**11/11 检出**（时延单位、复系数相干和、预算、矩阵分母、共同相位掩码、严格幅度边界、无效相位不得置零、相干功率、缺输入参考的绝对功率、正式结论、末帧缺口）。仅内存变异，生产文件未改。
- 实样测试 `tests/test_channel_comparison_sample.py` 只读加载指定八路 ASC、MPDB 和 members.npz，与指定历史 frames.csv、points.csv 独立结果对照。逐项核对 **37 帧 × 2 方法 × 16 端口对 × 41 频点 = 48,544 条**，全部一致（NMSE 相对容差 1e-9/绝对 1e-12，相位绝对容差 1e-7°）。无样本或参考 CSV 时明确 skip，不能替代本次实样通过记录。
- 前 37 帧（0..36）各八路成员方案均用满 24 抽头。ASC 拐点恢复时间与 0..36 秒最大数值残差 **7.105427357601002e-15 s**。这只是 ASC 数值时间，不是 MPDB 物理源时间证据。
- ASC 左右外推频响差相对原始最大幅度的最大比值 **9.97605328246381e-13**；载频系数相对原始的最大误差 **2.8148947576478816e-12**。这些支持本批数值参考，**时延零点的物理含义仍未核验**，沿用输入秒制时延且不作逐口零点平移。
- 源末帧 **37** 无共同 ASC 帧参考，单列为未核验；不能把 36.141625497604686 秒的 ASC 末采样当作源帧 37。
- 在 ±50 MHz / 41 点 / 幅度下限 1e-8 下，共同相位有效覆盖 **24,272/24,272=100%**（每方法），矩阵点 **1,517/1,517=100%**。该覆盖只适用于本批数据及此下限，不代表其他帧/其他极化/更低幅度/现场覆盖。

| 方法 | 矩阵 NMSE P50 | 矩阵 NMSE P95 | 相位绝对误差 P50 | 相位绝对误差 P95 |
|---|---:|---:|---:|---:|
| 已有 ASC（历史 CSV 标签 Lauraycs） | 0.08417480096926037 | 2.7832462622408487 | 5.507847605820343° | 75.54268898009221° |
| 同源成员相干和（历史标签 Meta-3D） | 0.06741682200220137 | 2.0224255891125456 | 4.938104736098367° | 66.08165099809521° |

两方法都与原始射线共同参考比较，指标为线性 NMSE 而非 dB。P50/P95 不是逐点保证，也不是方案自动采纳条件。无输入功率参考，绝对接收功率保持未知。测试中的合成数据仅验证数学与失败分支，未进入真实样本统计或任何正式判定。全程未调用 HyperRT、未连接设备、未写回输入工程、未执行 git 提交/发布。
