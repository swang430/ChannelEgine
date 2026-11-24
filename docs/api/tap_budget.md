# P2-109 计算接入契约

实现位于 `/private/tmp/p2109-ce/mimo_ota_simulator/tap_budget.py`，新增文件可直接与 P2-108 的 `channel_comparison.py` 一起接入。未操作 git。已核对 `/private/tmp/p2109-vrt/docs/roadmap-first-call.md` 第 19 行。

## 调用

```python
from mimo_ota_simulator.tap_budget import compare_tap_budgets
result = compare_tap_budgets(
    paths, offsets_hz, port_pairs,
    budgets=(24, 26, 31), amplitude_floor=1e-8,
    time_basis=source_time_note, delay_zero_basis=source_delay_note,
)
```

`paths` 为单帧八路有序列表，每路 `(complex_coefficients, delays_seconds, membership)`，三个同长一维数组。成员为非负整数；必须沿原始 MPDB 非零路径顺序与 members.npz 一一对应。零复系数射线应由调用方同步过滤系数、时延和成员；本接口拒绝零系数、无效成员、NaN、负时延、空端口、预算不足。端口可缺少某个共同簇或某个半簇。**不重新聚簇，不填充零抽头来凑预算。**

`offsets_hz` 默认建议 `np.arange(-20,21)*2.5e6`；`port_pairs` 为唯一整数列表 `(a,b),a<b`，默认八路取同 TX 或同 RX 的 16 对。幅度下限必须为正数，实样用 1e-8。

## 算法及预算含义

- 固定成员，每帧按簇原始射线与未拆单抽头响应的八路带内复频响误差平方和降序，稳定排序。两种时延策略分别排序；这是研究基准的一次静态排序，并非每拆一次重排残差。
- 每簇用八路合并路径功率，在有效时延间隙上最大化早晚加权重心间的分离量，选一个八路共用的早晚分界。早组 `delay<=cut`，晚组 `delay>cut`；同延迟射线不能被分到两边。
- `independent`：每端口、每半簇各自按路径功率加权时延。`shared`：对应半簇用八路全部路径功率共同加权时延。系数两策略都保留原端口复系数相干和。
- B 预算最多选前 `B-共同簇数` 个可拆簇；每口实际数量按非空簇及两半是否均非空计算，保证 `used_taps<=B`。因此 26、31 不保证恰好用满；不可为凑数额外拆低排名簇。24 对应本批 24 个共同成员簇基线。
- 保留输入秒制时延零点，频响只乘相对频偏，不再次乘载频，不逐口减最早时延。

## 返回

顶层 `schema=tap-budget-comparison/1, reference`（NumPy complex[port,F]）, `offsets_hz,port_pairs,base_cluster_count,strategies,time_basis,delay_zero_basis,alignment_status='unverified',formal_verdict=None`。

`strategies` 两键 `independent/shared`，每个包含：

- `ranking`：有序记录 `cluster_id,score,last_early_delay_s,extra_taps_per_port`。不可拆簇 cut 为 null，选择时跳过。
- `budgets`：键为字符串 `'24'/'26'/'31'`。每个记录 `requested_taps,used_taps[port],split_clusters,response`（NumPy complex[port,F]）。NumPy 复响应仅供计算，**不要直接 JSON 序列化整个返回**。
- 每预算 `metrics` 可直接 JSON：复用 P2-108 `compare_responses`，`methods.candidate` 和 `methods.baseline` 含 `matrix_nmse[F]`、`phase_error_deg[pair,F]`。baseline 始终为本策略原始未拆簇方案。`phase_valid_mask` 在本策略所有请求预算和基线间共用；零参考能量 NMSE 为 null，低幅度相位为 null。
- 每预算 `delta.matrix_nmse/phase_error_deg` 含同形 `values`（候选减 24 基线；改善为负，变差为正，无效为 null）、`valid`、`improved/worse/unchanged` 计数。容差为 1e-7，NMSE 为线性值、相位为度。可由索引与调用方 frame_id、offsets_hz、port_pairs 一一导出。两个策略的覆盖各自独立；若跨策略比较分位数，调用方须取共同掩码。

## 来源和边界由接入方核验

计算函数没有 MPDB/NPZ 文件 IO。接入方负责解压安全、源哈希、成员哈希与源行顺序、同频同极化、4×2 唯一端口、帧映射及缺证提示。本次实样来源 SHA256 为 `217585159349afe88ca9864be4a98158625996fefc2639af4403e9727682a2dd`，members.npz 为 `c3b9da980667ef38cf44f7275a3a12105dcc1b4e95eacbffdf23e0f3eb1bcd1a`。

分析用 MPDB 源帧 **0–37 共38帧**；第37帧可以计算射线与固定成员预算，但没有共同 ASC 参考，不能冒充 P2-108 对齐的第38个 ASC 帧。已有 ASC 共同参考仍仅0–36。输入 time/delay 说明只记录、不升级对齐状态。无 HyperRT，无设备，不写回工程，不生成正式判定，不自动选用方案。

## 验证和实样

先红：8项测试均因计算模块不存在而断言失败；实现后绿。补充共同分界、缺半簇、弱幅度无效点测试后，本片10项通过。相关回归73项通过（9.68秒）。7/7内存变异被检出：预算强制用满、相干和改幅度和、空半簇虚增数、丢失共用分界、正式判定污染、排序方向反转、共享时延误用独立时延。

实样测试逐项核对研究 `two-delay-comparison/points.npz`：38帧×两策略×三预算×41矩阵频点=9,348个矩阵点，及149,568个相位点，外加每路实际数量；全部一致（NMSE rtol=1e-9/atol=1e-12，相位 atol=1e-7度）。合成测试不混入实样汇总。

导出文件位于 `/private/tmp/p2109-ce/docs/evidence/p2109/`：`summary.json`、`phase-points.csv`、`matrix-points.csv`、`ranking.csv`、`used-taps.csv`。相位表逐行保留 frame/strategy/budget/port_a/port_b/offset/error/baseline/delta/change/valid；矩阵表按帧/频点保留整体矩阵值，不能把矩阵误差称为单端口误差。全部改善/变差比较均相对同策略24基线。38帧本批两策略三预算共同相位有效点均为24928/24928，矩阵1558/1558。

| 策略 | 预算 | 实际抽头范围 | NMSE P95 | 相位P95(度) | 相位改善/变差 |
|---|---:|---:|---:|---:|---:|
| independent |24|24|1.9959755325|65.0963724378|0/0|
| independent |26|26|0.5858199776|55.1031792650|15771/8549|
| independent |31|28–31|0.5668307564|52.8963215066|16623/7697|
| shared |24|24|2.0095579391|64.2387095397|0/0|
| shared |26|24–26|0.5739980466|54.5566444504|15097/9223|
| shared |31|28–31|0.5638625034|52.1389703398|15679/8641|

分位数改善不代表逐点保证。shared/31仍有310个矩阵频点、8641个相位点较本策略24基线变差；independent/31分别315及7697。全部保留，禁止只展示收益。

复跑：
```sh
cd /private/tmp/p2109-ce
python3 -m pytest tests/test_tap_budget.py tests/test_channel_comparison.py tests/test_channel_comparison_sample.py tests/test_joint_cluster_response.py tests/test_imported_cluster_trajectory.py tests/test_port_evidence.py -q
python3 scripts/check_tap_budget_mutations.py
python3 scripts/export_tap_budget_sample.py
```
实样回归依赖本机所指定外部实样、参考解析器和研究NPZ；缺少时失败而非伪称实样通过。它们只服务本机实样验收，不成为计算模块运行依赖。
