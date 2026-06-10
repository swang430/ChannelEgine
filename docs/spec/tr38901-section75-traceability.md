# TR 38.901 §7.5 实现可追溯表

> Audit batch 2 / PR16 (2026-06-11)。约定:实现状态有变化时**必须**同步本表;
> "简化"必须在此显式声明,不允许只活在代码注释里(历史教训:CLAUDE.md 曾声称
> K 因子进簇功率,与代码不符达数月)。
>
> 基准:TR 38.901 V19.0.0。参数文件 `parameters1013_tr38901_v19.0.0.json`
> (唯一权威,PR #30 起备用文件已删除;加载时有 key 消费校验)。

| Step | 内容 | 状态 | 说明 / 遗留 |
|---|---|---|---|
| 1 | 环境设定(网络布局、天线参数) | 🟡 部分 | 单 BS / 单 UE;ULA/URA;LCS↔GCS 朝向(bearing/downtilt/slant)✅;坐标系约定 issue #13(channel model ULA 沿 +y vs OTA 沿 x)未决 |
| 2 | LOS/NLOS 概率 | ✅ | per-scenario 概率公式硬编码于 `*_models.py`(authoritative;JSON 的 los_probability 块因不可 safe_eval 已删,PR #30) |
| 3 | Pathloss | ✅ | JSON formula 经 safe_eval;SF 从 `sigma_log10_SF` 采样 |
| 4 | LSP 生成 | ✅ | 互相关 Cholesky(双向 key,PR #16);**遗留**:ZSD 在 6 维相关之外独立采样(无 ZSD 互相关);无 §7.6.3 空间一致性 |
| 5 | 簇延迟 | ✅ | 指数采样 + 排序;LOS 时 ÷D(K)(PR #26,系数待 spec 核验);**遗留**:r_τ LOS/NLOS 分列机制已支持、数值待 V19.2.0 参数升级 |
| 6 | 簇功率 | ✅ | ζ per-scenario(`per_cluster_shadowing_std_db`,PR #26;SMa/InF/InH LOS 值经 Codex 对表修正);-25 dB 弱簇删除 ✅(判据为扩散功率;镜面分量由引擎独立合成,见 step 11);K 调整功率仅用于 step 7 角度归一化 |
| 7 | 簇角度 | ✅ | C_φ/C_θ 表 + LOS K 修正 + 几何 LOS 中心化 + LOS 首簇对齐 + 室内 NLOS ZOA 90° 中心(PR #25);**遗留**:NLOS μ_offset,ZOD 参数文件缺字段取 0(随 V19.2.0 补) |
| 8 | Ray 随机配对 | 🟡 部分 | SoS/FPGA 导出 ✅(PR #27:AOD/ZOA/ZOD 独立置换);**主合成路径无 ray 级**(见 step 11) |
| 9 | XPR | 🟡 簇级 | per-cluster 标量(spec 为 per-ray κ_n,m);dual-pol 消费在 mimo_ota_simulator(Phase 6,§7.3.2 2×2 极化矩阵) |
| 10 | 初相 | 🟡 簇级+ray | per-cluster 4 相位(Θθθ..Θφφ);SoS 导出 per-ray = base(Θθθ)+uniform(PR #27);spec 为 per-ray 4 相位 |
| 11 | 信道系数 | ⚠️ 簇级窄带近似 | **最大遗留项(audit F10)**:stochastic 用 Kronecker 相关 Rayleigh(R_rx^(1/2)·H_iid·R_tx^(H/2)),CDL 用 rank-1 平面波,均非 eq 7.5-22 的 ray-sum;H 为 flat-fading 单矩阵,delay 不进 H(宽带 tap 结构在 OTA 的 channel_impulse_response/F64 导出);Cholesky 失败静默退化 IID(待加 warning);LOS 经全局 Rician 混合 √(K/(K+1))·H_LOS + √(1/(K+1))·H_NLOS |

## 附加条款

| 条款 | 状态 | 说明 |
|---|---|---|
| §7.7.1 子簇分裂 | ✅ opt-in | `split_subclusters` + `c_ds_ns` |
| §7.7.1 Table 7.7.1-1..5 ray spread | ✅ | CDL `ray_spread_deg`(PR #18;数值来自模型记忆,经 Codex review 未被质疑,仍建议人工对表) |
| §7.7.3/§7.7.5.1 CDL 缩放 | ✅ opt-in | `scale_to_scenario`,含簇内 spread 同比例缩放(PR #18) |
| §7.3.2 极化 / Table 7.3-1 方向图 | ✅ | dual-pol synthesizer(Phase 6);DUT/探头方向图(PR #15/#19) |
| §7.6.x(O2I、blockage、spatial consistency、大带宽等) | ❌ 未实现 | 超出当前范围;O2I 缺失意味着室内判定只按场景名(InH/InF) |

## 防回归机制索引

- 参数敏感性守门:`tests/test_param_sensitivity_guard.py`(50 用例 + 登记完整性强制)
- 配置消费校验:`ParameterLoader._warn_unconsumed_keys`(未知 JSON key 告警)
- spec 数值待核验清单:Table 7.7.1 c_*(PR #18)、eq 7.5-10/15 K 多项式(PR #25)、step 5 D(K) 多项式(PR #26)、ζ 数值(PR #26,部分已经 Codex 对表修正)
