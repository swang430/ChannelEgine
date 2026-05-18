# Plan: Strict PFS Synthesis + Method Pruning

**Status**: DRAFT
**Owner**: @swang430
**Reviewers**: TBD (至少需要一个 MIMO-First 侧 reviewer 确认 spec 对齐)
**Created**: 2026-05-18
**Last updated**: 2026-05-18

**Sister-repo refs**:
- MIMO-First `docs/features/calibration/pfs-phase-immunity.md` (PR #52 / commit `35bb1b8`)
- 本 repo `CLAUDE.md` §"Cross-project context"

---

## 0. 目标 / Why

### Engineering driver（根本动机）

**Strict PFS 是 must-have，不是 nice-to-have**：在仿真侧从数学上保证 per-(probe, cluster) 独立 fading，**对应实测可省掉输出端相位校准**。

输出端相位校准在工程上需要：
- 专门仪表（vector network analyzer 在 OTA chamber 内布站）
- 现场重做（每次硬件换天线 / 换缆 / 移探头位置都要重做）
- 长期漂移监控（温度 / 接头老化都影响相位）

成本和难度都是要避免的。Strict PFS 把这个责任从硬件标定挪到信号合成，是结构性优化。

任何"我们已经有 jitter 了为什么还要 strict"的质疑，回答是：jitter 不替代输出 cal（依赖大规模平均；单次 realization 仍有 coherence 残留），strict PFS 才替代。

### Spec basis

- 3GPP TR 37.977 V18.0.0/V19.0.0 Annex F.2 + B.1
- CTIA 01.40 V6.0.1 §2.2.2.3 / §4.2.5
- Strict PFS 判据：`E[x_i(t) · x_j*(t)] = 0 for i≠j`（probe 间二阶相关恒零，per-realization 即满足，**不依赖 ensemble 平均**）

### Non-goal

- 不重构 ASC F64 输出格式（硬件强约束，与合成方法无关）
- 不预设其他硬件格式（case-by-case，等需求出现）
- 不修 MIMO-First 那侧的 `run_with_external_clusters` 缺口（scope 在 MIMO-First 仓库）
- 不做 GCM Studio cross-validation（属 MIMO-First，本 repo 不关心）

---

## 1. Decisions（先定再做）

| # | Decision | Locked | Notes |
|---|---|---|---|
| D1 | 新 method 命名 | ✅ `strict_pfs` | 不用 `ray_strict`，命名层面跟 ray 解耦 |
| D2 | 现有 `ray` method | ✅ **保留** | jitter 大平均场景仍有用；UI 标"传统/平均"语义 |
| D3 | 现有 `cluster` method | 🔍 **TBD（Phase 3 调研）** | 无强 use case 默认 deprecate |
| D4 | 现有 `pinv` method | 🔍 **TBD（Phase 3 调研）** | 同 D3 |
| D5 | `probe_phase_jitter` flag | ✅ 保留，仅 `ray` 下可用 | `strict_pfs` 下 UI grey-out + tooltip 说明 |
| D6 | ASC F64 输出格式 | ✅ **不动** | strict_pfs 必须能生成符合 F64 spec 的 .asc，否则方案失败 |
| D7 | Cluster mode rank-1 难题 | ✅ **绕过** | strict_pfs 直接走 ray-level 表示，不试图改 cluster |
| D8 | Phase cal cert 兼容 | ✅ strict_pfs + cal **应叠加正确** | 区别于 jitter+cal 互斥；Phase 2 验证 |
| D9 | `strict_pfs` 默认性 | ✅ **保守 opt-in** | 不替代当前默认 `ray`；等 1-2 release 验证后再考虑 |
| D10 | GCM Studio cross-validation | ✅ **完全 out-of-scope** | MIMO-First 的内容，本 repo 不做 |

---

## 2. 分阶段交付

### Phase 0 —— 扫雷（1 工作日，阻塞 Phase 1） ✅ **DONE 2026-05-18**

**Goal**: 现有用户不再被 UI 误导 / 不再静默毁数据

- [x] **T0.1** 改 `app.py:196` jitter UI 文案为 "±180度（完全随机相位）" + 说明适用场景
- [x] **T0.2** 验证 `gui.py` L185 仅有 label，无 help text 需改
- [x] **T0.3** 在 `simulator.py` synthesize_ota L532 加 runtime warning：jitter + cal 同时启用 → warn
- [x] **T0.4** 在 `app.py` 加 UI explicit hint：cal 上传 + jitter 勾选时显示 st.warning
- [x] **T0.5** 跑 regression (`test_regression`, `test_propsim_export`, `test_comprehensive_system`) + 4-case warning matrix smoke test

**Acceptance result**:
- ✅ UI 文案 ±180° 与代码 `jitter_rad = np.pi` 一致
- ✅ Warning matrix: A(jitter+cal)=fires, B(cal only)=silent, C(jitter only)=silent, D(neither)=silent
- ✅ 7/7 regression tests pass (test_regression, test_propsim_export, test_comprehensive_system)

---

### Phase 1 —— Strict PFS 核心实现（3-4 工作日）

**Goal**: `synthesis_method='strict_pfs'` 可用，per-(probe, cluster) 独立 fading

- [ ] **T1.1** 在 `simulator.py` 加 `synthesis_method` 第 4 个值 `'strict_pfs'`；schema/validation 更新
- [ ] **T1.2** 设计 per-(probe, cluster) 独立 fading 数据结构（参考 ray 现有 `phi_rnd` 标量，扩展为 `(N_probes,)` 向量；命名待定）
- [ ] **T1.3** 实现 strict_pfs 合成路径：每 cluster 每 probe 独立抽 phase；保持现有 angle/AS Gaussian weight 不变
- [ ] **T1.4** 复用现有 Tx steering vector 计算逻辑（不重写）
- [ ] **T1.5** 复用现有 Doppler / `ray_components` 数据结构（向 strict_pfs 兼容方式扩展）
- [ ] **T1.6** ASC F64 export adapter：确认 strict_pfs 输出能喂给现有 F64 writer 不需特殊化（D6 假设验证 —— **这是方案 go/no-go 关键节点**）
- [ ] **T1.7** UI 在 `app.py` / `gui.py` 加 `strict_pfs` 选项；标 "推荐（无需输出相位校准）"
- [ ] **T1.8** `strict_pfs` 模式下 jitter checkbox 自动 disable + tooltip 说明

**Acceptance**:
- `synthesis_method='strict_pfs'` 路径可跑通；输出 `.asc` 文件可被 F64 reader 解析
- UI 四个 method (`ray` / `cluster` / `pinv` / `strict_pfs`) 可切换；不需重启
- 单 realization 跑通无 numerical NaN/Inf

---

### Phase 2 —— Validation & 数学正确性证明（2-3 工作日）

**Goal**: 数学层面证明 strict PFS 收敛性 + 与 phase cal 叠加正确性

- [ ] **T2.1** 写 `tests/test_strict_pfs_correlation.py`：跑 N realizations，计算每对 probe 的 `<x_i · x_j*>`，验证收敛到 0（vs `ray` mode 的 non-zero residual）
- [ ] **T2.2** 写 spatial correlation test：strict_pfs 平均 N 次后 `R̂_{m,n}` vs target `R_{m,n}`，confirm 收敛速度比 jitter mode 快 ~L·M 倍
- [ ] **T2.3** 写 strict_pfs + calibration_vector 叠加测试：cal 相位补偿与 strict fading 不互斥（最终 effective H 包含两者效应）
- [ ] **T2.4** Regression test：`ray` + `cluster` + `pinv` 输出与现有 golden 文件 bit-identical

**Acceptance**:
- T2.1-T2.4 全绿
- T2.1 收敛曲线作为 PR 描述附图

---

### Phase 3 —— D3/D4 调研 & 旧 method 处置（1 工作日，可与 Phase 2 并行）

**Goal**: 决定 `cluster` / `pinv` 是 keep / deprecate

- [ ] **T3.1** Grep 全 repo + MIMO-First repo 找 `cluster` / `pinv` 的活跃 caller
- [ ] **T3.2** 对比 `.asc` 文件大小：三种 method 输出大小是否有显著差异（若 cluster 显著小，是 keep 理由）
- [ ] **T3.3** Decision doc 短结：keep 或 deprecate，理由
- [ ] **T3.4** 如 deprecate：加 `DeprecationWarning`，标记移除时间窗（建议 2 release 后移除）

**Acceptance**: D3/D4 locked，更新本 plan §1 表格

---

### Phase 4 —— 文档 + Cross-repo 同步（0.5 工作日）

- [ ] **T4.1** 更新本 repo `CLAUDE.md` §"Cross-project context"：strict_pfs 已实现，更新事实表
- [ ] **T4.2** 更新 memory `project_mimo_first_pfs_investigation.md`：roadmap P2-6 状态从 planned → done
- [ ] **T4.3** 在 MIMO-First 那边开 issue 通知 strict_pfs 可用，请求 `pfs-phase-immunity.md` 加 reference link
- [ ] **T4.4** README 或 GUI tooltip 加"何时用哪个 method"决策树

---

## 3. Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| ASC F64 格式跟 strict_pfs 输出不兼容（D6 假设错） | Low | **High** —— 阻塞整个方案 | Phase 1 T1.6 单独验证；如失败则 Phase 1 提前 stop，回到 design |
| Per-(probe, cluster) 独立 fading 计算时间 N_probes 倍增 | Med | Low-Med | 先实现后 profile；若 >10× ray 则考虑向量化优化 |
| 数学错误导致 strict_pfs 不真 strict | Med | High | T2.1 是关键 gate；不通过不能 ship |
| 现有 `ray` golden 测试因副作用回归 | Low | Med | T2.4 regression test + 默认 method 保持 `ray` 不变 |

---

## 4. Out of Scope（明确不做）

- MIMO-First `run_with_external_clusters` 缺失（修在 MIMO-First 侧）
- 其他硬件格式（非 F64）的输出 adapter
- Cluster mode rank-1 issue 的"修复"（直接绕过，走 ray-level）
- 时变信道 / 频率选择性扩展（未来工作）
- GCM Studio cross-validation（属 MIMO-First）
- GUI 大改版（只加 method 选项 + 文案修正）

---

## 5. Open Questions

- [ ] Q1: D3/D4 的 keep/deprecate 决定，是否需要等 MIMO-First 那边的产品意见？
- [ ] Q2: Phase 0 改 UI 文案为 "±180度" 是否需要同步翻译 / 跟 MIMO-First 文档对齐术语？

---

## 6. Definition of Done

- [ ] Phase 0-2 全部任务勾选完成
- [ ] CI 全绿（含新加的 T2.1-T2.4）
- [ ] 至少 1 个 reviewer approval
- [ ] CLAUDE.md cross-project section 更新完成
- [ ] Memory 文件更新完成
- [ ] MIMO-First 那边 issue 已开

---

## Changelog

- 2026-05-18: Initial draft (Claude + @swang430). 决策 D1-D10 锁定；GCM Studio 移出 scope。
