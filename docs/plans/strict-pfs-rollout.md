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
| D3 | 现有 `cluster` method | ✅ **Deprecate** | Phase 3 调研：无文件大小优势（17.5 vs 17.6 KB）、无生产 hardcoding、无 MIMO-First 依赖 |
| D4 | 现有 `pinv` method | ✅ **Deprecate（更强）** | 同 D3 + bonus：F64 export 已 broken（no wideband taps），实际不可用 |
| D5 | `probe_phase_jitter` flag | ✅ 保留，仅 `ray` 下可用 | `strict_pfs` 下 UI grey-out + tooltip 说明 |
| D6 | ASC F64 输出格式 | ✅ **不动** | strict_pfs 必须能生成符合 F64 spec 的 .asc，否则方案失败 |
| D7 | Cluster mode rank-1 难题 | ✅ **绕过** | strict_pfs 直接走 ray-level 表示，不试图改 cluster |
| D8 | Phase cal cert 兼容 | ✅ strict_pfs + cal **应叠加正确** | 区别于 jitter+cal 互斥；Phase 2 验证 |
| D9 | `strict_pfs` 默认性 | ✅ **保守 opt-in** | 不替代当前默认 `ray`；等 1-2 release 验证后再考虑 |
| D10 | GCM Studio cross-validation | ✅ **完全 out-of-scope** | MIMO-First 的内容，本 repo 不做 |
| D11 | MIMO-First adapter API（`run_with_external_clusters`）是否在本 repo 实现 | ✅ **不实现** | 本 repo 是 library 不是 service；适配工作由 MIMO-First 那边做，本 repo 只承诺 `run()` 签名稳定 + 文档化 |

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

### Phase 1 —— Strict PFS 核心实现 ✅ **DONE 2026-05-18**

**Goal**: `synthesis_method='strict_pfs'` 可用，per-(probe, cluster) 独立 fading

- [x] **T1.6** F64 兼容性 POC GO/NO-GO gate (commit `f71c0bb`)
- [x] **T1.1** synthesis_method='strict_pfs' 路由 slot + Literal type (commit `e5c0989`)
- [x] **T1.2** phi_init 数据契约 lock + 4 unit tests (commit `558751a`)
- [x] **T1.3** strict_pfs 完整合成路径（merged with ray branch, 3-line delta）(commit `54017cf`)
- [x] **T1.4** Tx steering / w_map reuse 验证（实现自证, 同 commit `54017cf`）
- [x] **T1.5** UL/DL Option A reciprocity + axis-aware exporter broadcast (commit `92cdf8c`)
- [x] **T1.7** UI 加 strict_pfs 选项 app.py + gui.py (commit `efe19fc`)
- [x] **T1.8** UI strict_pfs ⇄ jitter mutex (Streamlit disabled=True + simulator runtime warn) (commit `efe19fc`)
- [x] **T1.9** docs/api/mimo_ota_simulator.md public API contract (commit `929ed30`)

**Acceptance result**:
- ✅ `synthesis_method='strict_pfs'` 跑通；F64 .asc 8 文件 82637 bytes (与 ray 结构一致)
- ✅ UI 4 个 method 可切换；strict_pfs 选中时 jitter 自动 disabled
- ✅ 单 realization A 无 NaN/Inf；strict vs ray mean abs diff 0.35 (RNG state divergent，对)
- ✅ UL/DL: DL (8,2) ↔ UL (2,8)，phi_ul == -phi_dl，UL export 165047 bytes
- ✅ Warning matrix 4/4 case 正确（strict+jitter / strict alone / ray+jitter / ray alone）
- ✅ 测试: 15/15 pass (4 phi_init contract + 4 strict_pfs smoke + 7 regression)

---

### Phase 2 —— Validation & 数学正确性证明 ✅ **DONE 2026-05-18**

**Goal**: 数学层面证明 strict PFS contracts + 与 phase cal 叠加正确性

- [x] **T2.4** Regression 加固 (commit `7c4ce9c`): 修复 pre-existing channel_builders antenna config bug + 新 ray golden + skip stale pinv mock
- [x] **T2.3** strict_pfs + cal 叠加 3 sub-tests (commit `ed2b0a0`)
- [x] **T2.1** E[A_i·A_j*]→0 + ray contrast (commit `f8c23d8`)
- [x] **T2.2** Cross-batch variance comparison — **L·M× speedup claim 实测不成立**; relax 为 regression guard (ratio > 0.5)

**Acceptance result**:
- ✅ T2.1-T2.4 全 pass (73 + 1 skip)
- ⚠️ T2.2 design contract revised: strict_pfs 跟 ray+jitter 2nd-moment cross-batch variance 几乎相等。strict_pfs 真正优势是 per-realization 严格 0 cross-corr + cal 叠加 + deterministic seed
- 📌 Bonus: channel_builders antenna config bug 修复 (cluster mode 输出之前是 rank-1 退化的)

---

### Phase 3 —— D3/D4 调研 & 旧 method 处置 ✅ **DONE 2026-05-18**

**Goal**: 决定 `cluster` / `pinv` 是 keep / deprecate

- [x] **T3.1** Grep 全 repo + MIMO-First repo 找 `cluster` / `pinv` 的活跃 caller —— Explore agent + 跨 repo 验证
- [x] **T3.2** 对比 `.asc` 文件大小：cluster=17.5KB, ray=17.6KB（0.6% 差异可忽略）；**pinv F64 export 已 broken**（impulse_response 为空 → IndexError）
- [x] **T3.3** Decision: 两个都 deprecate（见 §1 D3/D4），原因写入 commit message
- [x] **T3.4** 加 `DeprecationWarning` 到 `simulator.py` L375-394（pinv + cluster 两个分支顶部）

**Key finding（影响 Phase 1 设计）**：
MIMO-First 当前**有**两条想调用本 repo OTA simulator 的代码路径，但**接口三层全断**：
- 类名错：`OTASimulator` vs `MIMO_OTA_Simulator` → ImportError
- 构造签名错：MIMO-First 传 ctor 参数，本 repo 都在 `run()` 时传 → TypeError
- 方法名错：`run_with_external_clusters` 不存在 → AttributeError

实际行为：`hardware_pipeline.py` fallback 到 mock（用户看到的 .asc 是假的）；
`ota.py / channel_engine.py` 绕过 OTA simulator，自己重写 weight 计算。

**结论**：deprecate cluster/pinv **不影响** MIMO-First（反正都到不了）。
Phase 1 落地 strict_pfs 时**顺便文档化 `run()` 签名作为 stable public API**（见 D11 + Phase 1 T1.9）。

**Acceptance**: ✅ D3/D4 locked；DeprecationWarning 在测试中正确 fire；7/7 regression 仍 pass

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
- 2026-05-18: Phase 0 完成（commit `9eb1a06`）。
- 2026-05-18: Phase 3 完成；D3/D4 lock 为 deprecate；新增 D11（不为 MIMO-First 实现 adapter）+ T1.9（API 文档化）。MIMO-First 三层 API gap 发现记入 §Phase 3。
- 2026-05-18: Phase 1 完成 (feature/strict-pfs branch, commits f71c0bb..929ed30, 6 commits)。剩 Phase 2 (statistical validation tests) + Phase 4 (cross-repo sync)。
- 2026-05-18: Phase 2 完成 (feature/strict-pfs-validation branch)。L·M× speedup claim revised。Bonus 修复 channel_builders antenna config bug。剩 Phase 4 (cross-repo sync)。
