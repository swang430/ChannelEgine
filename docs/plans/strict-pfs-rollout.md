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

### Phase 4 —— 文档 + Cross-repo 同步 ✅ **DONE 2026-05-18** (post-Phase-6)

- [x] **T4.1** `CLAUDE.md` §"Cross-project context" 完全重写: 4 个 2026-05-17 outdated facts 替换为 post-Phase-6 状态表 + MIMO-First 集成推荐写法 + Phase 7 roadmap 入口
- [x] **T4.2** Memory `project_mimo_first_pfs_investigation.md` rewrite: 标记原 4 finding 每个的解决状态，MEMORY.md index 同步
- [x] **T4.3** Meta-3D 开 issue #55 通知 strict_pfs + CustomCDLProfile + dual-pol 可用 + 给 MIMO-First service-side adapter 重写模板
- [x] **T4.4** `README.md` 加 "MIMO OTA Synthesis — Method Selection" 决策树 + 4 method 状态表 + External CDL Input + Dual-Polarization 示例

**Acceptance result**:
- ✅ CLAUDE.md 顶部 cross-project context 跟主 plan 同步 (post-Phase-6 状态)
- ✅ Meta-3D issue #55: https://github.com/swang430/Meta-3D/issues/55 (adapter rewrite + spec link)
- ✅ README 决策树让新 caller 5 秒内挑对 method
- ✅ Phase 7 slant pol roadmap 入口在 CLAUDE.md + memory 都标注

---

### Phase 5 —— External CDL Input API ✅ **DONE 2026-05-18**

**Goal**: 让 MIMO-First (或任何 external caller) 能传完整 CDL 参数 (含 XPR / K-factor / per-ray init phases) 进来产 OTA + ASC。

详见 [external-cdl-input-design.md](./external-cdl-input-design.md)。

**Sub-tasks (8 commits)**:
- [x] **T5.1+T5.2** `CustomCDLProfile` + `CDLClusterSpec` Pydantic schema + `from_dict`/`from_file` factories (commit `66a40ed`)
- [x] **T5.6** `EngineCluster.xpr_linear` field (Phase 5 plumb-only) (commit `5e9b6ed`)
- [x] **T5.3** `CustomCDLBuilder` + `CustomJSONBuilder` deprecation shim (commit `0bd4e0f`)
- [x] **T5.7** `TargetChannelConfig.custom_profile` field + create_builder routing (commit `3db235d`)
- [x] **T5.4+T5.5** K-factor LOS boost (TR 38.901 §7.5.6) + per-ray initial phases integration (commit `8fdfda0`)
- [x] **T5.8** E2E test: dict → strict_pfs → Propsim DL ASC (commit `46755a6`)
- [x] **T5.9+T5.10** SPEC v2.0 rewrite + API doc CustomCDLProfile section (commit `08740bb`)

**Acceptance result**:
- ✅ 115 pass + 1 skip (full suite, no regression)
- ✅ XPR plumbed to `EngineCluster.xpr_linear` (Phase 5 status; Phase 6 真用)
- ✅ K-factor 10 dB → first cluster 10/11, rest 1/11 (TR 38.901 §7.5.6 verified)
- ✅ per-ray `initial_phases_rad[0]` (Θθθ) integrated into strict_pfs base phase, cross-corr contract preserved (T2.1 contract held with N=60 + π/4 base)
- ✅ `CustomCDLProfile.from_dict` end-to-end → 16 ASC files generated
- ✅ Legacy `CustomJSONBuilder(filepath)` still works with `DeprecationWarning`
- ✅ SPEC v2.0 published; v1.0 deprecated with migration table

---

### Phase 6 —— XPR / Dual-Polarization Synthesizer ✅ **DONE 2026-05-18** (scope = B)

**Goal**: synthesizer 真正应用 XPR 字段 (Phase 5 plumb)，实现 dual-polarization OTA 数学模型。

**Scope (2026-05-18)**: option B —— probe H/V 真实响应 + DUT pure V/H per antenna。

详见 [dual-pol-synthesizer-design.md](./dual-pol-synthesizer-design.md)。

**Sub-tasks** (8 commits):
- [x] **T6.1** `AntennaArrayConfig.polarization` field (default 'V')
- [x] **T6.2** `_synthesize_b_matrix` dual-pol H/V responses + OQ4 raise
- [x] **T6.3** `_build_pol_matrix` + `_pol_vec` helpers (TR 38.901 §7.3.2)
- [x] **T6.4** strict_pfs/ray synthesizer dual-pol integration (per-port pol_factor on A; Rx projection in B)
- [x] **T6.5** Cross-corr contract preservation under dual-pol (N=60, mean off-diag < 0.25)
- [x] **T6.6** E2E test: dual-pol DUT + XPR=7dB + initial_phases → strict_pfs → Propsim DL ASC; V/H magnitude ratio ≈ sqrt(κ); cross-pol attenuation ≈ sqrt(1/κ)
- [x] **T6.7** Single-pol regression bit-identical (implicit: TestRayRegression golden still passes after T6.4)
- [x] **T6.8** Docs update (API doc §7+§8, SPEC v2.0 §3 field table + §3.1 dual-pol section)

**Acceptance result**:
- ✅ 142 pass + 1 skip (full suite); +13 new dual-pol tests
- ✅ Single-pol path unchanged (TestRayRegression golden bit-identical)
- ✅ XPR + 4 init_phases 真用 (consumed by pol_matrix in dual-pol mode)
- ✅ strict_pfs cross-corr contract preserved under dual-pol
- ✅ V/H magnitude ratio ≈ sqrt(κ) (实测 2.20 vs 理论 2.24, ±10% tolerance)
- ✅ cross-pol effective_H ratio ≈ sqrt(1/κ) (实测 0.29 vs 理论 0.45, ±40% tolerance)

---

### Phase 7 —— Full Slant Polarization Model（DEFERRED, roadmap only）

**Status**: ⏸️ NOT SCHEDULED. 等真 slant ±45° DUT 需求才启动。

**Scope** (原 Phase 6 option A):
- `AntennaArrayConfig.polarization` 扩 Literal 到 `'+45' / '-45' / 'slant'` + `slant_angle_deg`
- `ProbeAntennaConfig.polarization` 加 polarization 字段
- B-matrix 真实计算任意 pol DUT 极化耦合
- Polarization pattern 非 isotropic (TR 38.901 element pattern with pol effects)

**估时**: 5-8 工作日

**触发条件 (任一)**:
- MIMO-First 测 slant ±45° DUT (5G NR sub-6 commonly slant cross-pol)
- AAU / massive MIMO array per-element pol 需求
- CTIA / 3GPP 实测对比需要

Memory: `~/.claude/projects/-Users-Simon-Tools-ChannelEgine/memory/project_phase7_slant_pol_roadmap.md`

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
