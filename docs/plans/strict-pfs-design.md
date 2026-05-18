# Design: Strict PFS Synthesis Method

**Status**: DRAFT — pending review
**Parent plan**: [strict-pfs-rollout.md](./strict-pfs-rollout.md)
**Phase**: 1
**Created**: 2026-05-18
**Owner**: @swang430
**Branch**: `feature/strict-pfs`（待创建）

---

## 1. Context

Phase 0 已完成 jitter UI 校正 + cal/jitter 冲突 guard（commit `9eb1a06`）。
Phase 3 已完成 cluster/pinv deprecation（commit `030e2ce`）。

本文档是 Phase 1 (strict_pfs 核心实现) 的**设计契约**：在写代码前锁定
数学、数据结构、API 契约、F64 兼容性、UL/DL 行为。Sub-task 都引用本文。

**核心 driver**（再次重申）：strict PFS = 替代输出端相位校准的工程方案，
不是精度优化。判据 `E[x_i(t) · x_j*(t)] = 0 for i≠j` per-realization 即满足。

---

## 2. 数学模型

### 现状（`ray` mode）

```
对每 cluster 拆 20 sub-ray:
  phi_rnd     = U[-π, π]                          # 标量，per-ray，跨 probe 共享
  W_base[p,t] = sqrt(p_ray) · w_map[p] · g_tx[t]  # (N_probes, N_tx)，实数权重 × 复 steering
  A_ray[p,t]  = W_base[p,t] · exp(j·phi_rnd)      # 标量 phase 广播到所有 (p,t)
  A_total    += A_ray
```

Probe 间二阶相关：
`E[A_i · A_j*] ∝ W_base[i,t] · W_base[j,t]*` —— **非零**，因为相位项消掉。
所以 ray mode 给出**几何决定的 cross-coupling**，需要大量 realization
平均 + jitter ±π 才能近似 strict PFS。

### Strict PFS（新 `strict_pfs` mode）

**唯一改动**：phi 从标量 → per-probe 独立向量

```
对每 cluster 拆 20 sub-ray:
  phi_rnd[p]  = U[-π, π], iid per probe         # (N_probes,) 向量
  W_base[p,t] = sqrt(p_ray) · w_map[p] · g_tx[t]   # 不变
  A_ray[p,t]  = W_base[p,t] · exp(j·phi_rnd[p])    # 向量 phase per probe
  A_total    += A_ray
```

Probe 间二阶相关：
`E[A_i · A_j*] ∝ W_base[i,t] · W_base[j,t]* · E[exp(j(phi_i - phi_j))]`
`= 0 for i≠j`（独立均匀分布 → 期望为 0）—— **per-realization 即满足**

### Doppler 处理（保持现状）

Doppler `fd = v · k_hat_rx / λ` 是**物理量**（取决于 ray 方向，不取决于 probe），
所以 `doppler_hz` 保持 per-ray 标量。时间演化：
`A_ray(t) = W_base · exp(j·phi_rnd[p]) · exp(j·2π·fd·t)`

Phi 是**初始随机相位**（fixed per realization），doppler 是**时间演化**。
两者正交，strict PFS 的统计性质对 doppler 透明。

### Phase calibration cert 叠加（D8 验证目标）

Cal cert 通过 `b_matrix *= calibration_vector` (simulator.py L77-79) 注入。
最终 `effective_H = B · A_total`，cal 在 B 维度，strict 在 A 维度，**正交叠加**。
区别于 jitter+cal：jitter 跟 cal 都在 A 维度的 probe phase 上打架。

---

## 3. 数据结构变更

### `ray_components[i]` schema（向后兼容扩展）

| 字段 | 现状（ray） | strict_pfs |
|---|---|---|
| `W_base` | `complex (N_probes, N_tx)` | 不变 |
| `phi_init` | `float`（标量） | **`complex (N_probes,)` 向量** |
| `doppler_hz` | `float` | 不变 |
| `cluster_index` | `int` | 不变 |

**判别方式**：消费者通过 `np.isscalar(rc['phi_init'])` 区分两种 mode；
或者 strict_pfs 加新字段 `pfs_mode: bool`（显式 flag）。
**决策**：用 `np.isscalar()` 隐式判别，避免冗余字段。

### `target_H_matrix` / `transmit_mapping_matrix_A`

形状不变：`(N_rx, N_tx)` 和 `(N_probes, N_tx)`。strict_pfs 完全 drop-in。

### `channel_impulse_response`（F64 export 输入）

形状不变：`List[{delay, coeff:(N_probes, N_tx), cluster_index}]`。
strict_pfs 跟 ray mode 完全同 shape，**F64 静态 export 路径零改动**。

---

## 4. F64 Export 兼容性（D6 假设验证）⚠️

**关键发现**：`exporters.py:250` 在 dynamic doppler tensor 计算中做：
```python
w = r['W_base'] * np.exp(1j * r['phi_init']) * scaling_factor * cal_vector
```

- 当前 `phi_init` 是标量，`np.exp(1j·标量)` 仍标量，广播到 `(N_probes, N_tx)`
- Strict_pfs `phi_init` 是 `(N_probes,)` 向量，`np.exp(...)` 是 `(N_probes,)`
- 与 `W_base (N_probes, N_tx)` 相乘：**numpy 不会自动 axis-align**，需要 reshape

**修法**（**最小侵入**）：在 exporters.py 改一行：
```python
phi = r['phi_init']
phi_factor = np.exp(1j * phi) if np.isscalar(phi) else np.exp(1j * phi)[:, None]
w = r['W_base'] * phi_factor * scaling_factor * cal_vector
```

**结论**：
- F64 **格式** (.asc, sample rate, 文件命名) 零改动
- F64 **生成代码**需要 1 处兼容适配（5 行 diff）
- D6 假设修正为："F64 格式不动；exporter 加 minimal 兼容 patch"
- T1.6 是 **GO/NO-GO gate**：必须先证明这个小改动可工作，再投入完整实现

---

## 5. UL/DL 互易（需 T1.5 验证）

`_build_ul_ray_components` (simulator.py L155-179) 现状：
```python
if mode == "conjugate_transpose":
    w_ul = rc['W_base'].conj().T          # (N_probes, N_tx) → (N_tx, N_probes)
    phi_ul = -rc['phi_init']              # 标量 / 向量 都 OK（numpy negation）
    doppler_ul = -rc['doppler_hz']
```

**潜在问题**：W_base 转置后 "probe 维度" 与 "tx 维度" 互换，
而 strict_pfs 的 `phi_init[p]` 仍标在 "DL 时的 probe" 上。UL reciprocity
应该让 phi vector 维度跟 tx-side 对齐？

**需要 T1.5 explicit answer**：
- Option A: UL 路径下 phi_init 重 broadcast 到新维度（保持互易性）
- Option B: UL 路径下重新独立采 phi（独立 UL fading，不互易）
- Option C: 维持现状（数学上 UL 不是严格互易，但代码不改）

**默认选 A**（保物理互易性），但需要 spec check。

---

## 6. UI 变更

### app.py

- `synthesis_method` selectbox：增加 `'strict_pfs'` 第 4 个选项
  - 顺序：`('pinv', 'cluster', 'ray', 'strict_pfs')`
  - `index=2` 保持（默认 ray，符合 D9 保守 opt-in）
  - help text 更新："**Strict PFS（推荐，无需输出相位校准）** / Ray（传统/平均场景）/ Cluster (deprecated) / Pinv (deprecated, broken F64)"
- `probe_phase_jitter` checkbox：当 `synthesis_method == 'strict_pfs'` 时 disabled
  - tooltip："strict_pfs 数学上已实现 per-probe 独立 fading，jitter 多余且会污染"

### gui.py

同样改 combobox 选项 + 同样的 jitter 互斥逻辑。

---

## 7. API 公共契约（T1.9）

**Public API**: `MIMO_OTA_Simulator.run(chamber_config, channel_config, synthesis_method, probe_phase_jitter)`

新增 `docs/api/mimo_ota_simulator.md`（或 README 一节）说明：
- 4 个 synthesis_method 的语义、推荐场景、deprecation 状态
- jitter 跟 cal 互斥
- 返回 dict 的稳定字段
- **不**承诺 `run_with_external_clusters` / `OTASimulator` 等 MIMO-First 期望的别名 —— 由 MIMO-First 那边适配（D11）

---

## 8. 测试策略

| 测试 | 类型 | 通过条件 |
|---|---|---|
| `test_strict_pfs_smoke` | unit | strict_pfs path 跑通无 NaN/Inf |
| `test_strict_pfs_export_compat` | integration | strict_pfs → F64 .asc export 成功（GO/NO-GO 复用） |
| `test_strict_pfs_correlation` (Phase 2 T2.1) | statistical | `E[A_i·A_j*]/sqrt(...)` < 0.1 for i≠j，N=10 realizations |
| `test_spatial_correlation_convergence` (T2.2) | statistical | strict_pfs 收敛 ~L·M× 快于 ray-w-jitter |
| `test_strict_pfs_cal_superposition` (T2.3) | integration | strict_pfs + cal → effective_H 包含 cal phase shift |
| `test_ray_regression` (T2.4) | regression | ray/cluster/pinv 输出 vs 既有 golden bit-identical |

---

## 9. Sub-task 分解（risk-first 排序）

| ID | 任务 | 依赖 | 工作量 | 风险 |
|---|---|---|---|---|
| **T1.6** | F64 兼容性 POC：scaffold minimal strict_pfs + 验证 exporter broadcast fix | 无 | 4h | **High** — go/no-go gate |
| T1.1 | `simulator.py` `synthesis_method` Literal 加 `'strict_pfs'` + 路由 branch | T1.6 pass | 1h | Low |
| T1.2 | `ray_components.phi_init` 数据结构 lock（vector 形式，`np.isscalar` 判别） | T1.6 pass | 0.5h | Low |
| T1.3 | strict_pfs 完整合成路径实现 | T1.1, T1.2 | 1d | Med |
| T1.4 | Tx steering / w_map 复用验证（应该完全 reuse） | T1.3 | 1h | Low |
| T1.5 | UL/DL ray_components 在 vector phi_init 下的正确性（选 A/B/C + 测试） | T1.3 | 4h | Med |
| T1.7 | UI 加 strict_pfs 选项（app.py + gui.py） | T1.3 done | 2h | Low |
| T1.8 | UI strict_pfs ⇄ jitter 互斥逻辑 | T1.7 | 1h | Low |
| T1.9 | `MIMO_OTA_Simulator.run()` public API 文档化 | T1.3 done | 2h | Low |

**Critical path**: T1.6 → T1.1 + T1.2 → T1.3 → (T1.4, T1.5, T1.7, T1.9 并行) → T1.8
**总工作量**: ~3 工作日（连续）或 4-5 天（带 review 间隙）

---

## 10. Risks & Mitigation（design 阶段）

| Risk | Mitigation |
|---|---|
| T1.6 发现 F64 兼容比预期复杂 | 早 stop；改回 design 阶段；考虑 phi_init shape `(N_probes, 1)` 替代方案 |
| T1.5 UL 互易性 spec 不明 | 默认 Option A + 测试；如有疑问搁置 UL 分支（DL-only release） |
| `np.isscalar` 判别在 numpy 0-d array 上的边缘情况 | 测试覆盖；必要时改用 `np.ndim(phi) == 0` |
| 现有 `ray_components` 消费者（除 exporter 外）有未发现的 phi_init 用法 | T1.6 阶段 grep 全 codebase `phi_init` |

---

## 11. Out of Scope（Phase 1 不做）

- Phase 2 的 statistical validation tests（独立 Phase）
- D11 adapter for MIMO-First（由 MIMO-First 那边适配）
- 改默认 method（保 ray，D9）
- Cluster mode 修复（已 deprecate）
- Pinv F64 修复（已 deprecate）
- 性能优化（先实现，后 profile）

---

## 12. Open Questions（设计阶段）

- [ ] OQ1: T1.5 UL/DL 选 A/B/C 哪个？需要 spec 引用（TR 37.977 是否要求 strict PFS 互易性？）
- [ ] OQ2: `np.isscalar` vs `np.ndim == 0` vs 显式 `pfs_mode` flag —— 哪个最 robust？
- [ ] OQ3: F64 exporter 修改是 ship 进 strict_pfs PR，还是单独一个 fix commit？
  - 倾向：一起 PR，因为修改是为 strict_pfs 服务的

---

## Changelog

- 2026-05-18: Initial design draft（Claude + @swang430 待 review）
