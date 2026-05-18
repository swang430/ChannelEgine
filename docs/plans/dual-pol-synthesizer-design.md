# Design: Phase 6 — Dual-Polarization Synthesizer (XPR semantic)

**Status**: DRAFT — pending review
**Parent plan**: [strict-pfs-rollout.md](./strict-pfs-rollout.md)
**Phase**: 6
**Created**: 2026-05-18
**Branch**: `feature/dual-pol-synthesizer`
**Depends on**: Phase 5 merged (PR #5, commit `5e2e3f5`) — `EngineCluster.xpr_linear` + `initial_phases_rad` plumbed

---

## 1. Context

Phase 5 仅 plumb 了 XPR 和 cross-pol initial phases (索引 1-3) 到 EngineCluster；synthesizer 不读它们。Phase 6 让 synthesizer **真用** 这些字段，实现 dual-polarization 数学。

### 当前 dual_polarized 是 stub

`simulator.py:82-86`：
```python
if not chamber_config.dual_polarized:
    b_matrix[r_idx, p_idx] = channel_response
else:
    b_matrix[r_idx, 2*p_idx] = channel_response       # H port
    b_matrix[r_idx, 2*p_idx + 1] = channel_response   # V port  — IDENTICAL!
```

dual_polarized=True 仅"双倍 probe 数"，**H/V 响应完全一样**。`_calculate_weights_for_cluster` 也只把同一个权重复制到两个 pol 槽（L279-282）。

→ **整个 dual-pol 路径目前是 placeholder**。
→ **XPR / cross-pol 字段从来没影响过任何输出**。

### Phase 6 工程驱动

让 strict_pfs / ray synthesizer **真正物理上正确地**emulate dual-polarization 信道：
- H 和 V 端口承载独立信号（不是 identity）
- per-cluster 2×2 polarization mixing matrix 用 XPR 和 4 个 init phases 构造
- DUT 一侧支持任意 polarization (slant 0°/+45°/-45°/etc)
- 让 OTA 测试 cover 真 dual-pol DUT 性能验证（不仅是 SISO 模拟）

---

## 2. 3GPP TR 38.901 §7.3.2 polarization model

Per-ray channel coefficient 数学：

```
                                           [exp(jΦ_θθ)        sqrt(κ⁻¹)exp(jΦ_θφ)]   [F_tx,θ(θ_zod, φ_aod)]
H_ray = [F_rx,θ(θ_zoa, φ_aoa)  F_rx,φ(...)] [sqrt(κ⁻¹)exp(jΦ_φθ)  exp(jΦ_φφ)        ] [F_tx,φ(θ_zod, φ_aod)]
```

字段映射：
- `κ` = XPR_linear（来自 `EngineCluster.xpr_linear`，由 Phase 5 plumb）
- `Φ_θθ, Φ_θφ, Φ_φθ, Φ_φφ` = `cluster.initial_phases` 索引 0-3
- `F_rx,θ / F_rx,φ` = 接收侧 antenna pattern 的 θ / φ 分量
- `F_tx,θ / F_tx,φ` = 发射侧 antenna pattern 的 θ / φ 分量

XPR = 共极化功率 / 交叉极化功率 = `|exp(jΦ_θθ)|² / |sqrt(κ⁻¹)exp(jΦ_θφ)|²` = `κ`

---

## 3. ⚠️ Phase 6 scope 决策点

### 选项 A — 完整 Antenna-Polarization model

**Scope**:
- DUT antenna 加 polarization 字段：`AntennaArrayConfig.polarization: Literal['V', 'H', '+45', '-45', 'slant']` + slant 角度
- 探头 antenna 同样：`ProbeAntennaConfig.polarization` 字段
- B-matrix 真实计算 H/V 探头 → DUT 极化耦合
- per-cluster 2×2 矩阵真用 XPR + 4 phases
- ray/strict_pfs 完整 dual-pol 重写

**工作量**: 5-8 工作日
**Risk**: 高 (改 AntennaArrayConfig schema 影响所有现有路径；ProbeAntennaConfig 同；e2e regression 风险大)
**收益**: 真 dual-pol OTA emulation，匹配 CTIA 01.40 / TR 37.977 设备验证

### 选项 B — 简化 Dual-Pol（probe 双 pol，DUT 假设 V/H pure）

**Scope**:
- 探头：dual_polarized=True 时 H/V 端口分别处理（2×2 极化矩阵作用）
- DUT 假设单一极化 per antenna（每个 antenna 是 V 或 H 之一，由 AntennaArrayConfig 新加 `polarization: Literal['V', 'H']` 字段，默认 'V'）
- per-cluster 2×2 XPR mixing 应用在 W_base
- ray/strict_pfs 部分重写（dual-pol path 跟 single-pol path 共存，by chamber.dual_polarized）

**工作量**: 3-4 工作日
**Risk**: 中 (B-matrix 改动 contained；只 dual-pol 路径有数学改动；single-pol regression 不动)
**收益**: 大部分 OTA 用例 cover；slant pol DUT 需要额外工作（不在此 scope）

### 选项 C — Channel-only Polarization（最小）

**Scope**:
- B-matrix 不动（H/V identity 仍 ok）
- 仅 per-cluster 2×2 XPR mixing 应用在 strict_pfs W_base 合成
- 把 4 polarization channel (HH, HV, VH, VV) 缩到等效 SISO（loss XPR information）
- 不加任何新 AntennaArrayConfig 字段

**工作量**: 1-2 工作日
**Risk**: 低 (改动小)
**收益**: 字面上 "use XPR + init_phases" 满足 Phase 5 plumb 承诺，但**没有真 dual-pol OTA 能力**。基本上是"假做 dual-pol"

---

## 4. 我的推荐：选项 B

理由：
1. **真做的最小可信范围**：选项 A 大部分价值在 slant pol DUT，是边缘场景；选项 C 没真 dual-pol 能力
2. **3-4 天能产出**：跟 Phase 5 节奏一致，不破坏 momentum
3. **schema 增量小**：只加 `AntennaArrayConfig.polarization: Literal['V', 'H'] = 'V'`，单 default 保 backward compat
4. **数学正确**：满足 TR 38.901 §7.3.2 dual-pol channel model（在 DUT pure pol 假设下）
5. **未来可扩展到选项 A**：slant pol 可作 Phase 7 增量（如果有需求）

下面继续按 **选项 B** 设计。

---

## 5. Schema 改动 (选项 B)

### AntennaArrayConfig

```python
class AntennaArrayConfig(BaseModel):
    ...
    polarization: Literal['V', 'H'] = Field('V', description="Polarization: V (vertical) or H (horizontal). Single pol per antenna. Default V for backward compat.")
```

Default `'V'` 保现有 (single pol) callers behavior 不变。

### ChamberConfig / ProbeAntennaConfig

不动。现有 `chamber.dual_polarized: bool` 字段语义保留：True 表示每物理探头有 H+V 端口（2 个）。

### EngineCluster

不动。`xpr_linear` 和 `initial_phases` 已在 Phase 5 plumb。

---

## 6. Synthesizer 改动

### B-matrix dual-pol（`_synthesize_b_matrix`）

当 `chamber.dual_polarized=True`：

```python
# DUT antenna polarization vector
if rx_antenna.polarization == 'V':
    dut_pol = np.array([1.0, 0.0])  # [θ-component, φ-component]
elif rx_antenna.polarization == 'H':
    dut_pol = np.array([0.0, 1.0])

# Probe H port -> matches DUT V if probe is V-pol; matches DUT H if probe is H-pol
# Standard assumption: probe H port is φ-polarized, V port is θ-polarized
probe_v_response = dut_pol[0] * physical_response  # θ component
probe_h_response = dut_pol[1] * physical_response  # φ component

b_matrix[r_idx, 2*p_idx]     = probe_v_response  # V port (θ)
b_matrix[r_idx, 2*p_idx + 1] = probe_h_response  # H port (φ)
```

For V-pol DUT: V probe ports drive the antenna; H ports contribute 0
For H-pol DUT: H probe ports drive; V ports contribute 0

### ray/strict_pfs synthesis dual-pol

Per ray, compute 2×2 polarization matrix from XPR + init_phases:

```python
# Build per-ray 2×2 polarization matrix
xpr_linear = cluster.xpr_linear if cluster.xpr_linear is not None else default_xpr  # 7 dB default
sqrt_kappa_inv = 1.0 / np.sqrt(xpr_linear)
phi = cluster.initial_phases  # (4,) [Θθθ, Θθφ, Θφθ, Θφφ]
pol_matrix = np.array([
    [np.exp(1j*phi[0]),                sqrt_kappa_inv * np.exp(1j*phi[1])],
    [sqrt_kappa_inv * np.exp(1j*phi[2]), np.exp(1j*phi[3])]
])  # (2, 2)

# DUT side projection (Rx polarization)
rx_pol_vec = ... (V or H from rx_antenna.polarization)

# Tx side projection (Tx polarization)
tx_pol_vec = ... (V or H from tx_antenna.polarization)

# Per-ray scalar coefficient = rx_pol_vec @ pol_matrix @ tx_pol_vec
ray_pol_scalar = rx_pol_vec @ pol_matrix @ tx_pol_vec

# Apply to W_base
W_base_pol = ray_pol_scalar * np.sqrt(p_ray) * (w_map_ray @ g_tx_ray)
```

For single-pol DUT (V or H), this collapses to a scalar coefficient per ray. The full polarization model is exercised only when DUT or probes are dual-pol.

### Behavior matrix

| chamber.dual_pol | tx.pol | rx.pol | Behavior |
|---|---|---|---|
| False | V/V | V/V | Single-pol path (Phase 5 unchanged) |
| True | V/V | V/V | Dual-pol path; cross-pol leak = 0 (V_rx @ pol_mat @ V_tx = exp(jΦ_θθ) only) |
| True | V/V | H/H | Dual-pol path; only cross-pol terms contribute (V_rx @ pol_mat @ H_tx = sqrt(κ⁻¹)exp(jΦ_θφ)) |
| True | V/H mixed | V/H mixed | Per-antenna polarization → effectively 2×N MIMO |

---

## 7. Sub-task 分解

| ID | 任务 | 依赖 | 工作量 |
|---|---|---|---|
| **T6.1** | `AntennaArrayConfig.polarization` field + 默认 V + validation tests | 无 | 0.3d |
| **T6.2** | `_synthesize_b_matrix` dual-pol 真实 H/V 响应 (chamber.dual_polarized=True) | T6.1 | 0.5d |
| **T6.3** | per-ray 2×2 polarization matrix helper + XPR default fallback | 无 | 0.5d |
| **T6.4** | strict_pfs / ray synthesizer integrate polarization matrix | T6.1, T6.3 | 1d |
| **T6.5** | strict_pfs cross-corr contract preservation test (T2.1 contract under dual-pol) | T6.4 | 0.3d |
| **T6.6** | E2E test: dual-pol DUT + XPR=7dB + Propsim DL ASC | T6.2, T6.4 | 0.5d |
| **T6.7** | regression: single-pol path bit-identical to pre-Phase-6 | T6.4 | 0.3d |
| **T6.8** | docs update: dual-pol behavior matrix + AntennaArrayConfig.polarization + EngineCluster.xpr_linear status from "plumbed" → "used" | T6.6 | 0.3d |

**Critical path**: T6.1 → T6.2 + T6.3 → T6.4 → T6.5/T6.6/T6.7 → T6.8
**总工作量**: ~3-4 工作日

---

## 8. Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Single-pol path 回归 (T6.4 synthesizer 改动) | Med | High | T6.7 explicit regression test ray + strict_pfs golden bit-identical; full suite regression每 sub-task |
| strict_pfs cross-corr contract broken under dual-pol | Med | High | T6.5 复用 T2.1 N=80 contract test |
| AntennaArrayConfig schema change 影响所有现有 callers | Low | Med | default='V' 不破现有；Pydantic field add 不 breaking |
| XPR 默认值争议 (TR 38.901 典型 7 dB UMa NLOS vs 8 dB UMi NLOS) | Low | Low | 用 7 dB 默认，文档标"per TR 38.901 §7.7-2 typical XPR_NLOS_mean"，caller 显式设 xpr_db 优先 |
| dual_polarized=True 但 caller 没设 xpr_db 历史 callers | Med | Med | XPR default 7 dB + 一次 warning（不阻塞），让既有 dual-pol callers 继续 work |

---

## 9. Out of Scope (Phase 6 不做)

- Slant ±45° polarization (Phase 7 if needed)
- DUT antenna 任意 polarization angle (只 V/H pure)
- ProbeAntennaConfig.polarization field (假设标准 V probes = θ, H probes = φ)
- Polarization pattern 非 isotropic (TR 38.901 element pattern with pol effects)
- AAU / massive MIMO dual-pol arrays
- Time-varying polarization rotation
- Calibration vector polarization-aware (current cal_vector 一维，dual-pol case 需要 (2P,) 维)
  - 这是 follow-up 项目；T6.2 检查 cal_vector.shape 跟 num_effective_probes 一致即可

---

## 10. Open Questions

- [ ] OQ1: 选项 A / B / C —— 推荐 **B**，等你确认
- [ ] OQ2: XPR 默认值（caller 没设时）：7 dB（TR 38.901 UMa NLOS typical）/ 0 dB / inf（无 cross-pol leak） —— 推荐 **7 dB + warning**
- [ ] OQ3: AntennaArrayConfig.polarization 默认 'V' 还是其他？—— 推荐 **'V'** (vertical 是常用 LTE/5G default)
- [ ] OQ4: dual_polarized=False + tx/rx.polarization='H' 时如何处理？—— 推荐 raise (无意义 config) / 或 silent treat as 'V' / 或 warn

---

## 11. Definition of Done

- [ ] T6.1-T6.8 全 pass
- [ ] 全 suite regression 0 fail
- [ ] strict_pfs cross-corr contract (T2.1) 在 dual-pol 下仍满足
- [ ] strict-pfs-rollout.md Phase 6 标 done
- [ ] CLAUDE.md cross-project context 更新提到 Phase 6 完成 + dual-pol 可用
- [ ] PR review approval

---

## Changelog

- 2026-05-18: Initial design draft (Claude + @swang430 待 review)
