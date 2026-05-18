# Design: Strict PFS Statistical Validation (Phase 2)

**Status**: DRAFT — pending review
**Parent plan**: [strict-pfs-rollout.md](./strict-pfs-rollout.md)
**Phase**: 2
**Created**: 2026-05-18
**Branch**: `feature/strict-pfs-validation`
**Depends on**: Phase 1 (merged in PR #1, commit `5290c3a`)

---

## 1. Context

Phase 1 实现了 `synthesis_method='strict_pfs'`，per-(probe, cluster) 独立 fading。
Phase 1 的 acceptance 只是"代码跑通 + F64 export 成功 + 4 个 smoke test"。

Phase 2 的目标是**数学正确性证明**：strict_pfs 确实满足 strict PFS 判据，
且优于 ray+jitter 的近似方式，且与 phase calibration 叠加正确。

这些测试是**支撑"用 strict_pfs 替代输出端相位校准"工程决策的证据基础**。
没有 Phase 2，strict_pfs 只是个新选项；有了 Phase 2，strict_pfs 是个**有证据
支持**的推荐方法。

---

## 2. Math contracts to verify

### Contract A: per-realization cross-correlation → 0

**Claim**: 对 strict_pfs，per-realization `<A_i(·) · A_j*(·)>_t = 0 for i≠j` 期望意义下成立。
（实测 noise floor ~ 1/sqrt(N_samples_per_realization)。）

**Verification (T2.1)**: N realizations，每个抽 strict_pfs A 矩阵，
平均 sample cross-correlation。期望 i≠j 项 →0 随 N 增加；ray mode 不收敛到 0。

### Contract B: spatial correlation 收敛速度

**Claim**: strict_pfs 模式下 `R_hat = mean(A_n · A_n^H)` 收敛速度比 ray+jitter 快 ~L·M 倍
（L = clusters, M = sub-rays per cluster = 20，3GPP 典型 L·M ~ 200-500）。
理论原因：ray+jitter 实际"独立 RNG draws"数量比 strict_pfs 少 L·M 倍。

**Verification (T2.2)**: 同 target_R，比较两个 mode 的 Frobenius distance
`||R_hat(N) - R_target||_F` 衰减曲线。strict_pfs 应该在 N 小时就收敛。

### Contract C: phase cal 叠加（区别于 jitter 互斥）

**Claim**: strict_pfs + cal_vector → effective_H 包含 cal 的 per-probe phase shift
（不像 jitter 把 cal phase swamp 掉）。Plan D8 假设。

**Verification (T2.3)**: 同 seed 跑两次（cal off / cal on），比较 A 矩阵在 probe 维度
的 phase 关系。cal off 的 phase 跟 cal on 的 phase 差应该等于 cal vector 提供的 phase。

### Contract D: deprecated methods 仍 bit-identical 跟 golden（regression safety net）

**Claim**: Phase 1 ray/strict_pfs 合并 elif 不应改变 ray 路径输出；ray/cluster/pinv
仍跟 v1.0 golden 文件 bit-identical（test_regression.py 现成）。

**Verification (T2.4)**: 跑现有 test_regression + 必要时补 ray/cluster/pinv 独立 golden。

---

## 3. Test design

### T2.1 — `test_strict_pfs_correlation`

```python
# tests/test_strict_pfs_correlation.py (new file)

def test_strict_pfs_zero_cross_correlation_per_realization():
    """Verify strict_pfs satisfies E[A_i · A_j*] = 0 for i≠j."""
    P = 16
    N_REALIZATIONS = 200  # noise floor ~ 1/sqrt(200) ~ 0.07
    THRESHOLD_OFF_DIAG = 0.15  # 2x noise floor for safety

    R_strict_sum = np.zeros((P, P), dtype=complex)
    R_ray_sum    = np.zeros((P, P), dtype=complex)

    for seed in range(N_REALIZATIONS):
        channel.random_seed = seed
        a_strict = sim.run(..., synthesis_method='strict_pfs')['transmit_mapping_matrix_A']
        a_ray    = sim.run(..., synthesis_method='ray')['transmit_mapping_matrix_A']
        # A shape (P, Tx) — take column 0 as the probe-domain vector
        x_strict = a_strict[:, 0]
        x_ray    = a_ray[:, 0]
        # Outer product accumulates
        R_strict_sum += np.outer(x_strict, x_strict.conj())
        R_ray_sum    += np.outer(x_ray,    x_ray.conj())

    R_strict_mean = R_strict_sum / N_REALIZATIONS
    R_ray_mean    = R_ray_sum    / N_REALIZATIONS

    # Normalize to correlation coefficients
    d_strict = np.sqrt(np.abs(np.diag(R_strict_mean)))
    rho_strict = R_strict_mean / np.outer(d_strict, d_strict)
    d_ray = np.sqrt(np.abs(np.diag(R_ray_mean)))
    rho_ray = R_ray_mean / np.outer(d_ray, d_ray)

    off_diag_strict = np.abs(rho_strict - np.diag(np.diag(rho_strict)))
    off_diag_ray    = np.abs(rho_ray    - np.diag(np.diag(rho_ray)))

    # strict_pfs: max off-diag < threshold (converges to 0)
    assert off_diag_strict.max() < THRESHOLD_OFF_DIAG
    # ray: max off-diag >> threshold (geometry coupling visible)
    assert off_diag_ray.max() > 2 * THRESHOLD_OFF_DIAG
```

**Risk**: threshold 选择 brittle —— 若 noise floor 估计错或 RNG variance 高可能 flaky。
**Mitigation**: N_REALIZATIONS 调到 200 给足头 room；threshold 设 2x 理论 noise floor。

### T2.2 — `test_spatial_correlation_convergence`

```python
def test_strict_pfs_converges_faster_than_ray_jitter():
    """strict_pfs 在 N_drops 较小时已接近 target R；ray+jitter 需要 L·M× 更多."""
    P = 16
    Ns = [10, 50, 200]

    target_R = compute_theoretical_target_R(...)  # 从 channel config 解析得到

    for N in Ns:
        R_strict = average_over_drops(sim, N, method='strict_pfs')
        R_ray_jit = average_over_drops(sim, N, method='ray', jitter=True)
        d_strict = frobenius_distance(R_strict, target_R)
        d_ray_jit = frobenius_distance(R_ray_jit, target_R)
        # At each N, strict_pfs should be closer to target
        assert d_strict < d_ray_jit, f"At N={N}: strict={d_strict:.3f} ray_jit={d_ray_jit:.3f}"
```

**Risk**: `target_R` 怎么算？没有简单封闭解。
**Mitigation 选 A**: 用 N_REF=2000 drops 的 ray+jitter 收敛结果作为 empirical target。
**Mitigation 选 B**: 用 strict_pfs N_REF=2000 作为 target（自我参照，弱）。
**Recommend**: Option A —— ray+jitter 大平均是公认 ground truth approximation。

### T2.3 — `test_strict_pfs_cal_superposition`

```python
def test_cal_phase_preserved_under_strict_pfs():
    """同 seed strict_pfs，cal off 和 cal on 的 A 应在 probe-domain 差出 cal phase."""
    cal_phase = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
    cal_vec = np.exp(1j * cal_phase).reshape(-1, 1)  # (P, 1)

    # Reset RNG to same seed both runs
    np.random.seed(42)
    r_off = sim.run(chamber_no_cal, channel, synthesis_method='strict_pfs')
    np.random.seed(42)
    r_on  = sim.run(chamber_with_cal, channel, synthesis_method='strict_pfs')

    # Cal is applied to B matrix (b_matrix *= cal_vector at L77-79 simulator.py)
    # so effective_H = B_cal @ A = (B_no_cal * cal_vector) @ A
    # → effective_H_on / effective_H_off should reveal the cal phase pattern
    ratio = r_on['effective_H_matrix'] / r_off['effective_H_matrix']
    # check magnitude ratio is 1, phase delta is cal_phase
    ...
```

**Risk**: cal 注入路径是 B matrix 不是 A，effective_H 是 B·A，phase 关系受 B 的几何影响。需要谨慎拆解。
**Mitigation**: 简化为 1 probe / 1 antenna 的极端 case 验证 phase pass-through；或直接比较 B 矩阵 (`r_off['B_matrix']` vs `r_on['B_matrix']`) 而不是 effective_H。后者更干净。

### T2.4 — Regression 加固

现有 `test_regression.py` 跑 `golden_UMa_LOS_2x2_101.npz` + `golden_UMi-StreetCanyon_NLOS_4x2_102.npz`。
检查这俩 golden 用哪个 synthesis_method 生成（estimate `pinv` 因为是 v1.0 默认）。

**Action**:
1. 看 golden 文件生成时的 method（grep create_golden_file.py）
2. 若只覆盖 pinv，补 ray + cluster golden（用相同 seed 跑当前代码生成 baseline）
3. strict_pfs **不**需要 golden（per-realization stochastic + 期望强度未定，inappropriate for golden）

---

## 4. Sub-task 分解 + 排序

| ID | 任务 | 风险 | 依赖 |
|---|---|---|---|
| T2.4 | Regression 加固（看 golden 覆盖、补缺） | Low | 无 |
| T2.3 | cal superposition (拆 B vs A 验证) | Med | 无 |
| T2.1 | per-realization cross-corr → 0 | Low | 无 |
| T2.2 | 收敛速度对比 (target_R 选定 + Frobenius) | High（target_R 决策） | T2.1 通过提高信心 |

**推荐顺序**: T2.4 → T2.3 → T2.1 → T2.2。先低风险打底，高风险 T2.2 最后。
但也可以并行 T2.1+T2.3+T2.4，最后 T2.2。

---

## 5. Risks & Mitigation

| Risk | Mitigation |
|---|---|
| 测试 flakiness（统计 threshold 难定） | 大 N（200+）+ threshold 设 2x noise floor + 用 fixed seed 控制可重复 |
| target_R 没有 closed-form | Empirical baseline (ray+jitter N=2000) |
| CI 时间过长（N=2000 慢） | T2.2 用 N=200，N=2000 留作 "manual_only" mark；T2.1 N=200 应 < 30s |
| cal superposition 测试边界（B 矩阵的几何影响） | 用 1×1 极端 case 隔离 phase pass-through 验证 |
| Phase 1 ray 路径回归 | T2.4 加固 + test_regression 现成 7/7 仍 pass |

---

## 6. Out of Scope

- GCM Studio 金标准 cross-validation (D10, MIMO-First scope)
- Performance profiling
- 任何 strict_pfs 实现层 bug fix（如果 T2.x 发现就回 Phase 1 follow-up PR）
- Doppler / time-series 行为的统计测试（独立未来工作）

---

## 7. Open Questions

- [ ] OQ1: target_R 选 ray+jitter N_REF=2000 还是 closed-form? (推荐前者)
- [ ] OQ2: T2.4 是否需要补新 golden 文件 (ray/cluster)? 看 grep 结果定
- [ ] OQ3: T2.2 失败时（strict_pfs 没真 faster）回 Phase 1 fix 还是 lower expectation? (推荐前者，是 spec contract)

---

## 8. Definition of Done

- [ ] T2.1-T2.4 全 pass，至少 1 个 reviewer approval
- [ ] 测试运行时间 < 60s（CI friendly）
- [ ] 任何发现的 strict_pfs bug 已 fix 或 file follow-up issue
- [ ] strict-pfs-rollout.md Phase 2 标 done
- [ ] CHANGELOG / README 提及 strict_pfs 已通过统计验证

---

## Changelog

- 2026-05-18: Initial design draft（Claude + @swang430 待 review）
