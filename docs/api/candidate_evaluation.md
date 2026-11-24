# 候选评估（P2-110）

`mimo_ota_simulator.candidate_evaluation` 接受 P2-109 每帧每预算的计算结果，也可接受按 P2-108 同一约定整理的已有 ASC。不会启动聚簇、调用设备、生成或写回工程。无 HyperRT 依赖。

先创建不可修改的 `EvaluationPlan`，固定源与成员 SHA256、连续相邻的选择/留出源帧段、预算上限、两项 P95 阈值、目的和模式。再调用 `evaluate_candidates(plan, candidates, provenance=..., manual_candidate=None)`。真实预设选择0–25、留出26–37；31抽头、矩阵误差P95≤1、相位P95≤60°。这些是本次示例约束，不是认证标准。

候选结构为 `ID: {strategy, requested_taps, frames: {源帧整数: {used_taps, metrics, delta}}}`。strategy 为 independent/shared/existing_asc，后者只适用于实际已有ASC参考帧。metrics 原样沿用109的 `methods.candidate.matrix_nmse` 与 `phase_error_deg`，delta 沿用同策略24基线的改善/变差计数。来源对象包含 `source_sha256,members_sha256,time_basis,delay_zero_basis,offsets_hz,port_pairs,amplitude_floor`，可附加适用帧与方法说明。

计算先完成全部选择集统计及约束，按目的误差P95、另一误差P95、声明预算、ID稳定排序，冻结选择及指纹后，才计算所选方案的留出复核。更改留出值不会改变选择指纹。不输出所有候选留出排名。缺帧、任意无效点或预算超限均不能获得达标建议，覆盖率和缺帧仍保留；此保守完整覆盖规则防止用删点提高排名。

返回 `selection` 每候选两项P95、有效/总点数、覆盖率、最差点（源帧/频偏/相位端口对）、实际抽头范围、对同策略基线的改善/变差计数；`holdout` 只有已冻结建议方案的一组独立结果。最差点并列时取最早源帧与最先输入的频偏/端口对。矩阵误差是全部端口矩阵误差，不称作单端口误差。

无候选达标为 `status=no_candidate_meets_constraints,selected_candidate=null,holdout=null`。`manual_candidate` 独立记录操作员指定方案；即使未达标也可以查看 `manual_holdout`，但不变成自动建议或正式通过。参数与选择集逐点输入均有SHA256指纹。`unmatched_frame_review` 单列37帧；其固定成员计算有效不代表有ASC参考。所有38帧的原始扫描仅为描述，不能用于选跨帧参数。

传导与OTA由plan.mode分别建报告；OTA始终标 `ota_status=unverified`，本接口尚未核验场参考。来源说明不自动证明同频同极化、真实时间和时延零点；这些及4×2端口映射、文件安全由接入方核验。`formal_verdict=null,automatic_adoption=false,alignment_status=unverified`。预算是模型复杂度代价，局部变差计数不能被总体P95改善掩盖。

实样复跑：`python3 scripts/export_candidate_evaluation_sample.py`。读取与108/109相同的外部MPDB与members.npz，哈希不匹配立即失败；输出至 `docs/evidence/p2110/`，包括默认约束、零阈值无候选、缺OTA场参考、人工shared/31复核四份报告。人工示例不改变无候选达标结论。脚本不使用全38帧的统计值来挑选参数。

## 内审修正：直接调用的证据校验

已有ASC仅统计 `provenance.existing_asc_reference.applicable_frames` 明确包含的源帧。该字段缺失、为空或无效时不默认覆盖计划帧；`declared_applicable_frames` 只是声明，不能代替实际适用范围。兼容只提供顶层 `asc_applicable_frames` 的直接调用；存在 `existing_asc_reference` 对象时以该对象为准，不得用顶层字段覆盖其空范围。适用范围不会升级物理对齐状态。

所有候选每帧须提供固定4×2八路的 `used_taps`，长度恰好8，每路必须是大于0的整数（不接受布尔值、null或零），同时不超过候选声明预算和冻结计划预算。缺字段、缺路、额外路、无效数值或超预算均将该帧保留为未核验。`unverified_frames` 列出源帧，`excluded_frame_evidence` 逐帧记录原因；这些帧不进入分位数、最差点、抽头范围或改善/变差统计，分母仍保留预定全部帧，因此不能凭缺证获得达标建议。`missing_frames` 单独表示整个帧记录缺失。选择集及所选/人工留出复核均应用相同校验，手动指定不能绕过。

已与 Meta `scripts/mpdb_purpose.py` 核对：实际ASC采用 `existing_asc_reference.applicable_frames`，实际八路数量来自108的 `asc_used_taps` 和109的 `used_taps`；未上传ASC时的 `declared_applicable_frames` 不会让已有ASC候选获得证据。端口对索引限制在0–7。
