"""P2-108 同预算离线计算；不接设备，不生成或采纳工程，不给正式判定。

复系数已含载频相位，频响仅用相对载频偏移。时延始终使用输入零点，
禁止逐口减最早时延或拟合相位以掩盖差异。返回 JSON 安全的逐点量。
"""
from __future__ import annotations

import numpy as np


def _real(value, name):
    a = np.asarray(value)
    if np.iscomplexobj(a) or a.dtype.kind not in 'iuf':
        raise ValueError(f'{name}必须为实数')
    a = a.astype(float)
    if not np.isfinite(a).all():
        raise ValueError(f'{name}必须为有限数值')
    return a


def _positive(value, name):
    a = _real(value, name)
    if a.ndim != 0 or a <= 0:
        raise ValueError(f'{name}必须为有限正数')
    return float(a)


def _paths(coefficients, delays_s):
    h = np.asarray(coefficients, dtype=complex)
    t = _real(delays_s, '时延')
    if h.ndim != 1 or h.shape != t.shape or not np.isfinite(h).all() or np.any(t < 0):
        raise ValueError('复系数和非负时延须为同长有限一维数组')
    return h, t


def frequency_response(coefficients, delays_s, offsets_hz):
    """H(Δf)=Σh exp(-j2πΔfτ)，输入时延单位秒，返回复数组。"""
    h, t = _paths(coefficients, delays_s)
    f = _real(offsets_hz, '频偏')
    if f.ndim != 1 or not len(f):
        raise ValueError('至少需要一个频点')
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.exp(-2j * np.pi * f[:, None] * t) @ h
    if not np.isfinite(result).all():
        raise ValueError('频响超过可表示范围')
    return result


def member_taps(coefficients, delays_s, membership, *, tap_budget=24):
    """已有成员表相干和及每端口功率加权时延；超预算拒绝，不截簇。"""
    h, t = _paths(coefficients, delays_s)
    labels = np.asarray(membership)
    if (type(tap_budget) is not int or tap_budget < 1 or labels.shape != h.shape
            or labels.dtype.kind not in 'iu' or np.any(labels < 0)):
        raise ValueError('成员须为每条射线对应的非负整数，预算须为正整数')
    ids = np.unique(labels)
    if len(ids) > tap_budget:
        raise ValueError('成员簇数超过同预算，不能静默截断')
    coeff, delay = [], []
    for cid in ids:
        sub = labels == cid
        with np.errstate(over='ignore', invalid='ignore'):
            w = abs(h[sub])**2
            total = w.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError('簇功率必须为可表示的正数')
        coeff.append(h[sub].sum())
        delay.append(float((w / total) @ t[sub]))
    coeff, delay = _paths(coeff, delay)
    return dict(cluster_ids=ids, coefficients=coeff, delays_s=delay,
                requested_taps=tap_budget, used_taps=len(ids))


def power_summary(coefficients, *, input_power_w=None):
    """载频相干功率增益和抽头功率和均非 dBm；绝对功率需输入参考。"""
    h, _ = _paths(coefficients, np.zeros(len(coefficients)))
    with np.errstate(over='ignore', invalid='ignore'):
        coherent = float(abs(h.sum())**2)
        taps = float(np.sum(abs(h)**2))
    if not np.isfinite([coherent, taps]).all():
        raise ValueError('功率超过可表示范围')
    received = None
    if input_power_w is not None:
        p = _real(input_power_w, '输入参考功率')
        if p.ndim != 0 or p < 0:
            raise ValueError('输入参考功率须为非负标量')
        received = float(p) * coherent
        if not np.isfinite(received):
            raise ValueError('接收功率超过可表示范围')
    return dict(carrier_coherent_gain=coherent, tap_power_sum=taps,
                input_power_w=input_power_w, absolute_received_power_w=received)


def _nullable(array):
    return [float(v) if np.isfinite(v) else None for v in array]


def compare_responses(reference, methods, offsets_hz, port_pairs, *, amplitude_floor,
                      time_basis=None, delay_zero_basis=None, simulated=False):
    """比较一个源帧的 [port, frequency] 矩阵。所有方法共用相位有效掩码。

    matrix_nmse 为每频点全端口 Frobenius 误差平方比；相位误差为
    abs(arg(Hm[a]conj(Hm[b])conj(Hr[a])Hr[b]))，度制。幅度严格大于
    下限才可用。零参考能量没有可定义 NMSE。依据字符串仅记录，不证明对齐。
    """
    floor = _positive(amplitude_floor, '幅度下限')
    f = _real(offsets_hz, '频偏')
    raw = np.asarray(reference, dtype=complex)
    if raw.ndim != 2 or not raw.size or f.shape != (raw.shape[1],):
        raise ValueError('频响须为非空端口×频点矩阵，与频偏一一对应')
    if not methods or not np.isfinite(raw).all():
        raise ValueError('需要至少一个候选及有限参考频响')
    arrays = {}
    for name, value in methods.items():
        a = np.asarray(value, dtype=complex)
        if not isinstance(name, str) or not name or a.shape != raw.shape or not np.isfinite(a).all():
            raise ValueError('候选方法须有名称、相同维度和有限频响')
        arrays[name] = a
    pairs = list(port_pairs)
    if not pairs or any(len(p) != 2 or any(type(i) is not int for i in p)
                        or not 0 <= p[0] < p[1] < raw.shape[0] for p in pairs) or len(set(map(tuple, pairs))) != len(pairs):
        raise ValueError('端口对须唯一且满足 0≤a<b<端口数')
    usable = np.minimum.reduce([abs(a) for a in (raw, *arrays.values())]) > floor
    masks = np.asarray([usable[a] & usable[b] for a, b in pairs])
    # 按频点缩放全部模型，避免正常有限复数在平方时上溢。
    scale = np.maximum.reduce([abs(a).max(axis=0) for a in (raw, *arrays.values())])
    scale[scale == 0] = 1
    ref = raw / scale
    energy = np.sum(abs(ref)**2, axis=0)
    valid = energy > 0
    output = {}
    for name, a in arrays.items():
        nmse = np.full(len(f), np.nan)
        np.divide(np.sum(abs(a/scale-ref)**2, axis=0), energy, out=nmse, where=valid)
        errors = []
        for k, (left, right) in enumerate(pairs):
            # 使用相角差，避免极弱或极强系数四项相乘的溢出/下溢。
            angle = np.angle(a[left])-np.angle(a[right])-np.angle(raw[left])+np.angle(raw[right])
            err = abs(np.rad2deg(np.angle(np.exp(1j*angle))))
            err[~masks[k]] = np.nan
            errors.append(_nullable(err))
        output[name] = dict(matrix_nmse=_nullable(nmse), phase_error_deg=errors)
    return dict(schema='channel-comparison/1', offsets_hz=f.tolist(),
                port_pairs=[list(p) for p in pairs], amplitude_floor=floor, methods=output,
                phase_valid_mask=masks.tolist(), matrix_valid_mask=valid.tolist(),
                coverage=dict(phase_valid=int(masks.sum()), phase_total=int(masks.size),
                              phase_fraction=float(masks.mean()), matrix_valid=int(valid.sum()),
                              matrix_total=int(valid.size), matrix_fraction=float(valid.mean())),
                time_basis=time_basis, delay_zero_basis=delay_zero_basis,
                alignment_status='unverified', simulated=bool(simulated), formal_verdict=None)


def recover_asc_frames(data, *, rate_hz, source_frame_count, threshold_ns=1e-7):
    """恢复分段线性时延的数值拐点，并左右外推幅度及最短弧相位。

    仅适用参考脚本的 ASC 布局 [sample,port,tap,(delay_ns,re,im)]。
    要求恰好恢复 N-1 个源帧，末帧保持未配对；不推断物理源时间。
    """
    rate = _positive(rate_hz, 'ASC 更新率')
    threshold = _positive(threshold_ns, '拐点阈值')
    x = _real(data, 'ASC')
    if (x.ndim != 4 or x.shape[-1] != 3 or min(x.shape[:3]) < 1 or x.shape[0] < 4
            or np.any(x[..., 0] < 0) or type(source_frame_count) is not int or source_frame_count < 2):
        raise ValueError('ASC 布局或源帧数无效')
    q = np.median(abs(np.diff(x[..., 0], n=2, axis=0)), axis=(1, 2))
    ix = np.flatnonzero(q > threshold) + 1
    if (len(ix) != 2*(source_frame_count-2) or not np.all(ix[1::2] == ix[::2]+1)
            or np.any(ix[::2] < 1) or np.any(ix[::2]+2 >= len(x))):
        raise ValueError('不能恢复唯一完整的 N-1 帧数值参考，不得猜测或补齐')
    anchors = [0.] + [float(j+q[j]/(q[j-1]+q[j])) for j in ix[::2]]

    roundoff = dict(clamped_taps=0, maximum_absolute_correction=0.0,
                    relative_tolerance=1e-10)

    def extract(k, alpha):
        a, b = x[k], x[k+1]
        h0, h1 = a[..., 1]+1j*a[..., 2], b[..., 1]+1j*b[..., 2]
        delay = (a[..., 0]+alpha*(b[..., 0]-a[..., 0]))*1e-9
        magnitude = abs(h0)+alpha*(abs(h1)-abs(h0))
        # 拐点位置由十进制 ASC 的二阶差分恢复，误差也传入外推系数。
        rounding = 1e-10*(abs(h0)+abs(alpha)*(abs(h0)+abs(h1)))
        if np.any(delay < 0) or np.any(magnitude < -rounding):
            raise ValueError('外推得到负时延或负幅度，数值参考未核验')
        negative = magnitude < 0
        roundoff['clamped_taps'] += int(negative.sum())
        roundoff['maximum_absolute_correction'] = max(
            roundoff['maximum_absolute_correction'], float(np.max(-magnitude[negative], initial=0)))
        # 只修正零点附近的数值负幅度，修正计数与界限随结果报告。
        magnitude = np.maximum(magnitude, 0)
        h = magnitude*np.exp(1j*(np.angle(h0)+alpha*np.angle(h1*np.conj(h0))))
        return delay, h

    left, right = [], []
    for i, v in enumerate(anchors):
        k = int(np.floor(v))
        left.append(extract(k-1, v-k+1) if i else extract(0, 0))
        right.append(extract(k+1, v-k-1) if i else extract(0, 0))
    return dict(sample_positions=anchors, times_s=[v/rate for v in anchors],
                delays_s=np.stack([a[0] for a in left]), coefficients=np.stack([a[1] for a in left]),
                right_delays_s=np.stack([a[0] for a in right]), right_coefficients=np.stack([a[1] for a in right]),
                unmatched_source_frames=[source_frame_count-1], time_status='unverified',
                amplitude_roundoff=roundoff,
                time_basis='ASC 时延二阶差分拐点；左右幅度/最短弧相位外推；非物理源时间证据')
