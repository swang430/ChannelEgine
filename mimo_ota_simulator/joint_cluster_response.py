"""按已知簇成员关系归并一条阵元链路，保留该链路的实际幅相与时延。"""

import numpy as np


def _real_column(value):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if np.any(array.imag != 0):
            raise ValueError("时延、角度及多普勒必须是实数")
        array = array.real
    return np.asarray(array, dtype=float)


def reduce_link(
    membership,
    coefficients,
    delays_s,
    azimuth_deg,
    doppler_hz,
    *,
    departure_azimuth_deg=None,
    arrival_zenith_deg=None,
    departure_zenith_deg=None,
):
    labels = np.asarray(membership)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("每条路径需要已确认的非负整数簇编号")
    h = np.asarray(coefficients, dtype=complex)
    delay, angle, doppler = (
        _real_column(column) for column in (delays_s, azimuth_deg, doppler_hz)
    )
    extra_angles = {
        name: _real_column(value)
        for name, value in (
            ("departure_azimuth", departure_azimuth_deg),
            ("arrival_zenith", arrival_zenith_deg),
            ("departure_zenith", departure_zenith_deg),
        )
        if value is not None
    }
    for column in (h, delay, angle, doppler, *extra_angles.values()):
        if column.shape != labels.shape or not np.isfinite(column).all():
            raise ValueError("各列必须每条射线对应一个有限数值")
    if np.any(delay < 0):
        raise ValueError("路径时延不能为负")
    result = {}
    for label in np.unique(labels):
        mask = labels == label
        with np.errstate(over="ignore"):
            power = np.abs(h[mask]) ** 2
            total = power.sum()
        if not np.isfinite(total):
            raise ValueError("路径功率超过可表示范围")
        if total == 0:
            raise ValueError("零功率簇没有可定义的功率加权统计")
        weights = power / total
        bearing = np.sum(weights * np.exp(1j * np.deg2rad(angle[mask])))
        mean_angle = np.rad2deg(np.angle(bearing))
        deviation = (angle[mask] - mean_angle + 180) % 360 - 180
        mean_doppler = np.sum(weights * doppler[mask])
        result[int(label)] = dict(
            path_power=float(total),
            azimuth_deg=float(mean_angle) if abs(bearing) > 1e-12 else None,
            doppler_mean_hz=float(mean_doppler),
            coefficient=complex(h[mask].sum()),
            delay_s=float(np.sum(weights * delay[mask])),
            azimuth_spread_deg=(
                float(np.sqrt(np.sum(weights * deviation**2)))
                if abs(bearing) > 1e-12
                else None
            ),
            doppler_spread_hz=float(
                np.sqrt(np.sum(weights * (doppler[mask] - mean_doppler) ** 2))
            ),
        )
        for name, values in extra_angles.items():
            samples = values[mask]
            if name.endswith("zenith"):
                if np.any((samples < 0) | (samples > 180)):
                    raise ValueError("天顶角必须在 0 至 180 度之间")
                center = float(np.sum(weights * samples))
                residual = samples - center
            else:
                vector = np.sum(weights * np.exp(1j * np.deg2rad(samples)))
                center = (
                    float(np.rad2deg(np.angle(vector))) if abs(vector) > 1e-12 else None
                )
                residual = (
                    (samples - center + 180) % 360 - 180 if center is not None else None
                )
            result[int(label)][name + "_deg"] = center
            result[int(label)][name + "_spread_deg"] = (
                float(np.sqrt(np.sum(weights * residual**2)))
                if residual is not None
                else None
            )
        if any(
            value is not None and not np.isfinite(value)
            for value in result[int(label)].values()
        ):
            raise ValueError("簇统计量超过可表示范围")
    return result


def associate_frames(previous, current, *, delay_limit_s, angle_limit_deg):
    """用互为唯一最近邻的时延和到达角提出关联；未配对能量单列。"""
    if not all(np.isfinite(v) and v > 0 for v in (delay_limit_s, angle_limit_deg)):
        raise ValueError("时延和角度容差必须为有限正数")
    for group in (previous, current):
        for cluster in group.values():
            for field in ("path_power", "delay_s"):
                if not np.isfinite(cluster[field]) or cluster[field] < 0:
                    raise ValueError("路径功率和时延应为有限非负数")
            angle = cluster["azimuth_deg"]
            if angle is not None and not np.isfinite(angle):
                raise ValueError("方位角应为有限值或未知")
    candidates = {}
    for a, left in previous.items():
        for b, right in current.items():
            if left["azimuth_deg"] is None or right["azimuth_deg"] is None:
                continue
            delay = abs(left["delay_s"] - right["delay_s"]) / delay_limit_s
            angle = (
                abs((left["azimuth_deg"] - right["azimuth_deg"] + 180) % 360 - 180)
                / angle_limit_deg
            )
            if delay <= 1 and angle <= 1:
                candidates[a, b] = delay**2 + angle**2

    def closest(axis):
        groups = {}
        for pair, cost in candidates.items():
            groups.setdefault(pair[axis], []).append((cost, pair))
        selected = set()
        for choices in groups.values():
            choices.sort(key=lambda item: item[0])
            if len(choices) == 1 or not np.isclose(choices[0][0], choices[1][0]):
                selected.add(choices[0][1])
        return selected

    matched = {b: a for a, b in closest(0) & closest(1)}
    scale = max((c["path_power"] for c in current.values()), default=0.0)
    relative = {b: c["path_power"] / scale for b, c in current.items()} if scale else {}
    total = sum(relative.values())
    unmatched = sum(power for b, power in relative.items() if b not in matched)
    return dict(
        previous_by_current=matched,
        unmatched_power_fraction=unmatched / total if total else None,
    )
