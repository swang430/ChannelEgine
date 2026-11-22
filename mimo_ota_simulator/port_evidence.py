"""按输入成员顺序只读复算单帧数据。调用方负责核对来源和筛选规则。"""

from itertools import combinations, product

import numpy as np


def _complex(z):
    return {"real": float(z.real), "imag": float(z.imag)}


def _directions(a, z):
    a, z = np.deg2rad(a), np.deg2rad(z)
    return np.column_stack((np.sin(z) * np.cos(a), np.sin(z) * np.sin(a), np.cos(z)))


def _prepare(link):
    data = {}
    for key in ("h", "delay", "aoa", "zoa", "aod", "zod", "labels", "row_ids"):
        arr = np.asarray(link[key])
        if arr.ndim != 1 or not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} 需要有限的一维数组")
        if key in ("labels", "row_ids") and not np.all(arr == arr.astype(int)):
            raise ValueError(f"{key} 需要整数")
        data[key] = arr.astype(
            complex if key == "h" else int if key in ("labels", "row_ids") else float
        )
    if len({len(a) for a in data.values()}) != 1 or len(set(data["row_ids"])) != len(
        data["h"]
    ):
        raise ValueError("成员数组长度不一致或源行号重复")
    data["arrival"] = _directions(data["aoa"], data["zoa"])
    data["departure"] = _directions(data["aod"], data["zod"])
    data["expected"] = {int(k): v for k, v in link.get("expected", {}).items()}
    return data


def _compare(data, ref):
    """分簇批量查最近邻，完整候选仅保留在每个已分配簇的代表行上。"""
    from scipy.spatial import cKDTree

    labels = data["labels"]
    ref_labels = np.unique(ref["labels"]) if ref is not None else []
    n = len(labels)
    axes = ("delay", "arrival", "departure")
    # 每条射线仅保存三个单维最佳候选和已分配簇，内存与参考簇数无关。
    best = np.full((n, 3), np.inf)
    second = best.copy()
    winner = np.full((n, 3), -1, dtype=int)
    nearest = np.zeros((n, 3, 3), dtype=int)
    runner = winner.copy()
    runner_members = nearest.copy()
    assigned = np.zeros((n, 3), dtype=int)
    assigned_delta = np.zeros((n, 3))
    for cluster in ref_labels:
        subset = np.flatnonzero(ref["labels"] == cluster)
        scores = np.empty((n, 3))
        indices = np.empty((n, 3), dtype=int)
        for axis, key in enumerate(axes):
            width = 1 if axis == 0 else 3
            points = ref[key].reshape(-1, width)[subset]
            distance, pos = cKDTree(points).query(data[key].reshape(-1, width))
            indices[:, axis] = subset[pos]
            scores[:, axis] = (
                distance
                if axis == 0
                else np.rad2deg(2 * np.arcsin(np.minimum(1, distance / 2)))
            )
        same = labels == cluster
        assigned[same] = indices[same]
        assigned_delta[same] = scores[same]
        for axis in range(3):
            improved = scores[:, axis] < best[:, axis]
            next_best = ~improved & (scores[:, axis] < second[:, axis])
            runner[improved, axis] = winner[improved, axis]
            runner_members[improved, axis] = nearest[improved, axis]
            runner[next_best, axis] = cluster
            runner_members[next_best, axis] = indices[next_best]
            second[:, axis] = np.minimum(
                second[:, axis], np.maximum(best[:, axis], scores[:, axis])
            )
            best[improved, axis] = scores[improved, axis]
            winner[improved, axis] = cluster
            nearest[improved, axis] = indices[improved]
    tied = np.isclose(best, second, rtol=1e-12, atol=[0, 1e-12, 1e-12])
    distinct = (
        (winner == winner[:, :1]).all(axis=1)
        & ~tied.any(axis=1)
        & np.isin(labels, ref_labels)
    )
    samples, coverage = [], {}
    power = abs(data["h"]) ** 2
    for cluster in np.unique(labels):
        ids = np.flatnonzero(labels == cluster)
        ambiguous = ids[~distinct[ids]]
        i = int(ambiguous[0] if len(ambiguous) else ids[0])
        good = ids[distinct[ids]]
        coverage[int(cluster)] = dict(
            distinguishable_count=len(good),
            assigned_nearest_count=int(np.count_nonzero(winner[good, 0] == cluster)),
            distinguishable_power_fraction=float(power[good].sum() / power[ids].sum())
            if power[ids].sum()
            else None,
        )
        candidates = []
        retained = {}
        if cluster in ref_labels:
            retained[cluster] = assigned[i]
        if len(ref_labels):
            for axis, other in enumerate(winner[i]):
                retained[other] = nearest[i, axis]
                if other == cluster and runner[i, axis] >= 0:
                    retained[runner[i, axis]] = runner_members[i, axis]
        for other, positions in sorted(retained.items()):
            members = []
            for k in positions:
                chord = [
                    np.linalg.norm(data[axis][i] - ref[axis][k]) for axis in axes[1:]
                ]
                arrival, departure = np.rad2deg(
                    2 * np.arcsin(np.minimum(1, np.array(chord) / 2))
                )
                members.append(
                    dict(
                        row_id=int(ref["row_ids"][k]),
                        delay_s=float(abs(data["delay"][i] - ref["delay"][k])),
                        arrival_deg=float(arrival),
                        departure_deg=float(departure),
                    )
                )
            candidates.append(dict(cluster=int(other), members=members))
        samples.append(
            dict(
                row_id=int(data["row_ids"][i]),
                assigned_cluster=int(cluster),
                candidates=candidates,
                nearest_cluster=int(winner[i, 0]) if distinct[i] else None,
                reason=None
                if distinct[i]
                else (
                    "参考端口缺少该簇成员"
                    if cluster not in ref_labels
                    else "候选并列或各维最近候选不同"
                ),
                status="distinguishable" if distinct[i] else "unverified",
            )
        )
    raw = dict(
        row_id=data["row_ids"].tolist(),
        assigned_cluster=labels.tolist(),
        distinguishable=distinct.tolist(),
    )
    has_assigned = np.isin(labels, ref_labels)
    for axis, field in enumerate(("delay_s", "arrival_deg", "departure_deg")):
        columns = dict.fromkeys(
            (
                "nearest_cluster",
                "nearest_row",
                "nearest_delta",
                "assigned_row",
                "assigned_delta",
            )
        )
        if len(ref_labels):
            columns.update(
                nearest_cluster=winner[:, axis].tolist(),
                nearest_row=ref["row_ids"][nearest[:, axis, axis]].tolist(),
                nearest_delta=best[:, axis].tolist(),
                assigned_row=np.where(
                    has_assigned, ref["row_ids"][assigned[:, axis]], None
                ).tolist(),
                assigned_delta=np.where(
                    has_assigned, assigned_delta[:, axis], None
                ).tolist(),
            )
        else:
            columns = {key: [None] * n for key in columns}
        raw[field] = columns
    return samples, coverage, raw


def analyze_frame(frame, links, offsets_hz, amplitude_floor=None):
    """端口从零编号，参考端口为 (0,0)。空数组表示已知空链路，缺字段表示未核验。"""
    prepared, rows, warnings = {}, [], []
    ports = set(product(range(4), range(2)))
    if set(links) - ports:
        raise ValueError("端口范围为发射端 0 至 3、接收端 0 至 1")
    for tx, rx in sorted(ports):
        key = tx, rx
        row = dict(
            tx=int(tx), rx=int(rx), clusters=[], rays=[], raw=None, coverage=None
        )
        rows.append(row)
        try:
            data = _prepare(links[key])
        except (KeyError, TypeError, ValueError) as err:
            row.update(status="unverified", reason=f"缺少有效的成员数据：{err}")
            continue
        prepared[key] = data
        row["rays"], coverage, row["raw"] = _compare(data, prepared.get((0, 0)))
        n = len(data["h"])
        row.update(
            status="analyzed",
            coverage=sum(c["distinguishable_count"] for c in coverage.values()) / n
            if n
            else None,
        )
        total = np.sum(abs(data["h"]) ** 2)
        for label in sorted(set(data["labels"]) | set(data["expected"])):
            mask = data["labels"] == label
            h = data["h"][mask]
            actual = dict(count=len(h), **_complex(h.sum()))
            expected = data["expected"].get(label)
            cluster = dict(
                cluster=int(label),
                **actual,
                **coverage.get(int(label), {}),
                power_fraction=float(np.sum(abs(h) ** 2) / total) if total else None,
                expected_status="unverified",
            )
            row["clusters"].append(cluster)
            if expected is not None:
                tol = float(16 * np.finfo(float).eps * max(1, len(h)) * np.sum(abs(h)))
                checks = {k: "unverified" for k in actual}
                for k in actual.keys() & expected.keys():
                    checks[k] = (
                        "matched"
                        if abs(actual[k] - expected[k]) <= (0 if k == "count" else tol)
                        else "warning"
                    )
                cluster["checks"] = checks
                failed = [k for k in actual if checks[k] == "warning"]
                if any(v != "unverified" for v in checks.values()):
                    cluster["expected_status"] = (
                        "warning"
                        if failed
                        else "partial"
                        if "unverified" in checks.values()
                        else "matched"
                    )
                if failed:
                    warnings.append(
                        dict(
                            frame=int(frame),
                            tx=int(tx),
                            rx=int(rx),
                            cluster=int(label),
                            fields=failed,
                            expected={
                                k: float(expected[k]) for k in actual if k in expected
                            },
                            actual=actual,
                            coefficient_abs_tolerance=tol,
                            sample_row_id=int(data["row_ids"][mask][0])
                            if len(h)
                            else None,
                        )
                    )
    return dict(
        frame=int(frame),
        p2_106=dict(
            links=rows,
            warnings=warnings,
            method="分别按时延、到达夹角、出发夹角列最近成员；每个已分配簇优先展示不可区分的源行，展示已分配簇和各维最近候选；各维均唯一且指向同簇才标为可区分。",
            raw_guide=(
                "在每条链路的 raw 中取各列相同下标即可重建一条射线。row_id 为原始源行，"
                "assigned_cluster 为已分配簇，distinguishable 表示三个维度是否一致且可区分。"
                "delay_s、arrival_deg、departure_deg 分别给出绝对时延差（秒）、到达和出发夹角（度）。"
                "每维 nearest_cluster、nearest_row、nearest_delta 为全参考端口最近簇、源行及差值，"
                "assigned_row、assigned_delta 为已分配簇内最近参考源行及差值。"
                "参考源行均属于本帧端口 (0,0)，缺少对应参考时为 null，并列时保留一个最近候选。"
            ),
            tie_tolerance=dict(
                relative=1e-12,
                absolute_delay_s=0,
                absolute_angle_deg=1e-12,
                source="浮点计算近似相等，仅用于区分候选，不作为工程容差。",
            ),
            physical_identity="unverified",
        ),
        p2_107=_responses(frame, prepared, offsets_hz, amplitude_floor),
    )


def _response(grid, h, delay):
    # 每个指数数组至多 65536 项，避免一次展开所有频点与所有射线。
    result = np.zeros(len(grid), dtype=complex)
    delay = np.asarray(delay)
    h = np.asarray(h, dtype=complex)
    for f in range(0, len(grid), 256):
        frequencies = grid[f : f + 256]
        block = 65536 // len(frequencies)
        for j in range(0, len(h), block):
            phase = -2j * np.pi * frequencies[:, None] * delay[j : j + block]
            result[f : f + 256] += np.exp(phase) @ h[j : j + block]
    return result


def _responses(frame, links, offsets, floor):
    grid = np.asarray(offsets, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
        raise ValueError("频偏必须是非空的有限一维数组")
    if floor is not None and (not np.isfinite(floor) or floor < 0):
        raise ValueError("幅度下限必须是有限非负数")
    keys = list(product(range(4), range(2)))
    complete = all(key in links for key in keys)
    shared, groups = {}, {}
    for key, data in links.items():
        groups[key] = {}
        for label in np.unique(data["labels"]):
            mask = data["labels"] == label
            h = data["h"][mask]
            power = abs(h) ** 2
            moment = power @ data["delay"][mask]
            weight = power.sum()
            groups[key][label] = (h.sum(), moment / weight if weight else 0)
            m, w = shared.get(label, (0, 0))
            shared[label] = (m + moment, w + weight)
    rows, responses = [], {}
    for key in keys:
        row = dict(
            tx=key[0],
            rx=key[1],
            raw=None,
            A=None,
            B=None,
            reason=None if complete else "八路成员不全，共享时延未核验",
        )
        rows.append(row)
        if key not in links:
            continue
        data, clusters = links[key], groups[key]
        raw = _response(grid, data["h"], data["delay"])
        resp = {"raw": raw}
        h = [v[0] for v in clusters.values()]
        resp["A"] = _response(grid, h, [v[1] for v in clusters.values()])
        if complete:
            resp["B"] = _response(
                grid,
                h,
                [shared[k][0] / shared[k][1] if shared[k][1] else 0 for k in clusters],
            )
        responses[key] = resp
        for scheme, values in resp.items():
            row[scheme] = [_complex(v) for v in values]
            if scheme != "raw":
                row[scheme + "_error"] = [_complex(v) for v in values - raw]
    matrix = []
    for i, f in enumerate(grid):
        point = dict(offset_hz=float(f))
        for scheme in ("A", "B"):
            error = energy = None
            if complete:
                error = sum(
                    abs(responses[k][scheme][i] - responses[k]["raw"][i]) ** 2
                    for k in keys
                )
                energy = sum(abs(responses[k]["raw"][i]) ** 2 for k in keys)
            point[scheme] = dict(
                squared_error=error, nmse=error / energy if energy else None
            )
        matrix.append(point)
    pairs = []
    for left, right in combinations(keys, 2):
        if left[0] != right[0] and left[1] != right[1]:
            continue
        pair = dict(left=list(left), right=list(right))
        for scheme in ("A", "B"):
            errors = [None] * len(grid)
            if floor is not None and all(
                scheme in responses.get(k, {}) for k in (left, right)
            ):
                a, b = responses[left], responses[right]
                valid = (
                    np.minimum.reduce(
                        [abs(r[s]) for r in (a, b) for s in ("raw", scheme)]
                    )
                    > floor
                )
                phase = (
                    np.angle(a[scheme])
                    - np.angle(b[scheme])
                    - np.angle(a["raw"])
                    + np.angle(b["raw"])
                )
                for i in np.flatnonzero(valid):
                    errors[i] = float((np.rad2deg(phase[i]) + 180) % 360 - 180)
            valid_ids = [i for i, v in enumerate(errors) if v is not None]
            magnitudes = [abs(errors[i]) for i in valid_ids]
            worst = valid_ids[int(np.argmax(magnitudes))] if valid_ids else None
            pair[scheme] = dict(
                errors_deg=errors,
                valid_count=len(valid_ids),
                valid_fraction=len(valid_ids) / len(grid),
                p95_abs_deg=float(np.percentile(magnitudes, 95)) if valid_ids else None,
                worst=None
                if worst is None
                else dict(
                    frame=int(frame),
                    offset_hz=float(grid[worst]),
                    error_deg=errors[worst],
                ),
            )
        pairs.append(pair)
    steps = np.diff(grid)
    return dict(
        offsets_hz=grid.tolist(),
        grid_source="调用方提供的分析频偏",
        bandwidth_hz=float(np.ptp(grid)),
        point_count=len(grid),
        step_hz=float(steps[0])
        if len(steps) and np.allclose(steps, steps[0])
        else None,
        phase_note="四个响应幅度均须高于给定下限；缺少下限时相位未核验。",
        amplitude_floor=None if floor is None else float(floor),
        links=rows,
        matrix=matrix,
        phase_pairs=pairs,
        shared_delay_note="共享时延按八路成员功率加权，功率大的端口权重更高。",
    )
