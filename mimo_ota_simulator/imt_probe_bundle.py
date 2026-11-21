"""从 RT-IMT-2030 下行单频 MPDB 生成源帧双极化探头 ASC。"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from .incident_field_ota import fit_probe_taps

C = 299792458.0


def directions(azimuth, zenith):
    a, z = np.deg2rad(azimuth), np.deg2rad(zenith)
    return np.array([np.sin(z) * np.cos(a), np.sin(z) * np.sin(a), np.cos(z)]).T


def polar_basis(a, z):
    a, z = np.deg2rad(a), np.deg2rad(z)
    return np.stack(
        [
            np.array([np.cos(z) * np.cos(a), np.cos(z) * np.sin(a), -np.sin(z)]).T,
            np.array([-np.sin(a), np.cos(a), np.zeros_like(a)]).T,
        ],
        axis=1,
    )


def field(points, az, zen, components, frequency):
    phase = np.exp(2j * np.pi * frequency / C * (points @ directions(az, zen).T))
    xyz = np.einsum("rp,rpc->rc", components, polar_basis(az, zen))
    return phase @ xyz


def read_columns(path, *, dynamic=False):
    with zipfile.ZipFile(path) as z:
        if z.getinfo("R0B0S0D.mpdb").file_size > 256 * 1024**2:
            raise ValueError("下行样本超过大小上限")
        raw = z.read("R0B0S0D.mpdb")
    with zipfile.ZipFile(io.BytesIO(raw)) as db:
        if sum(i.file_size for i in db.infolist()) > 512 * 1024**2:
            raise ValueError("内层数据超过大小上限")
        desc = json.loads(db.read("mpdb.json"))
        if desc["meta"]["lauraycsPlatform"]["direction"] != "downlink":
            raise ValueError("需要下行传播数据")
        meta = desc["meta"]
        if (
            meta["lauraycsPlatform"].get("angleCoordinateSystem") != "GCS"
            or meta.get("numFrequency") != 1
            or meta.get("mode") != "CIR"
            or meta["polarization"]["txModeOrder"][0] != "+45"
        ):
            raise ValueError("本生成器要求全局角度、单频 CIR 和发射第一极化 +45°")
        result = {}
        for table, names in {
            "CHANNEL": "LINK_ID POLAR JONES_T FADE_L PHASE DELAY AOA ZOA"
            + (" DOPPLER" if dynamic else ""),
            "LINK": "LINK_ID FRAME_ID TX_ANT RX_ANT FREQUENCY"
            + (" RX_ANT_POSITION" if dynamic else ""),
        }.items():
            for name in names.split():
                column = desc["records"][table][name]
                chunks = []
                cursor = 0
                for shard in sorted(
                    column["shards"], key=lambda item: item["rowStart"]
                ):
                    begin, end = shard["rowStart"], shard["rowEnd"]
                    if begin != cursor or not begin < end <= column["rowCount"]:
                        raise ValueError(f"{table}.{name} 分片行范围重叠、缺失或越界")
                    cursor = end
                    data = db.read(shard["entry"])
                    n = int.from_bytes(data[:8], "little")
                    spec = json.loads(data[8 : 8 + n])["data"]
                    dtype = {
                        "I32": "<i4",
                        "I64": "<i8",
                        "F32": "<f4",
                        "F64": "<f8",
                        "C64": "<c8",
                        "C128": "<c16",
                    }[spec["dtype"]]
                    if spec["shape"][0] != end - begin:
                        raise ValueError(f"{table}.{name} 分片实际行数不符")
                    chunks.append(
                        np.frombuffer(data[8 + n :], dtype=dtype).reshape(spec["shape"])
                    )
                if cursor != column["rowCount"]:
                    raise ValueError(f"{table}.{name} 分片未覆盖整列")
                arr = np.concatenate(chunks)
                if list(arr.shape) != column["shape"] or not np.isfinite(arr).all():
                    raise ValueError(f"{table}.{name} 形状或数值无效")
                result[table, name] = arr
    return result, hashlib.sha256(raw).hexdigest()


def run(archive, geometry, output, rate):
    if output.exists() or not np.isfinite(rate) or rate <= 0:
        raise ValueError("需要新的输出目录和正的源帧更新率")
    g = json.loads(geometry.read_text())
    ports = np.asarray(g["probe_azimuth_zenith_deg"], float)
    points, check = [
        np.asarray(g[key], float) for key in ("fit_points_m", "check_points_m")
    ]
    for xyz in (points, check):
        if (
            xyz.ndim != 2
            or xyz.shape[1] != 3
            or not len(xyz)
            or not np.isfinite(xyz).all()
        ):
            raise ValueError("验证和拟合坐标应为有限的三维点")
    if set(map(tuple, points)) & set(map(tuple, check)):
        raise ValueError("验证点与拟合点应独立")
    if (
        ports.ndim != 2
        or ports.shape[1] != 2
        or not len(ports)
        or not np.isfinite(ports).all()
    ):
        raise ValueError("需要有限的探头方位角和天顶角")
    if np.any((ports[:, 1] < 0) | (ports[:, 1] > 180)):
        raise ValueError("探头天顶角必须在 0 至 180 度之间")
    data, digest = read_columns(archive)

    def link(k):
        return data["LINK", k].reshape(-1)

    def rays(k):
        return data["CHANNEL", k]

    if (
        set(link("TX_ANT")) != set(range(4))
        or set(link("RX_ANT")) != {0, 1}
        or len(np.unique(link("LINK_ID"))) != len(link("LINK_ID"))
        or not np.isin(rays("LINK_ID"), link("LINK_ID")).all()
    ):
        raise ValueError("源文件必须为四发两收且路径与链路逐项对应")
    if np.any(rays("DELAY") < 0) or np.any((rays("ZOA") < 0) | (rays("ZOA") > 180)):
        raise ValueError("源路径的时延或天顶角越界")
    az, zen = ports.T
    nf = len(ports) * 2
    if 4 * nf > 128:
        raise ValueError("超过 128 个逻辑通道")
    (freq,) = np.unique(link("FREQUENCY"))
    freq = float(freq)
    if freq <= 0:
        raise ValueError("载频必须为正")

    def basis(x, f):
        phase = np.exp(2j * np.pi * f / C * (x @ directions(az, zen).T))
        return np.einsum("xp,pvc->xcpv", phase, polar_basis(az, zen)).reshape(-1, nf)

    A = basis(points, freq)
    # 直角坐标三个分量同时拟合；同一线性逆用于所有发射端口、源帧和时延抽头。
    probe_response = A.reshape(len(points), 3, nf)
    frames = np.unique(link("FRAME_ID"))
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("源帧不连续")
    taps = 24
    weights = np.zeros((len(frames), 4, nf, taps), complex)
    delays = np.empty((len(frames), 4, taps))
    links = rays("LINK_ID").ravel()
    offsets = (-50e6, 0.0, 50e6)
    targets = {}
    for fi in frames:
        for tx in range(4):
            (lid,) = link("LINK_ID")[
                (link("FRAME_ID") == fi)
                & (link("TX_ANT") == tx)
                & (link("RX_ANT") == 0)
            ]
            mask = links == lid
            e = (rays("POLAR")[mask] @ rays("JONES_T")[mask]).reshape(-1, 2)
            e *= (rays("FADE_L")[mask] * rays("PHASE")[mask]).reshape(-1, 1)
            tau = rays("DELAY")[mask].ravel().astype(float)
            ao = rays("AOA")[mask].ravel()
            zo = rays("ZOA")[mask].ravel()
            power = np.sum(abs(e.astype(complex)) ** 2, axis=1)
            keep = power > 0
            if not keep.any():
                raise ValueError("源帧没有有效入射场")
            tau, ao, zo, e, power = (x[keep] for x in (tau, ao, zo, e, power))
            # 离线压缩：按时延排序后按累积入射功率分组；这些分组不声明跨帧物理身份。
            order = np.argsort(tau)
            tau, ao, zo, e, power = (x[order] for x in (tau, ao, zo, e, power))
            group = np.minimum(
                ((np.cumsum(power) - power / 2) / power.sum() * taps).astype(int),
                taps - 1,
            )
            clusters = []
            for k in np.unique(group):
                m = group == k
                clusters.append(
                    (
                        np.average(tau[m], weights=power[m]),
                        field(points, ao[m], zo[m], e[m], freq),
                    )
                )
            solved = fit_probe_taps(
                probe_response, np.stack([f for _, f in clusters])[None]
            )[0]
            clusters = [(item[0], w) for item, w in zip(clusters, solved)]
            while len(clusters) < taps:
                clusters.append((clusters[-1][0], np.zeros(nf)))
            delays[fi, tx] = [d for d, w in clusters]
            weights[fi, tx] = np.stack([w for d, w in clusters], axis=1)
            for off in offsets:
                targets[fi, tx, off] = field(
                    check,
                    ao,
                    zo,
                    e * np.exp(-2j * np.pi * off * tau[:, None]),
                    freq + off,
                ).ravel()
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        files = []
        metrics = []
        for tx in range(4):
            recovered = np.empty_like(weights[:, tx])
            for feed in range(nf):
                name = f"TX{tx}_PROBE{feed // 2:02d}_{('theta', 'phi')[feed % 2]}.asc"
                file = stage / name
                v = weights[:, tx, feed]
                values = np.stack([delays[:, tx] * 1e9, v.real, v.imag], axis=-1)
                header = (
                    f"***** Header *****\n{len(frames)} CIRs\n{taps} Taps/CIR\n{freq} Carrier_Frequency\nRoute_Open\nCIRUpdateRate_Locked\nCarrierFrequency_Locked\n5 Delay_Resolution\n2 Sample_Density\n{rate} CIR_Update_Rate\n***** Tap data *****\n"
                    + "\t".join(["Delay\tRe\tIm"] * taps)
                )
                np.savetxt(
                    file,
                    values.reshape(len(frames), -1),
                    header=header,
                    comments="",
                    delimiter="\t",
                    fmt="%.17g",
                )
                arr = np.loadtxt(file, skiprows=12, ndmin=2).reshape(values.shape)
                np.testing.assert_array_equal(arr[:, :, 0], values[:, :, 0])
                recovered[:, feed] = arr[:, :, 1] + 1j * arr[:, :, 2]
                files.append(
                    dict(
                        tx=tx,
                        probe=feed // 2,
                        polarization=("theta", "phi")[feed % 2],
                        file=name,
                        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    )
                )
            for fi in frames:
                for off in offsets:
                    ref = targets[fi, tx, off]
                    est = basis(check, freq + off) @ (
                        recovered[fi] @ np.exp(-2j * np.pi * off * delays[fi, tx])
                    )
                    power = np.vdot(ref, ref).real
                    metrics.append(
                        dict(
                            frame=int(fi),
                            tx=tx,
                            offset_hz=off,
                            field_nmse=float(np.vdot(est - ref, est - ref).real / power)
                            if power
                            else None,
                            power_ratio=float(np.vdot(est, est).real / power)
                            if power
                            else None,
                        )
                    )
        manifest = dict(
            mode="ota",
            direction="downlink",
            evidence="offline_ideal_plane_wave",
            smu_built=False,
            hardware_verified=False,
            source_sha256=digest,
            geometry=g,
            header=dict(
                cir_count=len(frames), taps=taps, carrier_hz=freq, rate_hz=rate
            ),
            source_frames_only=True,
            tx_mode="+45",
            reference_rx=0,
            rate_source="operator_supplied",
            delay_compression="power_quantiles_sorted_by_delay",
            links=files,
            metrics=metrics,
        )
        (stage / "manifest.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2)
        )
        stage.rename(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "geometry", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("--source-rate-hz", required=True, type=float)
    a = p.parse_args()
    run(a.archive, a.geometry, a.output, a.source_rate_hz)
