"""读取参考阵元的逐帧簇表，导出推定的轨迹、消失项与未匹配功率。"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from .joint_cluster_response import associate_frames


def evaluate(source, delay_limit_ns, angle_limit_deg):
    payload = source.read_bytes()
    frames = {}
    for row in csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))):
        frame, rank = int(row["FRAME_ID"]), int(row["CLUSTER_RANK"])
        clusters = frames.setdefault(frame, {})
        if rank in clusters:
            raise ValueError(f"源帧 {frame} 重复簇编号 {rank}")
        clusters[rank] = dict(
            delay_s=float(row["DELAY_NS"]) * 1e-9,
            azimuth_deg=float(row["AOA"]),
            path_power=10 ** (float(row["POWER_DB"]) / 10),
        )
    if not frames:
        raise ValueError("簇表为空")
    previous, previous_tracks = {}, {}
    next_track = 0
    report = []
    for frame, current in sorted(frames.items()):
        if report and frame != report[-1]["frame"] + 1:
            raise ValueError(f"源帧不连续，拒绝跨缺帧匹配：{frame}")
        result = associate_frames(
            previous,
            current,
            delay_limit_s=delay_limit_ns * 1e-9,
            angle_limit_deg=angle_limit_deg,
        )
        matches = result["previous_by_current"]
        tracks = {}
        for rank in sorted(current):
            if rank in matches:
                tracks[rank] = previous_tracks[matches[rank]]
            else:
                tracks[rank] = next_track
                next_track += 1
        report.append(
            dict(
                frame=frame,
                track_by_cluster=tracks,
                matched=len(matches),
                ended_tracks=sorted(
                    set(previous_tracks.values()) - set(tracks.values())
                ),
                unmatched_power_fraction=result["unmatched_power_fraction"]
                if previous
                else None,
            )
        )
        previous, previous_tracks = current, tracks
    return dict(
        input_sha256=hashlib.sha256(payload).hexdigest(),
        association="estimated",
        delay_limit_ns=delay_limit_ns,
        angle_limit_deg=angle_limit_deg,
        frames=report,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--delay-limit-ns", type=float, required=True)
    parser.add_argument("--angle-limit-deg", type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.csv, args.delay_limit_ns, args.angle_limit_deg),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
