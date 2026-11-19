"""用原导出表和独立时刻表复核命令行输出的每帧、每个频偏。"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def verify(output, cluster_csv, timing_csv, direction, offsets):
    data = json.loads(output.read_text())
    if (data['direction'] != direction or data['offsetsHz'] != offsets
            or data['sampling'] != 'source_samples_only'):
        raise ValueError('方向、采样策略或频偏有误')
    with timing_csv.open(encoding='utf-8-sig') as stream:
        times = [row for row in csv.DictReader(stream) if row['direction'] == direction]
    groups = defaultdict(list)
    with cluster_csv.open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            groups[int(row['LINK_ID'])].append(row)
    if not times or set(groups) != {int(t['link_id']) for t in times}:
        raise ValueError('来源链路与时间表不一致')
    maximum = 0.0
    for index, (row, time) in enumerate(zip(data['frames'], times, strict=True)):
        link = int(time['link_id'])
        identity = (row['sampleIndex'], row['frameId'], row['internalLinkId'], row['timeSeconds'])
        if identity != (index, int(time['frame_id']), link, float(time['time_s'])):
            raise ValueError(f'第 {index} 帧身份或时刻有误')
        if row['status'] != 'model_reconstruction' or row['reason'] is not None:
            raise ValueError(f'第 {index} 帧状态有误')
        for offset, real, imag in zip(offsets, row['real'], row['imag'], strict=True):
            terms = []
            for path in groups[link]:
                angle = math.radians(float(path['PHASE_DEG'])) - math.tau * offset * float(path['DELAY'])
                amplitude = 10 ** (-float(path['PL']) / 20)
                terms.append(amplitude * complex(math.cos(angle), math.sin(angle)))
            expected = complex(math.fsum(v.real for v in terms), math.fsum(v.imag for v in terms))
            error = abs(complex(real, imag) - expected)
            if not error <= 1e-12 * sum(map(abs, terms)):
                raise ValueError(f'第 {index} 帧 {offset} Hz 的复数结果超差')
            maximum = max(maximum, error)
    return dict(direction=direction, frames=len(times), frequencies=len(offsets),
                last=identity, maximumAbsoluteError=maximum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('output', 'cluster_csv', 'timing_csv'):
        parser.add_argument(argument, type=Path)
    parser.add_argument('--direction', required=True, choices=('uplink', 'downlink'))
    parser.add_argument('--offsets-hz', required=True, type=float, nargs='+')
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.cluster_csv, args.timing_csv,
                            args.direction, args.offsets_hz)))
