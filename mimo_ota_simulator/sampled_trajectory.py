"""从一个方向的聚簇轨迹文件计算全部采样时刻的复频响，将结果写到标准输出。"""
import argparse
import json

from .imported_cluster_trajectory import read_trajectory, synthesize_sampled_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', help='单方向聚簇轨迹 JSON 文件')
    parser.add_argument('--direction', required=True, choices=('downlink', 'uplink'))
    parser.add_argument('--offsets-hz', required=True, nargs='+', type=float)
    args = parser.parse_args()
    try:
        result = synthesize_sampled_trajectory(read_trajectory(args.trajectory),
                                               args.offsets_hz, direction=args.direction)
    except (ValueError, OSError, OverflowError) as exc:
        parser.exit(2, f'轨迹合成失败：{exc}\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
