"""真实样本只读回归；无样本时明确跳过，不能作为实样通过。"""
import csv
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from mimo_ota_simulator.channel_comparison import (
    compare_responses, frequency_response, member_taps, recover_asc_frames,
)


def test_imt_37_frames_reproduce_independent_csv(monkeypatch):
    root = Path(os.environ.get('P2108_SAMPLE_ROOT', '/Users/Simon/Tools/MIMO-First'))
    scripts = Path(os.environ.get('P2108_META_SCRIPTS', '/private/tmp/p2108-vrt/scripts'))
    baseline = Path(os.environ.get('P2108_BASELINE', '/Users/Simon/.codex/visualizations/2026/10/06/01a11166-83b0-7532-8f2f-30ee597e6d2f/asc-vs-joint'))
    if not (root / 'RT-IMT-2030').is_dir() or not scripts.is_dir() or not (baseline / 'points.csv').is_file():
        pytest.skip('本机缺少外部实样/参考解析器/独立CSV；未验证真实样本')
    # 外部只读解析器仅供本验收，不成为 ChannelEgine 的运行依赖。
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('p2108_reference_reader', scripts/'mpdb_port_evidence.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    digest, ports, rays = reader.read_downlink(next((root/'RT-IMT-2030').glob('*MPDB.zip')))
    data = np.stack([np.loadtxt(root/'RT-IMT-2030'/f'R0B0S0D_TX{t}_RX{r}_H.asc', skiprows=12).reshape(6188, 24, 3)
                     for t in range(4) for r in range(2)], axis=1)
    rate = float((root/'RT-IMT-2030'/'R0B0S0D_TX0_RX0_H.asc').read_text().splitlines()[9].split()[0])
    frames = recover_asc_frames(data, rate_hz=rate, source_frame_count=38)
    assert len(frames['sample_positions']) == 37
    assert frames['unmatched_source_frames'] == [37]
    np.testing.assert_allclose(frames['times_s'], np.arange(37), atol=1e-10)
    offsets = np.arange(-20, 21)*2.5e6
    pairs = [(a, b) for a in range(8) for b in range(a+1, 8) if a//2 == b//2 or a%2 == b%2]
    with (baseline/'points.csv').open() as fp:
        expected = list(csv.DictReader(fp))
    by_point = {(int(r['frame']), r['method'], float(r['offset_hz']), int(r['port_a']), int(r['port_b'])): r for r in expected}
    with (baseline/'frames.csv').open() as fp:
        frame_rows = list(csv.DictReader(fp))
    compared = 0
    with np.load(root/'docs/evidence/imt-three/members.npz', allow_pickle=False) as members:
        assert str(members['source_sha256']) == digest
        for frame in range(37):
            raw, grouped, asc, right = [], [], [], []
            for i in range(8):
                selected = (ports['frame_id'] == frame) & (ports['tx_ant'] == i//2) & (ports['rx_ant'] == i%2)
                lid, = ports['link_id'][selected]
                select = (rays['link_id'] == lid) & (abs(rays['h']) > 0)
                h, tau = rays['h'][select].astype(complex), rays['delay'][select].astype(float)
                taps = member_taps(h, tau, members[f'L{lid}'], tap_budget=24)
                assert taps['used_taps'] == 24
                raw.append(frequency_response(h, tau, offsets))
                grouped.append(frequency_response(taps['coefficients'], taps['delays_s'], offsets))
                asc.append(frequency_response(frames['coefficients'][frame, i], frames['delays_s'][frame, i], offsets))
                right.append(frequency_response(frames['right_coefficients'][frame, i], frames['right_delays_s'][frame, i], offsets))
            raw, asc, right = map(np.asarray, (raw, asc, right))
            actual = compare_responses(raw, {'Lauraycs': asc, 'Meta-3D': np.asarray(grouped)}, offsets, pairs,
                                       amplitude_floor=1e-8, time_basis=frames['time_basis'])
            assert actual['alignment_status'] == 'unverified'
            assert actual['formal_verdict'] is None
            diagnostic = frame_rows[frame]
            assert np.max(abs(asc-right))/np.max(abs(raw)) == pytest.approx(float(diagnostic['left_right_relative_difference']), abs=1e-10)
            assert np.max(abs(asc[:,20]-raw[:,20])/abs(raw[:,20])) == pytest.approx(float(diagnostic['carrier_max_relative_error']), abs=1e-10)
            for name, result in actual['methods'].items():
                for pair_index, (a, b) in enumerate(pairs):
                    for j, off in enumerate(offsets):
                        exp = by_point[frame, name, float(off), a, b]
                        assert result['matrix_nmse'][j] == pytest.approx(float(exp['matrix_nmse']), rel=1e-9, abs=1e-12)
                        phase = result['phase_error_deg'][pair_index][j]
                        if np.isnan(float(exp['phase_error_deg'])):
                            assert phase is None
                        else:
                            assert phase == pytest.approx(float(exp['phase_error_deg']), abs=1e-7)
                        compared += 1
    assert compared == len(expected) == 48544
