import importlib

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('mimo_ota_simulator.channel_comparison'), '缺少同预算比较计算模块'
    return importlib.import_module('mimo_ota_simulator.channel_comparison')


def test_response_uses_offsets_without_double_carrier_phase():
    c = api()
    np.testing.assert_allclose(c.frequency_response([1j], [1e-9], [0, 250e6]), [1j, 1])
    taps = c.member_taps([1, 1j], [1e-9, 3e-9], [4, 4], tap_budget=24)
    np.testing.assert_allclose(taps['coefficients'], [1 + 1j])
    np.testing.assert_allclose(taps['delays_s'], [2e-9])


def test_matrix_nmse_and_wrapped_phase_share_common_mask():
    c = api()
    raw = np.ones((2, 2), complex)
    asc = np.array([[1, 1e-10], [1j, 1]])
    models = {'asc': asc, 'members': raw.copy()}
    r = c.compare_responses(raw, models, [0, 1], [(0, 1)], amplitude_floor=1e-8)
    assert r['methods']['asc']['matrix_nmse'] == pytest.approx([1, (1-1e-10)**2/2])
    assert r['methods']['asc']['phase_error_deg'][0] == [90.0, None]
    assert r['methods']['members']['phase_error_deg'][0] == [0.0, None]
    assert r['coverage'] == {'phase_valid': 1, 'phase_total': 2, 'phase_fraction': .5,
                             'matrix_valid': 2, 'matrix_total': 2, 'matrix_fraction': 1.0}
    assert r['formal_verdict'] is None


def test_zero_reference_unknown_alignment_and_power_are_not_pass():
    c = api()
    r = c.compare_responses(np.zeros((2, 1)), {'asc': np.ones((2, 1))}, [0], [(0, 1)],
                            amplitude_floor=1e-8)
    assert r['methods']['asc']['matrix_nmse'] == [None]
    assert r['coverage']['matrix_fraction'] == 0
    assert r['alignment_status'] == 'unverified'
    p = c.power_summary([1, -1])
    assert p['carrier_coherent_gain'] == 0
    assert p['tap_power_sum'] == 2
    assert p['absolute_received_power_w'] is None
    assert c.power_summary([1, 1], input_power_w=2)['absolute_received_power_w'] == 8


@pytest.mark.parametrize('action', [
    lambda c: c.member_taps([1], [1], [-1]),
    lambda c: c.member_taps([1, 1], [1, 2], [0, 1], tap_budget=1),
    lambda c: c.frequency_response([1], [-1], [0]),
    lambda c: c.frequency_response([np.nan], [1], [0]),
    lambda c: c.compare_responses(np.ones((2, 1)), {'a': np.ones((1, 1))}, [0], [(0, 1)], amplitude_floor=1e-8),
    lambda c: c.compare_responses(np.ones((2, 1)), {'a': np.ones((2, 1))}, [0], [(0, 0)], amplitude_floor=1e-8),
    lambda c: c.compare_responses(np.ones((2, 1)), {'a': np.ones((2, 1))}, [0], [(0, 1)], amplitude_floor=0),
])
def test_invalid_inputs_rejected(action):
    with pytest.raises(ValueError):
        action(api())


def test_asc_sampling_recovers_37_anchors_and_separates_last_frame():
    c = api()
    # 两段线性时延；分界落在两个采样点之间，使用左右外推分别留存。
    v = np.arange(12, dtype=float)
    delay = np.where(v < 5.3, 10 + 2*v, 10 + 2*5.3 + 4*(v-5.3))
    data = np.zeros((12, 2, 1, 3))
    data[..., 0] = delay[:, None, None]
    data[..., 1] = 1
    sampled = c.recover_asc_frames(data, rate_hz=1, source_frame_count=3, threshold_ns=1e-7)
    assert sampled['sample_positions'] == pytest.approx([0, 5.3])
    assert sampled['unmatched_source_frames'] == [2]
    assert sampled['time_status'] == 'unverified'
    np.testing.assert_allclose(sampled['delays_s'][1], 20.6e-9)
    np.testing.assert_allclose(sampled['right_delays_s'][1], 20.6e-9)


def test_asc_cannot_invent_frames_from_unpaired_or_missing_kinks():
    c = api()
    with pytest.raises(ValueError):
        c.recover_asc_frames(np.zeros((8, 2, 1, 3)), rate_hz=1, source_frame_count=3)


def test_phase_wrap_floor_boundary_and_simulation_cannot_pass():
    c = api()
    raw = np.array([[np.exp(1j*np.deg2rad(179)), 1e-8], [1, 1]])
    model = np.array([[np.exp(1j*np.deg2rad(-179)), 1], [1, 1]])
    r = c.compare_responses(raw, {'model': model}, [0, 1], [(0, 1)],
                            amplitude_floor=1e-8, simulated=True,
                            time_basis='模拟时间', delay_zero_basis='模拟零点')
    assert r['methods']['model']['phase_error_deg'][0] == pytest.approx([2, None])
    assert r['simulated'] is True
    assert r['formal_verdict'] is None
    assert r['alignment_status'] == 'unverified'


def test_common_phase_mask_includes_the_other_model_and_no_hidden_rephasing():
    c = api()
    raw = np.ones((2, 1), complex)
    r = c.compare_responses(raw, {'good': raw, 'weak': np.array([[0], [1j]])}, [0], [(0, 1)], amplitude_floor=1e-8)
    assert r['methods']['good']['phase_error_deg'] == [[None]]
    assert r['coverage']['phase_valid'] == 0
    shifted = c.frequency_response([1], [1e-9], [250e6])
    r = c.compare_responses(raw, {'shift': np.array([shifted, [1]])}, [250e6], [(0, 1)], amplitude_floor=1e-8)
    assert r['methods']['shift']['phase_error_deg'][0] == pytest.approx([90.0])
    assert r['methods']['shift']['matrix_nmse'] == pytest.approx([1.0])


def test_json_serialization_and_empty_paths_preserve_unknowns():
    import json
    c = api()
    np.testing.assert_array_equal(c.frequency_response([], [], [0, 1]), [0, 0])
    r = c.compare_responses(np.zeros((2, 1)), {'empty': np.zeros((2, 1))}, [0], [(0, 1)], amplitude_floor=1e-8)
    json.dumps(r, allow_nan=False)
    assert r['methods']['empty']['phase_error_deg'] == [[None]]


def test_invalid_interpolation_cannot_turn_large_negative_amplitude_into_a_path():
    c = api()
    v = np.arange(12, dtype=float)
    data = np.zeros((12, 2, 1, 3))
    data[..., 0] = np.where(v < 5.3, 10+2*v, 20.6+4*(v-5.3))[:, None, None]
    data[..., 1] = 1
    data[4, ..., 1] = 10
    with pytest.raises(ValueError, match='负时延或负幅度'):
        c.recover_asc_frames(data, rate_hz=1, source_frame_count=3)
