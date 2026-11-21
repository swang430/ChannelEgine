import numpy as np
import pytest

from mimo_ota_simulator.joint_cluster_response import reduce_link


def test_shared_membership_preserves_each_links_complex_sum_and_delay():
    for coeff, delay in (([1, 1j], [3.0, 5.0]), ([-1j, -1], [4.0, 6.0])):
        result = reduce_link([7, 7], coeff, delay, [179.0, -179.0], [10.0, -10.0])
        cluster = result[7]
        assert cluster["coefficient"] == sum(coeff)
        assert cluster["delay_s"] == pytest.approx(np.mean(delay))
        assert cluster["azimuth_spread_deg"] == pytest.approx(1.0, abs=0.01)
        assert cluster["doppler_spread_hz"] == pytest.approx(10.0)


def test_unassigned_path_cannot_be_silently_grouped():
    with pytest.raises(ValueError):
        reduce_link([-1], [1j], [1.0], [0.0], [0.0])


def test_tracking_uses_geometry_and_reports_unmatched_power():
    from mimo_ota_simulator.joint_cluster_response import associate_frames

    def cluster(delay, angle, power):
        return dict(delay_s=delay, azimuth_deg=angle, path_power=power)

    previous = {8: cluster(1.0, 179.0, 4.0), 3: cluster(8.0, 15.0, 1.0)}
    current = {
        0: cluster(8.1, 15.0, 1.0),
        9: cluster(1.1, -179.0, 4.0),
        4: cluster(5.0, 90.0, 2.0),
    }
    result = associate_frames(previous, current, delay_limit_s=0.2, angle_limit_deg=3.0)
    assert result["previous_by_current"] == {0: 3, 9: 8}
    assert result["unmatched_power_fraction"] == pytest.approx(2 / 7)
    previous[7] = previous[8]
    result = associate_frames(previous, current, delay_limit_s=0.2, angle_limit_deg=3.0)
    assert 9 not in result["previous_by_current"]


def test_departure_wrap_and_zenith_spreads_use_each_paths_power():
    c = reduce_link(
        [0, 0],
        [1, 1j],
        [1, 2],
        [0, 0],
        [0, 0],
        departure_azimuth_deg=[-179, 179],
        arrival_zenith_deg=[30, 40],
        departure_zenith_deg=[100, 104],
    )[0]
    assert c["departure_azimuth_spread_deg"] == pytest.approx(1.0)
    assert c["arrival_zenith_spread_deg"] == pytest.approx(5.0)
    assert c["departure_zenith_spread_deg"] == pytest.approx(2.0)


@pytest.mark.parametrize("h,delay", [([1e155], [1]), ([1.0], np.array([1 + 2j]))])
def test_unrepresentable_paths_are_rejected(h, delay):
    with pytest.raises(ValueError):
        reduce_link([0], h, delay, [0], [0])


def test_large_finite_powers_preserve_unmatched_fraction():
    from mimo_ota_simulator.joint_cluster_response import associate_frames

    a = dict(delay_s=0.0, azimuth_deg=0.0, path_power=1e308)
    b = dict(a, delay_s=10.0)
    result = associate_frames(
        {0: a}, {0: a, 1: b}, delay_limit_s=1.0, angle_limit_deg=1.0
    )
    assert result["unmatched_power_fraction"] == 0.5
