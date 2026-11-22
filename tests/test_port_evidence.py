import json

import numpy as np
import pytest

from mimo_ota_simulator.port_evidence import analyze_frame


def link(angles=(359, 1), labels=(1, 2)):
    return dict(
        h=np.array([1, 1j]),
        delay=np.array([0, 0.25]),
        aoa=np.array(angles),
        aod=np.array(angles),
        zoa=np.full(2, 90),
        zod=np.full(2, 90),
        labels=np.array(labels),
        row_ids=np.array([10, 20]),
    )


def test_members():
    ref = link()
    moved = link((1, 1), (2, 2))
    moved["expected"] = {
        1: dict(count=1, real=1, imag=0),
        2: dict(count=1, real=0, imag=1),
    }
    result = analyze_frame(0, {(0, 0): ref, (1, 0): moved}, [0])["p2_106"]
    assert {w["cluster"] for w in result["warnings"]} == {1, 2}
    ray = result["links"][2]["rays"][0]
    assert ray["candidates"][0]["members"][0]["arrival_deg"] == pytest.approx(2)
    assert ray["assigned_cluster"] == 2
    json.dumps(result, allow_nan=False)


def test_tie_and_missing_reference_member():
    ref = link((0, 0))
    ref["delay"][:] = 0
    result = analyze_frame(2, {(0, 0): ref, (1, 0): link(labels=(9, 1))}, [0])
    rays = result["p2_106"]["links"][2]["rays"]
    assert rays[0]["status"] == rays[1]["status"] == "unverified"


def test_delays_and_frequency_sign():
    links = {(t, r): link(labels=(1, 1)) for t in range(4) for r in range(2)}
    links[1, 0]["delay"] = np.array([0, 0])
    result = analyze_frame(0, links, [-1, 0, 1], 1e-12)["p2_107"]
    row = result["links"][0]
    assert row["raw"][2] == pytest.approx(dict(real=2, imag=0))
    for scheme in ("A", "B"):
        assert row[scheme][1] == pytest.approx(row["raw"][1])
    assert row["A"][2] != row["B"][2]
    assert len(result["phase_pairs"]) == 16
    assert result["phase_pairs"][0]["A"]["errors_deg"][0] is None
    assert result["matrix"][1]["A"]["nmse"] == pytest.approx(0)
    json.dumps(result, allow_nan=False)


def test_missing_link_and_unknown_phase_floor():
    result = analyze_frame(1, {(0, 0): link()}, [0])["p2_107"]
    assert result["matrix"][0]["B"]["nmse"] is None
    assert result["links"][0]["B"] is None


def test_partial_expected_and_compact_samples():
    data = link(labels=(1, 1))
    data["expected"] = {1: {"real": 1.0, "imag": 0.0}}
    part = analyze_frame(0, {(0, 0): data}, [0])["p2_106"]
    assert len(part["links"][0]["rays"]) == 1
    assert part["warnings"][0]["fields"] == ["imag"]
    assert part["links"][0]["clusters"][0]["checks"]["count"] == "unverified"


def test_wrapped_phase():
    links = {(t, r): link(labels=(1, 1)) for t in range(4) for r in range(2)}
    # 原始响应相位为 0°，端口内加权时延方案在此例也恰为 0°。
    # 改变第二条射线的幅度，构造由 +179° 跨到负角的端口对。
    links[0, 0]["h"] = np.array([1, 0.5j])
    links[0, 1]["h"] = np.array([np.exp(-1j * np.deg2rad(179)), 0])
    part = analyze_frame(1, links, [0, 1], 1e-12)["p2_107"]
    phase = part["phase_pairs"][0]["A"]["errors_deg"][1]
    assert phase == pytest.approx(np.rad2deg(np.arctan(0.5)) - 18)


def test_only_compressed_delay_changes(monkeypatch):
    from mimo_ota_simulator import port_evidence

    links = {(t, r): link(labels=(1, 1)) for t in range(4) for r in range(2)}
    links[0, 0]["h"] = np.array([2, 1j])
    original = analyze_frame(1, links, [0, 1])["p2_107"]["links"]
    compute = port_evidence._response

    def shifted(grid, h, delay):
        # 唯有目标端口的压缩簇具有这个复系数，原始两条射线仍使用输入时延。
        if len(h) == 1 and h[0] == 2 + 1j:
            delay = np.asarray(delay) + 0.125
        return compute(grid, h, delay)

    monkeypatch.setattr(port_evidence, "_response", shifted)
    changed = analyze_frame(1, links, [0, 1])["p2_107"]["links"]
    assert original[1:] == changed[1:]
    assert original[0]["raw"] == changed[0]["raw"]
    for scheme, representative in (("A", 0.25 / 5), ("B", 8 * 0.25 / (5 + 7 * 2))):
        assert original[0][scheme][0] == changed[0][scheme][0]
        for result, shift in ((original, 0), (changed, 0.125)):
            predicted = (2 + 1j) * np.exp(-2j * np.pi * (representative + shift)) - 3
            assert result[0][scheme + "_error"][1] == pytest.approx(
                dict(real=predicted.real, imag=predicted.imag)
            )
        assert original[0][scheme + "_error"][1] != changed[0][scheme + "_error"][1]


def test_empty_link_and_poles():
    data = link()
    data["zoa"][:] = data["zod"][:] = 0
    links = {(t, r): data for t in range(4) for r in range(2)}
    links[0, 1] = {k: v[:0] for k, v in data.items()}
    part = analyze_frame(1, links, [0], 1e-12)
    assert part["p2_107"]["links"][1]["B"] == [dict(real=0, imag=0)]
    assert part["p2_107"]["phase_pairs"][0]["A"]["valid_count"] == 0
    assert part["p2_106"]["links"][0]["rays"][0]["status"] == "unverified"


def test_download_includes_every_source_row():
    data = link((1, 359), (2, 9))
    data["row_ids"] = np.array([80, 70])
    result = analyze_frame(1, {(0, 0): link(), (1, 0): data}, [0])
    raw = result["p2_106"]["links"][2]["raw"]
    assert raw["row_id"] == [80, 70]
    assert raw["assigned_cluster"] == [2, 9]
    assert raw["distinguishable"] == [False, False]
    assert raw["delay_s"]["nearest_cluster"] == [1, 2]
    assert raw["delay_s"]["nearest_row"] == [10, 20]
    assert raw["delay_s"]["assigned_row"] == [20, None]
    assert raw["delay_s"]["assigned_delta"] == [0.25, None]
    assert raw["arrival_deg"]["nearest_row"] == [20, 10]
    assert raw["departure_deg"]["nearest_delta"] == pytest.approx([0, 0])
    missing = analyze_frame(1, {(1, 0): data}, [0])["p2_106"]["links"][2]["raw"]
    assert missing["arrival_deg"]["nearest_delta"] == [None, None]
    json.dumps(missing, allow_nan=False)


@pytest.mark.parametrize(
    "expected,status",
    [
        ({"real": 1, "imag": 1}, "partial"),
        ({"real": 2}, "warning"),
        ({"count": 2, "real": 1, "imag": 1}, "matched"),
        ({}, "unverified"),
    ],
)
def test_expected_status(expected, status):
    data = link(labels=(2, 2))
    data["expected"] = {2: expected}
    result = analyze_frame(1, {(0, 0): data}, [0])
    assert result["p2_106"]["links"][0]["clusters"][0]["expected_status"] == status
