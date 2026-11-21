import numpy as np
import pytest

from mimo_ota_simulator.incident_field_ota import fit_probe_taps


def test_each_transmitter_and_cluster_uses_same_calibrated_probe_basis():
    # 两位置、每位置两极化、四个实际馈电口。列顺序即端口接线顺序。
    basis = np.eye(4, dtype=complex).reshape(2, 2, 4)
    basis[:, :, 2] *= 1j
    expected = np.arange(32).reshape(4, 2, 4) * (0.1 + 0.2j)
    field = np.einsum("xpf,tkf->tkxp", basis, expected)
    result = fit_probe_taps(basis, field)
    np.testing.assert_allclose(result, expected)


def test_rejects_port_basis_with_no_field_response():
    with pytest.raises(ValueError):
        fit_probe_taps(np.zeros((2, 2, 4)), np.ones((4, 2, 2, 2)))
