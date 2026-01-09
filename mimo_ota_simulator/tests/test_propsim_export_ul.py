import numpy as np

from mimo_ota_simulator.data_models import PropsimExportConfig
from mimo_ota_simulator.exporters import PropsimASCIIExporter


def _read_first_data_line(filepath: str) -> list[float]:
    in_data = False
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "***** Tap data *****" in line:
                in_data = True
                continue
            if in_data:
                return [float(x) for x in line.split("\t")]
    return []


def test_propsim_export_ul_uses_ul_impulse_response(tmp_path):
    dl_coeff = np.array([[1.0 + 0j], [0.5 + 0j]])
    ul_coeff = np.array([[2.0 + 0j, 0.25 + 0j]])

    sim_result = {
        "center_frequency_hz": 3.5e9,
        "pathloss_db": 0.0,
        "channel_impulse_response": [
            {"delay": 0.0, "coeff": dl_coeff, "cluster_index": 0}
        ],
        "channel_impulse_response_ul": [
            {"delay": 0.0, "coeff": ul_coeff, "cluster_index": 0}
        ],
    }

    export_config = PropsimExportConfig(
        filename=str(tmp_path / "propsim_dl.asc"),
        mode="B",
        duration_s=0.001,
        sample_rate_hz=1000.0
    )
    exporter = PropsimASCIIExporter(export_config)

    dl_files = exporter.export(sim_result, output_path=str(tmp_path / "propsim_dl.asc"), direction="dl")
    ul_files = exporter.export(sim_result, output_path=str(tmp_path / "propsim_ul.asc"), direction="ul")

    assert len(dl_files) == 2
    assert len(ul_files) == 2

    dl_line = _read_first_data_line(dl_files[0])
    ul_line = _read_first_data_line(ul_files[0])

    dl_re = dl_line[1]
    ul_re = ul_line[1]
    assert np.isclose(dl_re, 1.0, atol=1e-6)
    assert np.isclose(ul_re, 2.0, atol=1e-6)
