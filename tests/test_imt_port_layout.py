import numpy as np
import pytest

from mimo_ota_simulator.imt_probe_bundle import port_layout


def test_nonconsecutive_ports_and_missing_link_in_one_frame():
    rows = [(f, t, r) for f in (0, 1) for t in (1, 3) for r in (2, 4)]
    def inspect(records):
        columns = dict(zip(("FRAME_ID", "TX_ANT", "RX_ANT"), np.array(records).T))
        columns["LINK_ID"] = np.arange(len(records))
        return port_layout(columns.__getitem__)

    assert inspect(rows) == ([1, 3], [2, 4])
    with pytest.raises(ValueError, match="源帧 1 缺少发射端口 3 到接收端口 4"):
        inspect(rows[:-1])
