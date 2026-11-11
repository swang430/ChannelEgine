"""标准ASC更新率保真；真实导出入口，解析参考独立于生产合成函数。"""
import base64
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from mimo_ota_simulator.data_models import PropsimExportConfig
from mimo_ota_simulator.exporters import MFSWRawExporter, PropsimASCIIExporter


@pytest.fixture(params=[False, True], ids=['static', 'dynamic'])
def dynamic(request):
    return request.param


@pytest.fixture(params=['dl', 'ul'])
def direction(request):
    return request.param


@pytest.fixture(params=['disk', 'zip'])
def transport(request):
    return request.param


def result_for(dynamic):
    taps = [dict(cluster_index=0, delay=1e-7, coeff=np.ones((1, 1), complex))]
    rays = [dict(cluster_index=0, W_base=np.ones((1, 1), complex), phi_init=0, doppler_hz=40)]
    result = dict(pathloss_db=0, center_frequency_hz=2.4e9,
                  channel_impulse_response=taps, channel_impulse_response_ul=taps)
    if dynamic:
        result.update(ray_components=rays, ray_components_ul=rays)
    return result


def exporter(tmp_path, rate, cls=PropsimASCIIExporter):
    return cls(PropsimExportConfig(filename=str(tmp_path / 'wave.asc'), mode='B',
                                  duration_s=.02, sample_rate_hz=rate))


def read_export(instance, dynamic, direction, transport):
    result = result_for(dynamic)
    if transport == 'disk':
        return {Path(p).name: Path(p).read_text() for p in instance.export(result, direction=direction)}
    encoded = instance.export_to_zip_memory(result, base_name='wave', direction=direction)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        return {name: archive.read(name).decode() for name in archive.namelist()}


@pytest.mark.parametrize('rate', [1000.04, 1000.05, 160.11076569511297,
                                float(np.nextafter(1000.1, np.inf))])
def test_standard_rejects_lossy_rate(tmp_path, dynamic, direction, transport, rate):
    instance = exporter(tmp_path, rate)
    sentinel = tmp_path / 'wave_In1_Out1.asc'
    if transport == 'disk':
        sentinel.write_text('已有文件不得覆盖')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError) as error:
        read_export(instance, dynamic, direction, transport)
    assert '当前 ASC 导出器' in str(error.value)
    assert str(rate) in str(error.value) and format(rate, '.1f') in str(error.value)
    assert instance.config.sample_rate_hz == rate
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('rate', [1000., 1000.1, 26092.1])
def test_allowed_rate_header_and_phase(tmp_path, dynamic, direction, transport, rate):
    texts = read_export(exporter(tmp_path, rate), dynamic, direction, transport)
    assert list(texts) == ['wave_In1_Out1.asc']
    text = next(iter(texts.values()))
    rate_line = next(line for line in text.splitlines() if 'CIR_Update_Rate' in line)
    assert rate_line == f'{rate:.1f} CIR_Update_Rate'
    assert float(rate_line.split()[0]) == rate
    rows = np.loadtxt(io.StringIO(text.split('***** Tap data *****\n')[1]))
    count = int(.02 * rate)
    assert rows.shape == (count, 3)
    expected = np.exp(2j*np.pi*40*np.arange(count)/rate) if dynamic else np.ones(count)
    # 现有实虚部各六位科学计数输出，仅比较序列化精度，不作物理精度验收。
    np.testing.assert_allclose(rows[:, 1]+1j*rows[:, 2], expected, rtol=0, atol=8e-8)
    np.testing.assert_array_equal(rows[:, 0], np.full(count, 1e-7))


def test_raw_preserves_rate_and_existing_format(tmp_path, dynamic, direction, transport):
    rate = 1000.04
    texts = read_export(exporter(tmp_path, rate, MFSWRawExporter), dynamic, direction, transport)
    if transport == 'zip':
        metadata = texts.pop('mfsw_metadata.txt')
        assert f'SampleRate_Hz: {rate}\n' in metadata
        assert list(texts) == ['wave_In1_Out1.mfsw']
    else:
        assert list(texts) == ['wave_In1_Out1.asc']
    text = next(iter(texts.values()))
    count = int(.02*rate)
    expected = np.exp(2j*np.pi*40*np.arange(count)/rate) if dynamic else np.ones(count)
    expected_text = ''.join(f'{1e-7:.9e}\t{v.real:.6e}\t{v.imag:.6e}\n' for v in expected)
    assert text == expected_text
    assert 'CIR_Update_Rate' not in text
