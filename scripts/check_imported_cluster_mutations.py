"""仅在内存中变异导入模块；不改写生产文件。仓库根目录运行。"""
import contextlib
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pytest
from mimo_ota_simulator import imported_cluster_trajectory as module

source = Path(module.__file__).read_text(encoding='utf-8')
mutations = {
    'schema': ("data['schemaVersion'] == 'meta3d-cluster-trajectory/1'", 'True'),
    'units': ("data['units'] == UNITS", 'True'),
    'identity': ("data['clusterIdentity'] == 'frame-local-only'", 'True'),
    'time': ("number(frame['timeSeconds']) == time", 'True'),
    'finite': ('type(value) in (int, float) and math.isfinite(value)', 'True'),
    'duplicate': ("key not in result, f'JSON重复字段：{key}'", "True, f'JSON重复字段：{key}'"),
    'imaginary': ("complex(p['hReal'], p['hImag'])", "complex(p['hReal'], 0)"),
    'phase_sign': ('-2j * math.pi', '2j * math.pi'),
    'delay_units': ("p['delayNs'] * 1e-9", "p['delayNs'] * 1e-6"),
    'empty_zero': ('real=None, imag=None', 'real=[0.], imag=[0.]'),
    'actual_time': ("timeSeconds=frame['timeSeconds']", "timeSeconds=frame['sampleIndex']"),
    'all_clusters': ('for p in paths)', 'for p in paths[:1])'),
    'size_limit': ('len(raw) <= MAX_BYTES', 'True'),
}
failed = []
try:
    for name, (old, new) in mutations.items():
        if source.count(old) != 1:
            raise RuntimeError(f'{name}: 变异锚点未唯一命中')
        exec(compile(source.replace(old, new), module.__file__, 'exec'), module.__dict__)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = pytest.main([str(ROOT / 'tests/test_imported_cluster_trajectory.py'),
                                  '-q', '--basetemp=' + str(ROOT / '.pytest_tmp_mutations')])
        killed = result == pytest.ExitCode.TESTS_FAILED
        print(f'{name}: {"检出" if killed else "未检出或运行错误"} (exit={int(result)})')
        if not killed:
            failed.append(name)
            print(output.getvalue())
finally:
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
print(f'{len(mutations) - len(failed)}/{len(mutations)} 项变异被检出')
sys.exit(bool(failed))
