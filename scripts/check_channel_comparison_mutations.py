"""P2-108 仅内存变异：验证相位、单位、共同覆盖与未知状态拒绝退化。"""
import contextlib
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pytest
from mimo_ota_simulator import channel_comparison as module

source = Path(module.__file__).read_text(encoding='utf-8')
mutations = {
    'delay_units': ('*1e-9', '*1e-6'),
    'coherent_sum': ('coeff.append(h[sub].sum())', 'coeff.append(abs(h[sub]).sum())'),
    'tap_budget': ('if len(ids) > tap_budget:', 'if False:'),
    'matrix_denominator': ('energy = np.sum(abs(ref)**2, axis=0)', 'energy = np.ones(ref.shape[1])'),
    'common_phase_mask': ('(raw, *arrays.values())]) > floor', '(raw,)]) > floor'),
    'floor_strict': (']) > floor', ']) >= floor'),
    'invalid_phase_zero': ('err[~masks[k]] = np.nan', 'err[~masks[k]] = 0'),
    'coherent_power': ('coherent = float(abs(h.sum())**2)', 'coherent = float(np.sum(abs(h)**2))'),
    'absolute_power_without_reference': ('received = None', 'received = coherent'),
    'formal_verdict': ('formal_verdict=None)', "formal_verdict='pass')"),
    'last_frame': ('unmatched_source_frames=[source_frame_count-1]', 'unmatched_source_frames=[]'),
}
failed = []
try:
    for name, (old, new) in mutations.items():
        if source.count(old) != 1:
            raise RuntimeError(f'{name}: 变异锚点未唯一命中')
        exec(compile(source.replace(old, new), module.__file__, 'exec'), module.__dict__)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = pytest.main([str(ROOT/'tests/test_channel_comparison.py'), '-q'])
        killed = result == pytest.ExitCode.TESTS_FAILED
        print(f'{name}: {"检出" if killed else "未检出或运行错误"} (exit={int(result)})')
        if not killed:
            failed.append(name)
            print(output.getvalue())
finally:
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
print(f'{len(mutations)-len(failed)}/{len(mutations)} 项变异被检出')
sys.exit(bool(failed))
