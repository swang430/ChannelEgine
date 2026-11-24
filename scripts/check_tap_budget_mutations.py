"""仅内存变异，不修改生产文件；每个变异须被现有行为回归检出。"""
import importlib.util
from pathlib import Path
import sys
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mimo_ota_simulator.tap_budget as production
spec=importlib.util.spec_from_file_location('budget_tests',Path(__file__).resolve().parents[1]/'tests/test_tap_budget.py')
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
source=Path(production.__file__).read_text()
cases=[
 ('预算被当作必用数','used=base_counts.copy()','used=np.full_like(base_counts,budget)',tests.test_budget_is_ceiling_and_carrier_is_coherent),
 ('相干和被替换为幅度和','return h.sum()*','return abs(h).sum()*',tests.test_budget_is_ceiling_and_carrier_is_coherent),
 ('空半簇虚增抽头','int(all(m.any() for m in masks))','1',tests.test_shared_cut_and_each_port_coherent_sum_with_missing_halves),
 ('丢失全口共用分界','cuts.append(cut)','cuts.append(None)',tests.test_shared_cut_and_each_port_coherent_sum_with_missing_halves),
 ('正式判定污染',"formal_verdict=None","formal_verdict='pass'",tests.test_budget_is_ceiling_and_carrier_is_coherent),
 ('排序方向反转',"-np.array(scores)[:,si]","np.array(scores)[:,si]",tests.test_real_38_frames_against_independent_research),
 ('共享时延误用独立时延','[w@t/w.sum(),common]','[w@t/w.sum(),w@t/w.sum()]',tests.test_real_38_frames_against_independent_research),
]
failed=[]
try:
    for name,old,new,test in cases:
        assert old in source
        mutant=types.ModuleType(production.__name__);mutant.__package__=production.__package__
        exec(compile(source.replace(old,new),production.__file__,'exec'),mutant.__dict__)
        sys.modules[production.__name__]=mutant
        try: test()
        except (AssertionError,ValueError): print('检出:',name)
        else: failed.append(name);print('漏检:',name)
finally: sys.modules[production.__name__]=production
if failed: raise SystemExit('漏检:'+','.join(failed))
print(f'{len(cases)}/{len(cases)} 检出')
