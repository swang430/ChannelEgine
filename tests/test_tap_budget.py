import importlib.util
from pathlib import Path
import numpy as np
import pytest

MODULE = Path(__file__).parents[1]/'mimo_ota_simulator/tap_budget.py'

def api():
    assert MODULE.exists(), '缺少固定成员预算计算模块'
    from mimo_ota_simulator.tap_budget import compare_tap_budgets
    return compare_tap_budgets


def test_budget_is_ceiling_and_carrier_is_coherent():
    run = api()
    paths = [(np.array([1, 1j, .2]), np.array([0., 2e-8, 4e-8]), np.array([0,0,1]))]*2
    result = run(paths, [-1e7,0,1e7], [(0,1)], budgets=(2,4), amplitude_floor=1e-8)
    for scheme in result['strategies'].values():
        assert scheme['budgets']['4']['used_taps'] == [3,3]
        np.testing.assert_allclose(scheme['budgets']['4']['response'][:,1], [1.2+1j]*2)
        assert scheme['budgets']['2']['used_taps'] == [2,2]
    assert result['formal_verdict'] is None
    assert result['alignment_status'] == 'unverified'


@pytest.mark.parametrize('case', ['budget','labels','nan','empty','strategy_floor'])
def test_invalid_inputs_fail_closed(case):
    run=api()
    paths=[(np.array([1.,2.]),np.array([0.,1e-8]),np.array([0,1]))]*2
    kwargs=dict(budgets=(2,3),amplitude_floor=1e-8)
    if case=='budget': kwargs['budgets']=(1,3)
    if case=='labels': paths[0]=(paths[0][0],paths[0][1],np.array([-1,1]))
    if case=='nan': paths[0]=(np.array([np.nan,2]),paths[0][1],paths[0][2])
    if case=='empty': paths=[]
    if case=='strategy_floor': kwargs['amplitude_floor']=0
    with pytest.raises(ValueError): run(paths,[-1e7,0,1e7],[(0,1)],**kwargs)


def test_equal_delays_are_not_artificially_split():
    run=api()
    p=[(np.array([1,1j]),np.array([1e-8,1e-8]),np.array([0,0]))]*2
    r=run(p,[-1e7,0,1e7],[(0,1)],budgets=(1,3),amplitude_floor=1e-8)
    for s in r['strategies'].values():
        assert s['budgets']['3']['used_taps']==[1,1]
        assert s['budgets']['3']['split_clusters']==[]


def test_real_38_frames_against_independent_research():
    run=api()
    import sys
    sys.path.insert(0,'/private/tmp/p2109-vrt/scripts')
    from mpdb_port_evidence import read_downlink
    root=Path('/Users/Simon/Tools/MIMO-First')
    digest,ports,rays=read_downlink(next((root/'RT-IMT-2030').glob('*MPDB.zip')))
    baseline=Path('/Users/Simon/.codex/visualizations/2026/10/06/01a11166-83b0-7532-8f2f-30ee597e6d2f/two-delay-comparison/points.npz')
    pairs=[(a,b) for a in range(8) for b in range(a+1,8) if a//2==b//2 or a%2==b%2]
    with np.load(root/'docs/evidence/imt-three/members.npz',allow_pickle=False) as members,np.load(baseline,allow_pickle=False) as expected:
        assert str(members['source_sha256'])==digest==str(expected['mpdb_sha256'])
        for frame in range(38):
            paths=[]
            for link in range(8):
                lid,=ports['link_id'][(ports['frame_id']==frame)&(ports['tx_ant']==link//2)&(ports['rx_ant']==link%2)]
                mask=(rays['link_id']==lid)&(abs(rays['h'])>0)
                paths.append((rays['h'][mask],rays['delay'][mask],members[f'L{lid}']))
            result=run(paths,expected['frequency_offset_hz'],pairs,amplitude_floor=1e-8)
            for si,name in enumerate(('independent','shared')):
                for budget in (24,26,31):
                    r=result['strategies'][name]['budgets'][str(budget)]
                    np.testing.assert_allclose(r['metrics']['methods']['candidate']['matrix_nmse'],expected['matrix_nmse'][si,budget-24,frame],rtol=1e-9,atol=1e-12)
                    np.testing.assert_allclose(r['metrics']['methods']['candidate']['phase_error_deg'],expected['abs_phase_error_deg'][si,budget-24,frame],atol=1e-7)
                    np.testing.assert_array_equal(r['used_taps'],24+expected['extra_active_taps'][si,budget-24,frame])


def test_shared_cut_and_each_port_coherent_sum_with_missing_halves():
    run=api()
    paths=[(np.array([1,2j]),np.array([0.,1e-7]),np.array([0,0])),
           (np.array([3.]),np.array([0.]),np.array([0]))]
    result=run(paths,[-1e7,0,1e7],[(0,1)],budgets=(1,2),amplitude_floor=1e-8)
    for scheme in result['strategies'].values():
        assert scheme['budgets']['2']['used_taps']==[2,1]
        assert scheme['ranking'][0]['last_early_delay_s']==0
        np.testing.assert_allclose(scheme['budgets']['2']['response'][:,1],[1+2j,3])


def test_weak_port_point_is_null_and_not_improvement():
    run=api()
    paths=[(np.array([1.,-1.]),np.array([0.,1e-7]),np.array([0,0])),
           (np.array([1.]),np.array([0.]),np.array([0]))]
    r=run(paths,[-1e6,0,1e6],[(0,1)],budgets=(1,2),amplitude_floor=1e-8)
    for s in r['strategies'].values():
        for b in s['budgets'].values():
            assert b['metrics']['methods']['candidate']['phase_error_deg'][0][1] is None
            assert b['delta']['phase_error_deg']['values'][0][1] is None
            assert sum(b['delta']['phase_error_deg'][k] for k in ('improved','worse','unchanged'))==0
