import copy
import importlib.util
from pathlib import Path
import pytest

MODULE = Path(__file__).resolve().parents[1]/'mimo_ota_simulator/candidate_evaluation.py'

def api():
    assert MODULE.exists(), '候选评估尚未实现'
    from mimo_ota_simulator.candidate_evaluation import EvaluationPlan, evaluate_candidates
    return EvaluationPlan, evaluate_candidates

def inputs():
    provenance=dict(source_sha256='a'*64,members_sha256='b'*64,time_basis='源帧号',delay_zero_basis='输入秒不变',offsets_hz=[-1,1],port_pairs=[[0,1]],amplitude_floor=1e-8)
    def candidate(phase,hold):
        frames={}
        for i in range(4):
            p=phase if i<2 else hold
            frames[i]=dict(used_taps=[2]*8,metrics={'methods':{'candidate':dict(matrix_nmse=[.2,.4],phase_error_deg=[[p,p+1]])}},delta={})
        return dict(strategy='shared',requested_taps=2,frames=frames)
    return provenance, {'A':candidate(10,100),'B':candidate(20,1)}

def test_selection_never_uses_holdout_and_reports_worst_point():
    Plan,evaluate=api(); p,c=inputs()
    plan=Plan('a'*64,'b'*64,(0,1),(2,3))
    r=evaluate(plan,c,provenance=p)
    assert r['selected_candidate']=='A'
    assert r['holdout']['phase_error_deg']['p95']==101
    assert r['holdout']['phase_error_deg']['worst']['frame_id']==2
    assert r['holdout']['phase_error_deg']['worst']['port_pair']==[0,1]
    assert r['holdout']['constraints_met'] is False
    c['B']['frames'][2]['metrics']['methods']['candidate']['phase_error_deg']=[[0,0]]
    assert evaluate(plan,c,provenance=p)['selection_fingerprint']==r['selection_fingerprint']

def test_no_pass_keeps_manual_choice_without_auto_adoption():
    Plan,evaluate=api(); p,c=inputs()
    r=evaluate(Plan('a'*64,'b'*64,(0,1),(2,3),phase_p95_limit_deg=1),c,provenance=p,manual_candidate='B')
    assert r['selected_candidate'] is None and r['holdout'] is None
    assert r['manual_candidate']=='B' and r['status']=='no_candidate_meets_constraints'
    assert r['formal_verdict'] is None and r['automatic_adoption'] is False

def test_missing_points_and_ota_are_not_passed():
    Plan,evaluate=api(); p,c=inputs()
    c['A']['frames'][0]['metrics']['methods']['candidate']['matrix_nmse']=[None,None]
    c['B']['frames'].pop(1)
    r=evaluate(Plan('a'*64,'b'*64,(0,1),(2,3),mode='ota'),c,provenance=p)
    assert r['selected_candidate'] is None
    assert r['ota_status']=='unverified'
    assert r['selection']['A']['matrix_nmse']['valid']==2
    assert r['selection']['B']['missing_frames']==[1]

@pytest.mark.parametrize('sel,hold', [((0,2),(3,4)),((0,1),(1,2)),((0,1),(3,4)),((1,0),(2,3))])
def test_split_rejected(sel,hold):
    Plan,_=api()
    with pytest.raises(ValueError): Plan('a'*64,'b'*64,sel,hold)

def test_provenance_and_nonfinite_rejected():
    Plan,evaluate=api(); p,c=inputs(); plan=Plan('a'*64,'b'*64,(0,1),(2,3))
    p['source_sha256']='c'*64
    with pytest.raises(ValueError): evaluate(plan,c,provenance=p)
    p['source_sha256']='a'*64
    c['A']['frames'][0]['metrics']['methods']['candidate']['matrix_nmse'][0]=float('nan')
    with pytest.raises(ValueError): evaluate(plan,c,provenance=p)

def test_objective_budget_and_plan_are_fixed():
    from dataclasses import FrozenInstanceError
    Plan,evaluate=api(); p,c=inputs()
    for frame in c['A']['frames'].values(): frame['metrics']['methods']['candidate']['matrix_nmse']=[.8,.9]
    assert evaluate(Plan('a'*64,'b'*64,(0,1),(2,3),objective='matrix'),c,provenance=p)['selected_candidate']=='B'
    c['A']['requested_taps']=32
    assert evaluate(Plan('a'*64,'b'*64,(0,1),(2,3)),c,provenance=p)['selected_candidate']=='B'
    plan=Plan('a'*64,'b'*64,[0,1],[2,3])
    with pytest.raises(FrozenInstanceError): plan.budget_limit=99
    assert plan.selection_frames==(0,1)


def test_frame37_kept_separate_and_missing_holdout_not_reselected():
    Plan,evaluate=api(); p,c=inputs()
    for candidate in c.values():
        candidate['frames']={34+i:v for i,v in candidate['frames'].items()}
    c['A']['frames'].pop(37)
    r=evaluate(Plan('a'*64,'b'*64,(34,35),(36,37)),c,provenance=p)
    assert r['selected_candidate']=='A'
    assert r['holdout']['missing_frames']==[37]
    assert r['asc_unmatched_frames']==[37]
    assert r['holdout']['constraints_met'] is False

def test_manual_review_is_separate_from_failed_recommendation():
    Plan,evaluate=api(); p,c=inputs()
    r=evaluate(Plan('a'*64,'b'*64,(0,1),(2,3),phase_p95_limit_deg=1),c,provenance=p,manual_candidate='A')
    assert r['selected_candidate'] is None
    assert r['manual_holdout']['phase_error_deg']['p95']==101
    assert r['holdout'] is None
    assert r['manual_holdout']['constraints_met'] is False

@pytest.mark.parametrize('scope', [None, [], [2,3]])
def test_existing_asc_cannot_select_without_applicable_selection_frames(scope):
    Plan,evaluate=api(); p,c=inputs()
    c={'asc':c['A']}; c['asc']['strategy']='existing_asc'
    for row in c['asc']['frames'].values():
        row['metrics']['methods']['candidate']=dict(matrix_nmse=[0,0],phase_error_deg=[[0,0]])
    if scope is not None: p['existing_asc_reference']={'applicable_frames':scope,'status':'unverified'}
    r=evaluate(Plan('a'*64,'b'*64,(0,1),(2,3)),c,provenance=p,manual_candidate='asc')
    assert r['selected_candidate'] is None
    assert r['selection']['asc']['unverified_frames']==[0,1]
    assert r['selection']['asc']['matrix_nmse']['valid']==0
    assert r['selection']['asc']['matrix_nmse']['p95'] is None


def test_existing_asc_partial_scope_excludes_holdout_and_reports_unknown():
    Plan,evaluate=api(); p,c=inputs(); c={'asc':c['A']}; c['asc']['strategy']='existing_asc'
    p['existing_asc_reference']={'applicable_frames':[0,1,2], 'status':'unverified'}
    r=evaluate(Plan('a'*64,'b'*64,(0,1),(2,3)),c,provenance=p)
    assert r['selected_candidate']=='asc'
    assert r['holdout']['unverified_frames']==[3]
    assert r['holdout']['matrix_nmse']['valid']==2
    assert r['holdout']['constraints_met'] is False


@pytest.mark.parametrize('used', [[0], [2]*7, [2]*9, [0]*8, [2]*7+[None], [2]*7+[True], [3]*8, None])
def test_eight_port_tap_evidence_required_even_for_zero_error(used):
    Plan,evaluate=api(); p,c=inputs(); c={'A':c['A']}
    c['A']['frames'][0]['used_taps']=used
    c['A']['frames'][0]['metrics']['methods']['candidate']=dict(matrix_nmse=[0,0],phase_error_deg=[[0,0]])
    r=evaluate(Plan('a'*64,'b'*64,(0,1),(2,3)),c,provenance=p)
    assert r['selected_candidate'] is None
    assert r['selection']['A']['unverified_frames']==[0]
    assert r['selection']['A']['matrix_nmse']['valid']==2


def test_plan_budget_cannot_be_bypassed_by_declared_or_actual_taps():
    Plan,evaluate=api(); p,c=inputs(); c={'A':c['A']}
    c['A']['requested_taps']=32
    c['A']['frames'][0]['used_taps']=[32]*8
    r=evaluate(Plan('a'*64,'b'*64,(0,1),(2,3)),c,provenance=p)
    assert r['selected_candidate'] is None
    assert r['selection']['A']['unverified_frames']==[0]

@pytest.mark.parametrize('evidence', [
    {'asc_applicable_frames':[]},
    {'existing_asc_reference':{'declared_applicable_frames':[0,1,2,3]}},
    {'existing_asc_reference':{'applicable_frames':[True]}},
    {'asc_applicable_frames':[0,1,2,3], 'existing_asc_reference':{'applicable_frames':[]}},
])
def test_unverified_asc_scope_cannot_be_promoted_by_declared_metadata(evidence):
    Plan,evaluate=api(); p,c=inputs(); p.update(evidence)
    c={'asc':c['A']}; c['asc']['strategy']='existing_asc'
    r=evaluate(Plan('a'*64,'b'*64,(0,1),(2,3)),c,provenance=p)
    assert r['selected_candidate'] is None
    assert r['selection']['asc']['unverified_frames']==[0,1]
