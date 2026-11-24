"""P2-110 只读候选评估。阈值为本次示例约束，不是认证标准。"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import re
import numpy as np


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class EvaluationPlan:
    source_sha256: str
    members_sha256: str
    selection_frames: tuple
    holdout_frames: tuple
    budget_limit: int = 31
    matrix_p95_limit: float = 1.
    phase_p95_limit_deg: float = 60.
    objective: str = 'phase'
    mode: str = 'conducted'

    def __post_init__(self):
        for key in ('selection_frames', 'holdout_frames'):
            v = tuple(getattr(self, key))
            if not v or any(type(x) is not int or x < 0 for x in v) or v != tuple(range(v[0], v[-1]+1)):
                raise ValueError('必须按连续源帧划分非空选择集和留出集')
            object.__setattr__(self, key, v)
        if self.selection_frames[-1]+1 != self.holdout_frames[0]:
            raise ValueError('选择集与留出集须相邻且不重叠')
        if any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in (self.source_sha256,self.members_sha256)):
            raise ValueError('必须固定来源及成员SHA256')
        if type(self.budget_limit) is not int or self.budget_limit < 1:
            raise ValueError('预算必须为正整数')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for v in (self.matrix_p95_limit,self.phase_p95_limit_deg)):
            raise ValueError('阈值必须为有限非负数')
        if self.objective not in ('phase','matrix') or self.mode not in ('conducted','ota'):
            raise ValueError('评估目的或模式无效')


def _summary(plan, candidate, frames, provenance):
    offsets = provenance['offsets_hz']; pairs = provenance['port_pairs']
    points = {'matrix_nmse':[], 'phase_error_deg':[]}
    missing=[]; taps=[]; changes={}; unverified=[]; exclusions=[]
    applicable=None
    if candidate['strategy']=='existing_asc':
        evidence=provenance.get('existing_asc_reference')
        scope=evidence.get('applicable_frames') if isinstance(evidence,dict) else provenance.get('asc_applicable_frames')
        applicable=set(scope) if isinstance(scope,list) and all(type(f) is int and f>=0 for f in scope) else set()
    for frame in frames:
        row=candidate['frames'].get(frame)
        if row is None:
            missing.append(frame); continue
        reasons=[]
        if applicable is not None and frame not in applicable:
            reasons.append('已有ASC缺少本源帧的适用证据')
        used=row.get('used_taps')
        if not isinstance(used,(list,tuple)) or len(used)!=8 or any(type(v) is not int or not 1<=v<=candidate['requested_taps'] for v in used):
            reasons.append('缺少八路完整有效的正整数抽头证据，或超过声明预算')
        elif any(v>plan.budget_limit for v in used):
            reasons.append('实际抽头超过冻结计划预算上限')
        if reasons:
            unverified.append(frame)
            exclusions.append(dict(frame_id=frame,status='unverified',reasons=reasons))
            continue
        taps.extend(used)
        metrics=row['metrics']['methods']['candidate']
        for metric in points:
            arr=np.asarray(metrics[metric],dtype=object)
            expected=(len(offsets),) if metric=='matrix_nmse' else (len(pairs),len(offsets))
            if arr.shape!=expected: raise ValueError('逐点误差与频偏、端口对不匹配')
            for index in np.ndindex(arr.shape):
                value=arr[index]
                if value is None: continue
                if isinstance(value,bool) or not isinstance(value,(int,float,np.number)) or not math.isfinite(value) or value<0:
                    raise ValueError('误差必须为有限非负数或null')
                point=dict(frame_id=frame,offset_hz=offsets[index[-1]],value=float(value))
                if metric=='phase_error_deg': point['port_pair']=pairs[index[0]]
                points[metric].append(point)
            d=row.get('delta',{}).get(metric,{})
            counts=changes.setdefault(metric,dict(improved=0,worse=0,unchanged=0))
            for k in counts: counts[k]+=int(d.get(k,0))
    result=dict(frames=list(frames),missing_frames=missing,unverified_frames=unverified,excluded_frame_evidence=exclusions,actual_taps_min=min(taps) if taps else None,actual_taps_max=max(taps) if taps else None,changes_against_same_strategy_baseline=changes)
    for metric,values in points.items():
        total=len(frames)*len(offsets)*(len(pairs) if metric=='phase_error_deg' else 1)
        worst=max(values,key=lambda x:x['value']) if values else None
        result[metric]=dict(valid=len(values),total=total,coverage=len(values)/total,p95=float(np.percentile([x['value'] for x in values],95)) if values else None,worst=worst)
    result['constraints_met']=bool(not missing and not unverified and candidate['requested_taps']<=plan.budget_limit and all(result[m]['valid']==result[m]['total'] and result[m]['p95']<=limit for m,limit in [('matrix_nmse',plan.matrix_p95_limit),('phase_error_deg',plan.phase_p95_limit_deg)]))
    return result


def evaluate_candidates(plan, candidates, *, provenance, manual_candidate=None):
    """只用选择集排序；返回所选方案的独立留出结果，人工选择不改变建议。"""
    provenance=json.loads(json.dumps(provenance,allow_nan=False))
    for key in ('source_sha256','members_sha256'):
        if provenance.get(key)!=getattr(plan,key): raise ValueError('来源不符合冻结计划')
    for key in ('time_basis','delay_zero_basis','offsets_hz','port_pairs','amplitude_floor'):
        if not provenance.get(key): raise ValueError('来源、方法及适用范围依据缺失')
    offsets=provenance['offsets_hz']; pairs=provenance['port_pairs']
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in offsets) or len(set(offsets))!=len(offsets): raise ValueError('频偏必须有限且唯一')
    if any(len(p)!=2 or any(type(v) is not int for v in p) or not 0<=p[0]<p[1]<8 for p in pairs) or len(set(map(tuple,pairs)))!=len(pairs): raise ValueError('端口对无效')
    floor=provenance['amplitude_floor']
    if not isinstance(floor,(int,float)) or not math.isfinite(floor) or floor<=0: raise ValueError('幅度下限无效')
    if not candidates or manual_candidate is not None and manual_candidate not in candidates: raise ValueError('候选或人工选择无效')
    selection={}
    for name,c in candidates.items():
        if not isinstance(name,str) or not name or c.get('strategy') not in ('independent','shared','existing_asc') or type(c.get('requested_taps')) is not int or c['requested_taps']<1: raise ValueError('候选方法或预算无效')
        selection[name]=_summary(plan,c,plan.selection_frames,provenance)
    first,second=('phase_error_deg','matrix_nmse') if plan.objective=='phase' else ('matrix_nmse','phase_error_deg')
    eligible=[k for k,v in selection.items() if v['constraints_met']]
    selected=min(eligible,key=lambda k:(selection[k][first]['p95'],selection[k][second]['p95'],candidates[k]['requested_taps'],k)) if eligible else None
    frozen=dict(plan=asdict(plan),provenance=provenance,selection=selection,selected_candidate=selected,candidate_parameters={k:{key:c[key] for key in ('strategy','requested_taps')} for k,c in candidates.items()})
    frozen['selection_input_fingerprint']=_digest({k:{str(f):c['frames'].get(f) for f in plan.selection_frames} for k,c in candidates.items()})
    frozen['manual_candidate']=manual_candidate
    fingerprint=_digest(frozen)
    holdout=_summary(plan,candidates[selected],plan.holdout_frames,provenance) if selected else None
    return dict(schema='candidate-evaluation/1',**frozen,plan_fingerprint=_digest(asdict(plan)),selection_fingerprint=fingerprint,holdout=holdout,manual_holdout=_summary(plan,candidates[manual_candidate],plan.holdout_frames,provenance) if manual_candidate else None,unmatched_frame_review=_summary(plan,candidates[selected or manual_candidate],(37,),provenance) if 37 in plan.holdout_frames and (selected or manual_candidate) else None,status='selected' if selected else 'no_candidate_meets_constraints',automatic_adoption=False,formal_verdict=None,alignment_status='unverified',ota_status='unverified' if plan.mode=='ota' else 'not_evaluated',threshold_scope='本次示例约束，不是认证标准',asc_unmatched_frames=[f for f in plan.selection_frames+plan.holdout_frames if f==37],limitations=['只读辅助分析，不证明硬件或场性能','OTA场参考未核验' ] if plan.mode=='ota' else ['只读传导辅助分析，不证明硬件性能'])
