"""固定计划后读取实样；38帧全扫描只用于描述，不参与跨帧参数选择。"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mimo_ota_simulator.candidate_evaluation import EvaluationPlan, evaluate_candidates
from mimo_ota_simulator.tap_budget import compare_tap_budgets

SOURCE='217585159349afe88ca9864be4a98158625996fefc2639af4403e9727682a2dd'
MEMBERS='c3b9da980667ef38cf44f7275a3a12105dcc1b4e95eacbffdf23e0f3eb1bcd1a'

def main():
    # 在读取任何误差之前固定；阈值不是认证标准。
    plan=EvaluationPlan(SOURCE,MEMBERS,tuple(range(26)),tuple(range(26,38)))
    sys.path.insert(0,'/private/tmp/p2109-vrt/scripts')
    from mpdb_port_evidence import read_downlink
    root=Path('/Users/Simon/Tools/MIMO-First')
    archive=next((root/'RT-IMT-2030').glob('*MPDB.zip'))
    member_path=root/'docs/evidence/imt-three/members.npz'
    if hashlib.sha256(member_path.read_bytes()).hexdigest()!=MEMBERS: raise ValueError('成员文件已变更')
    digest,ports,rays=read_downlink(archive)
    if digest!=SOURCE: raise ValueError('源文件已变更')
    offsets=(np.arange(-20,21)*2.5e6).tolist()
    pairs=[(a,b) for a in range(8) for b in range(a+1,8) if a//2==b//2 or a%2==b%2]
    provenance=dict(source_sha256=SOURCE,members_sha256=MEMBERS,time_basis='MPDB source frame id; source physical time unverified',delay_zero_basis='input seconds unchanged; physical zero unverified',offsets_hz=offsets,port_pairs=pairs,amplitude_floor=1e-8,method='P2-109 fixed members, shared early/late cut, coherent coefficient sum',asc_applicable_frames=list(range(37)))
    candidates={f'{s}/{b}':dict(strategy=s,requested_taps=b,frames={}) for s in ('independent','shared') for b in (24,26,31)}
    with np.load(member_path,allow_pickle=False) as members:
        for frame in range(38):
            paths=[]
            for link in range(8):
                lid,=ports['link_id'][(ports['frame_id']==frame)&(ports['tx_ant']==link//2)&(ports['rx_ant']==link%2)]
                mask=(rays['link_id']==lid)&(abs(rays['h'])>0)
                paths.append((rays['h'][mask],rays['delay'][mask],members[f'L{lid}']))
            result=compare_tap_budgets(paths,offsets,pairs,time_basis=provenance['time_basis'],delay_zero_basis=provenance['delay_zero_basis'])
            for s in result['strategies']:
                for b,row in result['strategies'][s]['budgets'].items():
                    candidates[s+'/'+b]['frames'][frame]={k:row[k] for k in ('used_taps','metrics','delta')}
    report=evaluate_candidates(plan,candidates,provenance=provenance)
    # 人工复核为单独示例，不使无候选达标变成自动建议。
    manual=evaluate_candidates(plan,candidates,provenance=provenance,manual_candidate='shared/31')
    strict=evaluate_candidates(EvaluationPlan(SOURCE,MEMBERS,tuple(range(26)),tuple(range(26,38)),matrix_p95_limit=0,phase_p95_limit_deg=0),candidates,provenance=provenance)
    ota=evaluate_candidates(EvaluationPlan(SOURCE,MEMBERS,tuple(range(26)),tuple(range(26,38)),mode='ota'),candidates,provenance=provenance)
    output=Path(__file__).resolve().parents[1]/'docs/evidence/p2110'
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('evaluation',report),('manual-review',manual),('no-candidate',strict),('ota-unverified',ota)]:
        (output/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(selected=report['selected_candidate'],holdout=report['holdout'],strict_status=strict['status']),indent=2))

if __name__=='__main__': main()
