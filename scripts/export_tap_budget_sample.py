"""只读实样回归导出；外部解析器仅为本机验收工具。"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mimo_ota_simulator.tap_budget import compare_tap_budgets

def main():
    sys.path.insert(0,'/private/tmp/p2109-vrt/scripts')
    from mpdb_port_evidence import read_downlink
    root=Path('/Users/Simon/Tools/MIMO-First')
    archive=next((root/'RT-IMT-2030').glob('*MPDB.zip'))
    members_path=root/'docs/evidence/imt-three/members.npz'
    digest,ports,rays=read_downlink(archive)
    output=Path(__file__).resolve().parents[1]/'docs/evidence/p2109'
    output.mkdir(parents=True,exist_ok=True)
    pairs=[(a,b) for a in range(8) for b in range(a+1,8) if a//2==b//2 or a%2==b%2]
    offsets=np.arange(-20,21)*2.5e6
    summary={}; phase={}; matrix={}; usage={}
    with np.load(members_path,allow_pickle=False) as members,(output/'phase-points.csv').open('w') as ph,(output/'matrix-points.csv').open('w') as ma,(output/'ranking.csv').open('w') as ra,(output/'used-taps.csv').open('w') as us:
        if str(members['source_sha256'])!=digest: raise ValueError('成员来源不匹配')
        pw,mw,rw,uw=map(csv.writer,(ph,ma,ra,us))
        pw.writerow(['frame','strategy','budget','port_a','port_b','offset_hz','error_deg','baseline_error_deg','delta_deg','change','valid'])
        mw.writerow(['frame','strategy','budget','offset_hz','nmse','baseline_nmse','delta','change','valid'])
        rw.writerow(['frame','strategy','rank','cluster_id','score','last_early_delay_s','extra_taps_per_port'])
        uw.writerow(['frame','strategy','budget','port','used_taps','source_last_frame_without_asc'])
        for frame in range(38):
            paths=[]
            for link in range(8):
                lid,=ports['link_id'][(ports['frame_id']==frame)&(ports['tx_ant']==link//2)&(ports['rx_ant']==link%2)]
                m=(rays['link_id']==lid)&(abs(rays['h'])>0)
                paths.append((rays['h'][m],rays['delay'][m],members[f'L{lid}']))
            result=compare_tap_budgets(paths,offsets,pairs,time_basis='MPDB source frame id; frame 37 has no common ASC reference',delay_zero_basis='input seconds unchanged; physical zero unverified')
            for name,s in result['strategies'].items():
                for rank,row in enumerate(s['ranking'],1): rw.writerow([frame,name,rank,row['cluster_id'],row['score'],row['last_early_delay_s'],json.dumps(row['extra_taps_per_port'])])
                for budget,r in s['budgets'].items():
                    key=name+'/'+budget
                    phase.setdefault(key,[]); matrix.setdefault(key,[]); usage.setdefault(key,[])
                    for port,n in enumerate(r['used_taps']): uw.writerow([frame,name,budget,port,n,frame==37]); usage[key].append(n)
                    metric=r['metrics']['methods']; delta=r['delta']
                    def change(d): return 'unverified' if d is None else 'improved' if d < -1e-7 else 'worse' if d>1e-7 else 'unchanged'
                    for pi,(a,b) in enumerate(pairs):
                        for fi,f in enumerate(offsets):
                            e=metric['candidate']['phase_error_deg'][pi][fi]; old=metric['baseline']['phase_error_deg'][pi][fi]; d=delta['phase_error_deg']['values'][pi][fi]
                            pw.writerow([frame,name,budget,a,b,f,e,old,d,change(d),d is not None]); phase[key].append(e)
                    for fi,f in enumerate(offsets):
                        e=metric['candidate']['matrix_nmse'][fi]; old=metric['baseline']['matrix_nmse'][fi]; d=delta['matrix_nmse']['values'][fi]
                        mw.writerow([frame,name,budget,f,e,old,d,change(d),d is not None]); matrix[key].append(e)
                    if key not in summary: summary[key]={metric:{k:0 for k in ('improved','worse','unchanged')} for metric in delta}
                    for metric,v in delta.items():
                        for k in ('improved','worse','unchanged'): summary[key][metric][k]+=v[k]
    for key in summary:
        for label,values in [('phase_error_deg',phase[key]),('matrix_nmse',matrix[key])]:
            finite=[v for v in values if v is not None]
            summary[key][label].update(valid=len(finite),total=len(values),p50=float(np.percentile(finite,50)),p95=float(np.percentile(finite,95)))
        summary[key]['actual_taps_min']=min(usage[key]);summary[key]['actual_taps_max']=max(usage[key])
    report=dict(schema='tap-budget-sample/1',source_sha256=digest,members_sha256=hashlib.sha256(members_path.read_bytes()).hexdigest(),frames=list(range(38)),asc_unmatched_frames=[37],offsets_hz=offsets.tolist(),amplitude_floor=1e-8,delta_tolerance=1e-7,formal_verdict=None,alignment_status='unverified',hyperrt_used=False,device_used=False,summary=summary)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
