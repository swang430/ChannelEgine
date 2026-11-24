"""固定成员预算扫描。纯数组离线分析，无设备/HyperRT/工程写回。"""
from __future__ import annotations
import numpy as np
from .channel_comparison import _paths, _real, _positive, frequency_response, compare_responses


def compare_tap_budgets(paths, offsets_hz, port_pairs, *, budgets=(24,26,31),
                        amplitude_floor=1e-8, time_basis=None, delay_zero_basis=None):
    """单帧多端口固定成员扫描。返回 NumPy 频响及 JSON 安全 metrics。

    每策略按未拆簇相对原射线的带内八路误差平方和降序，稳定排序。
    每增加一个预算名额最多拆一个簇；同一共用切点用于全部端口。
    时延取每端口功率重心或所有端口功率重心，复系数始终相干求和。
    """
    floor=_positive(amplitude_floor,'幅度下限')
    f=_real(offsets_hz,'频偏')
    if f.ndim!=1 or not len(f): raise ValueError('需要一维频偏')
    budgets=tuple(budgets)
    if not budgets or any(type(b) is not int or b<1 for b in budgets) or len(set(budgets))!=len(budgets):
        raise ValueError('预算必须为唯一正整数')
    data=[]
    for h,t,g in paths:
        h,t=_paths(h,t); g=np.asarray(g)
        if g.shape!=h.shape or g.dtype.kind not in 'iu' or np.any(g<0) or not len(h):
            raise ValueError('固定成员必须与射线一一对应且为非负整数')
        if np.any(abs(h)==0): raise ValueError('仅接受非零射线；调用者须同步过滤成员')
        data.append((h,t,g))
    if not data: raise ValueError('需要端口射线')
    ids=np.unique(np.concatenate([g for _,_,g in data]))
    base_counts=np.array([len(np.unique(g)) for _,_,g in data])
    # 基准预算按共同成员数定义；允许部分端口缺少某簇。
    if min(budgets)<len(ids): raise ValueError('预算不足以容纳固定成员，不能截断')
    raw=np.stack([frequency_response(h,t,f) for h,t,_ in data])
    base=np.zeros((2,len(data),len(f)),complex)
    changes=[]; scores=[]; counts=[]; cuts=[]
    for cid in ids:
        grouped=[(h[g==cid],t[g==cid]) for h,t,g in data]
        ts=np.concatenate([t for _,t in grouped]); weights=abs(np.concatenate([h for h,_ in grouped]))**2
        if not np.isfinite(weights).all() or not np.isfinite(weights.sum()) or weights.sum()<=0:
            raise ValueError('簇功率无效')
        shared=weights@ts/weights.sum()
        order=np.argsort(ts,kind='stable'); u=(ts[order]-ts[order][0])*1e9; w=weights[order]
        left=np.cumsum(w)[:-1]; right=np.cumsum(w[::-1])[-2::-1]
        cut=None
        if np.any(np.diff(u)>0):
            lm=np.cumsum(w*u)[:-1]/left; rm=np.cumsum((w*u)[::-1])[-2::-1]/right
            gain=left*right/(left+right)*(lm-rm)**2; gain[np.diff(u)<=0]=-1
            cut=float(ts[order[np.argmax(gain)]])
        halves=[] if cut is None else [weights[m]@ts[m]/weights[m].sum() for m in (ts<=cut,ts>cut)]
        def response(h,t,common):
            if not len(h): return np.zeros((2,len(f)),complex)
            w=abs(h)**2
            return h.sum()*np.exp(-2j*np.pi*f*np.array([w@t/w.sum(),common])[:,None])
        score=np.zeros(2); change=np.zeros_like(base); count=np.zeros(len(data),int)
        for i,(h,t) in enumerate(grouped):
            one=response(h,t,shared); base[:,i]+=one
            score+=np.sum(abs(one-frequency_response(h,t,f))**2,axis=1)
            if cut is not None:
                masks=(t<=cut,t>cut)
                change[:,i]=sum(response(h[m],t[m],d) for m,d in zip(masks,halves))-one
                count[i]=int(all(m.any() for m in masks))
        scores.append(score); changes.append(change); counts.append(count); cuts.append(cut)
    strategies={}
    for si,name in enumerate(('independent','shared')):
        order=np.argsort(-np.array(scores)[:,si],kind='stable')
        ranked=[int(c) for c in order if cuts[c] is not None]
        results={}
        for budget in budgets:
            chosen=ranked[:budget-len(ids)]
            model=base[si].copy(); used=base_counts.copy()
            for c in chosen: model+=changes[c][si]; used+=counts[c]
            assert np.all(used<=budget)
            results[str(budget)]=dict(requested_taps=budget,used_taps=used.tolist(),
                split_clusters=[int(ids[c]) for c in chosen],response=model)
        # 一次比较所有预算，共同有效掩码；基线也计入，避免各预算覆盖偏差。
        metrics=compare_responses(raw,{'baseline':base[si],**{k:r['response'] for k,r in results.items()}},
            f,port_pairs,amplitude_floor=floor,time_basis=time_basis,delay_zero_basis=delay_zero_basis)
        baseline=metrics['methods']['baseline']
        for key,r in results.items():
            actual=metrics['methods'][key]
            r['metrics']={**metrics,'methods':{'candidate':actual,'baseline':baseline}}
            r['delta']={}
            for metric in ('matrix_nmse','phase_error_deg'):
                a=np.array(actual[metric],float); b=np.array(baseline[metric],float)
                d=a-b; valid=np.isfinite(d)
                r['delta'][metric]=dict(values=np.where(valid,d,np.nan).tolist(),
                    valid=valid.tolist(),improved=int(np.sum(d < -1e-7)),worse=int(np.sum(d>1e-7)),
                    unchanged=int(np.sum(valid & (abs(d)<=1e-7))))
                # JSON 无效值保留 null。
                r['delta'][metric]['values'] = _nulls(d)
        strategies[name]=dict(ranking=[dict(cluster_id=int(ids[c]),score=float(scores[c][si]),
            last_early_delay_s=cuts[c],extra_taps_per_port=counts[c].tolist()) for c in order],budgets=results)
    return dict(schema='tap-budget-comparison/1',strategies=strategies,reference=raw,
        offsets_hz=f.tolist(),port_pairs=[list(p) for p in port_pairs],
        base_cluster_count=len(ids),alignment_status='unverified',formal_verdict=None,
        time_basis=time_basis,delay_zero_basis=delay_zero_basis)


def _nulls(a):
    if a.ndim>1: return [_nulls(v) for v in a]
    return [float(v) if np.isfinite(v) else None for v in a]
