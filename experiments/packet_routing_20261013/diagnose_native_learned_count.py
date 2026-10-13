"""NONDEPLOYABLE count-oracle diagnostic of fixed learned50 on exposed C24."""
import os
from pathlib import Path
import time
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha
from tracking_cache import feature_input
from spectral_routing import PERMS
from spectral_routing_constrained import fit
from routing_statistics import score
from diagnose_count_routing import summary,verify_metrics

OUT=ROOT/'local/native_count_diagnosis_20261013_v1'
RUN=ROOT/'local/native_tracking_20261013_v1'
DEV=ROOT/'local/native_tracking_cache_20261013_v1'


def main():
    OUT.mkdir(exist_ok=True)
    protocol=read(RUN/'PROTOCOL.json');sources=dict(protocol['source_sha256'])
    for p in (Path(__file__),ROOT/'reports/2026-10-13/NATIVE_COUNT_DIAGNOSIS_PLAN_KO.md'):
        sources[str(p.relative_to(ROOT))]=sha(p)
    files={str(p):sha(p) for p in (RUN/'ACTUAL_050.pt',RUN/'C_050.json',RUN/'PROTOCOL.json',DEV/'RESULT.json')}
    assert not (OUT/'PLAN.json').exists()
    write(OUT/'PLAN.json',dict(registered=time.time(),source_sha256=sources,input_sha256=files,
        cases=24,optimizer_updates=0,epoch_fixed=50,deployable=False,true_count_diagnostic=True))
    assert all(sha(ROOT/p)==h for p,h in sources.items())
    import torch
    from learned_tracking_model import TrackingEmbedding
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    net=TrackingEmbedding().eval().requires_grad_(False)
    saved=torch.load(RUN/'ACTUAL_050.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['epoch']==50 and saved['updates']==600
    net.load_state_dict(saved['model']);del saved
    prior={r['index']:r for r in read(RUN/'C_050.json')['rows']};rows=[]
    for j,item in enumerate(read(DEV/'RESULT.json')['groups']['C24']):
        assert not item['training']
        write(OUT/'STATE.json',dict(status='RUNNING',case=j+1,pid=os.getpid(),time=time.time()))
        observed,powers,bounds=feature_input(item)
        with torch.no_grad():emb=net(torch.from_numpy(observed)[None])[0].numpy().astype(np.float64)
        emb=np.pad(emb,((0,0),(0,0),(0,384)))
        assert sha(item['scoring'])==item['scoring_sha256']
        with np.load(item['scoring'],allow_pickle=False) as z:stats={k:z[k].copy() for k in z.files}
        measures={}
        for name,count in [('predicted',item['predicted_count']),('oracle_count',item['count'])]:
            _,choices,receipt=fit(emb,powers,count)
            orders=np.array([PERMS[int(q)] for q in choices])
            measures[name]=score(stats,orders,item['count'],item['predicted_count'])
        verify_metrics(measures['predicted'],prior[item['index']])
        if item['count']==item['predicted_count']:assert measures['predicted']==measures['oracle_count']
        rows.append(dict(index=item['index'],count=item['count'],predicted_count=item['predicted_count'],
            count_correct=item['count']==item['predicted_count'],**measures))
    groups={}
    for name in ('predicted','oracle_count'):
        groups[name]={str(k):summary([r[name] for r in rows if r['count']==k]) for k in (1,2,3)}
        for label,correct in [('correct',True),('incorrect',False)]:
            groups[name][label]=summary([r[name] for r in rows if r['count_correct']==correct])
    assert all(sha(ROOT/p)==h for p,h in sources.items()) and all(sha(p)==h for p,h in files.items())
    result=dict(status='PASS',cases=rows,groups=groups,optimizer_updates=0,heldout_access=False,
        deployable=False,true_count_used_only_for_diagnostic_routing=True,plan_sha256=sha(OUT/'PLAN.json'))
    write(OUT/'RESULT.json',result)
    write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),predicted_metrics_reproduced=True))
    write(ROOT/'reports/2026-10-13/NATIVE_COUNT_DIAGNOSIS_RESULT.json',result)
    write(OUT/'STATE.json',dict(status='COMPLETE_AUDITED',pid=os.getpid(),time=time.time()))
    print(groups,flush=True)


if __name__=='__main__':main()
