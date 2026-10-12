"""Post-training positional-encoding ablation; NOT a retrained comparator."""
from pathlib import Path
import os
import time
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha
from tracking_cache import feature_input
from spectral_routing import PERMS
from spectral_routing_constrained import fit
from routing_statistics import score
from diagnose_count_routing import summary,verify_metrics

RUN=ROOT/'local/learned_tracking_20261013_v1'
DEV=ROOT/'local/learned_tracking_preparation_20261013_v1'
OUT=ROOT/'local/tracking_position_diagnosis_20261013_v1'


def main():
    OUT.mkdir(exist_ok=True)
    inputs={str(p):sha(p) for p in (RUN/'ACTUAL_050.pt',RUN/'RESULT.json',DEV/'RESULT.json')}
    sources=dict(read(RUN/'PROTOCOL.json')['source_sha256'])
    for p in (Path(__file__),ROOT/'reports/2026-10-13/TRACKING_POSITION_PLAN_KO.md'):
        sources[str(p.relative_to(ROOT))]=sha(p)
    assert not (OUT/'PLAN.json').exists()
    write(OUT/'PLAN.json',dict(registered=time.time(),input_sha256=inputs,source_sha256=sources,
        conditions=['original','time_gain_zero'],epoch=50,optimizer_updates=0,cases=48,
        post_training_intervention=True,not_a_retrained_comparison=True))
    import torch
    from learned_tracking_model import TrackingEmbedding
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    saved=torch.load(RUN/'ACTUAL_050.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['epoch']==50 and saved['updates']==600
    original=TrackingEmbedding().eval().requires_grad_(False)
    changed=TrackingEmbedding().eval().requires_grad_(False)
    original.load_state_dict(saved['model']);changed.load_state_dict(saved['model'])
    gain=float(original.time_gain)
    with torch.no_grad():changed.time_gain.zero_()
    assert all(torch.equal(v,changed.state_dict()[n]) for n,v in original.state_dict().items() if n!='time_gain')
    del saved
    groups={}
    for group,key in [('B24','B'),('C24','C')]:
        prior={r['index']:r for r in read(RUN/'RESULT.json')[key]['rows']};rows=[]
        for j,item in enumerate(read(DEV/'RESULT.json')['groups'][group]):
            assert not item['training'] and 'labels' not in item
            write(OUT/'STATE.json',dict(status='RUNNING',group=group,case=j+1,pid=os.getpid(),time=time.time()))
            observed,powers,bounds=feature_input(item)
            assert sha(item['scoring'])==item['scoring_sha256']
            with np.load(item['scoring'],allow_pickle=False) as z:stats={k:z[k].copy() for k in z.files}
            metrics={};orders={}
            for name,net in [('original',original),('time_gain_zero',changed)]:
                with torch.no_grad():v=net(torch.from_numpy(observed)[None])[0].numpy().astype(np.float64)
                _,choices,receipt=fit(np.pad(v,((0,0),(0,0),(0,384))),powers,item['predicted_count'])
                order=np.array([PERMS[int(q)] for q in choices]);orders[name]=order
                metrics[name]=score(stats,order,item['count'],item['predicted_count'])
            verify_metrics(metrics['original'],prior[item['index']])
            rows.append(dict(index=item['index'],count=item['count'],
                changed_block_permutations=int(np.any(orders['original']!=orders['time_gain_zero'],axis=1).sum()),**metrics))
        groups[group]=dict(cases=rows,by_condition={name:{str(k):summary([r[name] for r in rows if r['count']==k])
            for k in (1,2,3)} for name in ('original','time_gain_zero')})
    assert all(sha(p)==h for p,h in inputs.items()) and all(sha(ROOT/p)==h for p,h in sources.items())
    result=dict(status='PASS',groups=groups,original_time_gain=gain,optimizer_updates=0,
        checkpoint_unchanged=True,post_training_intervention=True,plan_sha256=sha(OUT/'PLAN.json'),
        limitation='Global self-attention still sees all33blocks; zero position does not remove global context and is not retraining')
    write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json')))
    write(ROOT/'reports/2026-10-13/TRACKING_POSITION_RESULT.json',result)
    write(OUT/'STATE.json',dict(status='COMPLETE_AUDITED',pid=os.getpid(),time=time.time()))
    print({g:v['by_condition'] for g,v in groups.items()},flush=True)


if __name__=='__main__':main()
