"""One fixed, parameter-free phase canonicalization candidate on native DEV630.

Adaptive development evaluation motivated by prior four-phase averaging. The
anchor rule is fixed before this evaluation, with no validation phase search.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/source_interaction_20261010'))
import train_comparison as worker
from canonical import CanonicalPhaseSeparator, phase_anchor

w=worker.watch
CHECK=ROOT/'reports/2026-10-10/CANONICAL_PHASE_CPU_CHECK.json'


def register(root, study, predecessor):
    if (root/'PROTOCOL.json').exists():raise ValueError('Duplicate phase evaluation')
    base=w.read(study/'PROTOCOL.json');checks=w.read(CHECK)
    assert checks['status']=='PASS'
    sources=dict(base['source_sha256']);sources.update(checks['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    p=dict(status='REGISTERED_ADAPTIVE_CANONICAL_PHASE_EVALUATION',source_sha256=sources,
        parent_checkpoint=base['parent_checkpoint'],parent_checkpoint_sha256=base['parent_checkpoint_sha256'],
        preparation=base['preparation'],preparation_sha256=base['preparation_sha256'],
        baseline=base['validation_identity_template'],baseline_sha256=base['validation_identity_sha256'],
        predecessor=str(predecessor),predecessor_complete_sha256=w.digest(predecessor/'COMPLETE.json'),
        gpu_display_policy=base['gpu_display_policy'],cpu_check_sha256=w.digest(CHECK),
        parameters=32142859,added_parameters=0,updates=0,validation_cases=630,inferences_per_case=1,
        anchor='First maximum-magnitude complex coefficient of the local mixture STFT512/hop128',
        inputs='Mixed IQ STFT, unchanged mixture power context and crop only; no references/count/categories',
        operation='Divide input STFT by anchor unit phase; run same frozen U-Net once; multiply all four output STFTs by that phase',
        baseline_reuse='Hash-checked retained-parent baseline, previously reproduced in both arms of the audited completed main study',
        acceptance='NMSE2/3 lower and complex SI-SDR2/3 higher than the same single-pass parent; weakest NMSE nonincrease in both counts',
        initialization='Same exact retained native parent e2; no training or phase-based checkpoint selection',
        heldout_read=False,adaptive_development_evaluation=True,independent_test=False,
        limitation='Repeated DEV630 from five recording groups; single three-category composition; anchor can be discontinuous near tied magnitudes',
        registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dst=root/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
    w.write(root/'PROTOCOL.json',p);return p


def verify(root,p):
    for rel,sha in p['source_sha256'].items():
        assert w.digest(ROOT/rel)==sha and w.digest(root/'source_snapshot'/rel)==sha
    for path,sha in [(Path(p['parent_checkpoint']),p['parent_checkpoint_sha256']),
        (Path(p['preparation'])/'PREPARATION.json',p['preparation_sha256']),
        (Path(p['baseline']),p['baseline_sha256']),(CHECK,p['cpu_check_sha256']),
        (Path(p['predecessor'])/'COMPLETE.json',p['predecessor_complete_sha256'])]:
        assert w.digest(path)==sha


class ObservedCanonical(CanonicalPhaseSeparator):
    def __init__(self,base,root):
        super().__init__(base);self.root=root;self.anchors=[]

    def forward(self,z,context,crop):
        top=z.flatten(1).abs().topk(2).values
        phase,valid,indices=phase_anchor(z)
        for index,pair in zip(indices[:,0].tolist(),top.tolist()):
            self.anchors.append(dict(index=index,relative_peak_gap=(pair[0]-pair[1])/pair[0] if pair[0] else 0.))
        if len(self.anchors)%50==0:
            w.write(self.root/'STATE.json',dict(status='GPU_EVALUATING',cases=len(self.anchors),total=630,pid=os.getpid(),time=time.time()))
        return super().forward(z,context,crop)


def run(root,p,public):
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p)
        assert torch.cuda.is_available()
        w.write(root/'GPU_START_CHECK.json',worker.guard.wait_for_predecessor(w.read(Path(p['predecessor'])/'STATE.json')['pid'],p['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        net=ObservedCanonical(worker.make_model('retained_unet',Path(p['parent_checkpoint'])),root).cuda().eval()
        assert sum(q.numel() for q in net.parameters())==p['parameters']
        baseline,ids=w.validate(Path(p['baseline']),None)
        validation=worker.NativeMixtures(p['preparation'],'validation_pack',1)
        w.write(root/'STATE.json',dict(status='GPU_EVALUATING',cases=0,total=630,pid=os.getpid(),time=time.time()))
        started=time.time()
        candidate=worker.fit.base.validate(net,validation,root/'CANONICAL.json',0)
        metrics,_=w.validate(root/'CANONICAL.json',ids)
        assert len(net.anchors)==630 and len(validation)==630
        old_rows=w.read(Path(p['baseline']))['rows'];new_rows=w.read(root/'CANONICAL.json')['rows']
        assert all(a['predicted_count']==b['predicted_count'] for a,b in zip(old_rows,new_rows))
        deltas=[]
        for count in (1,2,3):
            a,b=[next(g for g in r['by_count'] if g['count']==count) for r in (baseline,metrics)]
            deltas.append(dict(count=count,nmse_delta=b['mean_nmse']-a['mean_nmse'],si_sdr_delta=b['mean_si_sdr']-a['mean_si_sdr'],weakest_nmse_delta=b['weakest_nmse']-a['weakest_nmse']))
        passed=all(v['nmse_delta']<0 and v['si_sdr_delta']>0 and v['weakest_nmse_delta']<=0 for v in deltas[1:])
        verify(root,p)
        result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),
            baseline=baseline,candidate=metrics,comparison=deltas,criterion_met=passed,
            anchor_min_relative_peak_gap=min(v['relative_peak_gap'] for v in net.anchors),
            exactly_tied_anchors=sum(v['relative_peak_gap']==0 for v in net.anchors),
            count_predictions_unchanged=True,parameters=p['parameters'],updates=0,inferences_per_case=1,
            seconds=time.time()-started,heldout_read=False,independent_test=False,
            all_630_rows_reaggregated=True,all_sources_and_parent_hash_checked=True,
            candidate_sha256=w.digest(root/'CANONICAL.json'),time=time.time())
        w.write(root/'ANCHORS.json',net.anchors);w.write(root/'COMPLETE.json',result);w.write(public,result)
        w.write(root/'STATE.json',dict(status='COMPLETE',cases=630,pid=os.getpid(),time=time.time()))
        print(dict(status='COMPLETE',criterion_met=passed,comparison=deltas),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('study','predecessor','run','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    try:
        with (root/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            p=register(root,a.study.resolve(),a.predecessor.resolve());run(root,p,a.public.resolve())
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
