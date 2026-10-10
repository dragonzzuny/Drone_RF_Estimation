"""Bounded full-model continuation diagnosis on four fixed TRAIN mixtures."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/source_interaction_20261010'))
import balanced_head_probe as original
worker,w=original.worker,original.w
from drone_rf.waveform import waveform_metrics
OLD=ROOT/'local/balanced_head_probe_20261010_v1'
PUBLIC=ROOT/'reports/2026-10-11'


def state(root,status,**values):
    w.write(root/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**values))


def register(root):
    assert not (root/'PROTOCOL.json').exists()
    old=w.read(OLD/'PROTOCOL.json')
    sources=dict(old['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    for relative,digest in sources.items():
        assert w.digest(ROOT/relative)==digest,relative
        dest=root/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,dest)
    paths=[OLD/'PROTOCOL.json',OLD/'balanced_COMPLETE.json',Path(old['parent_checkpoint']),
        Path(old['preparation'])/'PREPARATION.json',Path(old['preparation'])/'NATIVE_MANIFEST.json',
        PUBLIC/'CONVERGENCE_PLAN_KO.md']
    paths+=list((Path(old['preparation'])/'features').glob('train_pack_001.*'))
    p=dict(source_sha256=sources,pinned_files={str(p):w.digest(p) for p in paths},
        preparation=old['preparation'],parent_checkpoint=old['parent_checkpoint'],
        train_indices=old['train_indices'],old_lr=1e-5,new_lr=1e-3,max_updates=256,
        effective_batch=4,microbatch=1,seed=0,parameters=32180747,
        observations=sorted(set([0,1,8,16]+list(range(32,257,32)))),
        loss='unchanged PIT NMSE+coherence+inactive/background plus0.1countCE',
        purpose='fixed TRAIN fitting diagnosis; not DEV performance or architecture superiority',
        early_target='all10 active source NMSE<=0.05; diagnostic budget rule',
        checkpoint='latest diagnostic state only; never replace incumbent',
        historical_control_retrained=False,validation_iq_read=False,heldout_read=False,registered_at=time.time())
    w.write(root/'PROTOCOL.json',p);return p


def verify(root,p):
    for relative,digest in p['source_sha256'].items():
        assert w.digest(ROOT/relative)==w.digest(root/'source_snapshot'/relative)==digest,relative
    for path,digest in p['pinned_files'].items():assert w.digest(Path(path))==digest,path


@torch.no_grad()
def evaluate(net,items,data,indices):
    net.eval();rows=[]
    for index,item in zip(indices,items):
        outputs,logits=worker.predict(net,item)
        scores=waveform_metrics(outputs,item['references'],item['active'],item['mixture'])
        active=item['active'][0];count=int(item['construction_count'])
        powers=scores['reference_power'][0][active].tolist()
        nmse=scores['nmse'][0][active].tolist();si=scores['si_sdr'][0][active].tolist()
        assert np.isfinite(nmse+si).all() and float(scores['sum_relative_error'][0])<1e-9
        source=data.rows[index];clips=[data.library.clips[int(i)] for i in source['indices'][:count]]
        rows.append(dict(index=index,count=count,categories=[c['category'] for c in clips],
            pack_ids=[c['pack_id'] for c in clips],reference_power=powers,nmse=nmse,si_sdr=si,
            weakest_index=int(np.argmin(powers)),assignment=scores['assignment'][0].tolist(),
            sum_relative_error=float(scores['sum_relative_error'][0]),predicted_count=int(logits.argmax(-1)[0])+1))
    grouped=[]
    for count in (2,3):
        group=[r for r in rows if r['count']==count]
        grouped.append(dict(count=count,cases=len(group),
            mean_nmse=float(np.mean([v for r in group for v in r['nmse']])),
            mean_si_sdr=float(np.mean([v for r in group for v in r['si_sdr']])),
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in group])),
            max_source_nmse=max(v for r in group for v in r['nmse'])))
    return dict(rows=rows,by_count=grouped,all_sources_below_005=all(v<=.05 for r in rows for v in r['nmse']))


def run(root):
    root.mkdir(parents=True,exist_ok=True);p=register(root);verify(root,p)
    ph=w.digest(root/'PROTOCOL.json');state(root,'WAITING_GPU_LOCK')
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert torch.cuda.is_available()
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=worker.NativeMixtures(p['preparation'],'train_pack',1)
        items=[worker.fit.base.batch([data[i]]) for i in p['train_indices']]
        net=original.make_model('balanced',Path(p['parent_checkpoint'])).cuda().eval()
        assert sum(t.numel() for t in net.parameters())==p['parameters']
        assert all(t.requires_grad for t in net.parameters())
        torch.manual_seed(0);opt=original.prior.optimizer(net,p['new_lr'])
        history=[];began=time.time();norms=[]
        for step in range(p['max_updates']+1):
            if step:
                net.train();opt.zero_grad(set_to_none=True);total=0.
                for item in items:
                    out,logits=worker.predict(net,item)
                    loss=original.prior.pit_waveform_loss(out,item['references'],item['active'],item['mixture'])['loss']
                    loss=loss+.1*F.cross_entropy(logits,item['construction_count']-1)
                    assert torch.isfinite(loss);(loss/4).backward();total+=float(loss.detach())/4
                norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
                opt.step();norms.append(norm)
                if step%4==0:state(root,'TRAINING',updates=step,total=256,training_loss=total,seconds=time.time()-began)
            if step not in p['observations']:continue
            result=dict(step=step,**evaluate(net,items,data,p['train_indices']),seconds=time.time()-began)
            history.append(result)
            if step in (0,64):
                old=next(h for h in w.read(OLD/'balanced_COMPLETE.json')['history'] if h['step']==step)
                delta={key:max(abs(a[key]-b[key]) for a,b in zip(result['by_count'],old['by_count']))
                    for key in ('mean_nmse','mean_si_sdr')}
                result['historical_reproduction_max_abs']=delta
                if step==0:assert max(delta.values())<1e-5,delta
            if step:
                assert {int(s['step']) for s in opt.state.values()}=={step}
                worker.atomic_torch(root/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),updates=step,
                    protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
                result['checkpoint_sha256']=w.digest(root/'LAST.pt')
            w.write(root/f'STEP_{step:03d}.json',result)
            report=dict(status='TRAIN_DIAGNOSIS',updates=step,history=history,protocol_sha256=ph,
                source_hash=data.rows_hash,validation_read=False,heldout_read=False)
            w.write(root/'HISTORY.json',report);w.write(PUBLIC/'CONVERGENCE_PROGRESS.json',report)
            print(dict(step=step,by_count=result['by_count']),flush=True)
            if result['all_sources_below_005']:break
        verify(root,p)
        receipt=dict(status='COMPLETED',updates=step,final=result,history=history,
            reason='DIAGNOSTIC_FIT_TARGET' if result['all_sources_below_005'] else 'MAX256_REVIEW',
            protocol_sha256=ph,history_sha256=w.digest(root/'HISTORY.json'),
            checkpoint_sha256=w.digest(root/'LAST.pt'),parameters=p['parameters'],
            original_control_retrained=False,validation_read=False,heldout_read=False,
            seconds=time.time()-began,time=time.time())
        w.write(root/'COMPLETE.json',receipt);w.write(PUBLIC/'CONVERGENCE_RESULT.json',receipt)
        state(root,'COMPLETED',updates=step,reason=receipt['reason'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve()
    try:run(root)
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        state(root,'FAILED');raise
