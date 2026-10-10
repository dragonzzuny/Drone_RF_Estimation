"""Prospective full TF-GridNet RF TRAIN4 fitting/throughput diagnosis."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import torch
from model import build,ROOT
sys.path.insert(0,str(ROOT/'experiments/convergence_20261011'))
import run as convergence
worker,w=convergence.worker,convergence.w


def state(root,status,**values):
    w.write(root/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**values))


def verify(root,p):
    for relative,sha in p['source_sha256'].items():
        assert w.digest(ROOT/relative)==w.digest(root/'source_snapshot'/relative)==sha,relative
    for path,sha in p['pinned_files'].items():assert w.digest(Path(path))==sha,path


def run(root):
    root.mkdir(exist_ok=False)
    preflight=ROOT/'reports/2026-10-11/TFGRIDNET_GPU_PREFLIGHT.json'
    check=w.read(preflight);assert check['status']=='PASS'
    assert w.read(ROOT/'reports/2026-10-11/CONVERGENCE_AUDIT.json')['status']=='PASS'
    old=w.read(ROOT/'local/convergence_20261011_v1/PROTOCOL.json')
    sources=dict(old['source_sha256'])
    for path in Path(__file__).parent.rglob('*'):
        if path.is_file() and '__pycache__' not in str(path):sources[str(path.relative_to(ROOT))]=w.digest(path)
    for relative,sha in sources.items():
        assert w.digest(ROOT/relative)==sha
        dest=root/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
    pins=dict(old['pinned_files'])
    for path in (preflight,ROOT/'reports/2026-10-11/TFGRIDNET_FIT_PLAN_KO.md'):
        pins[str(path)]=w.digest(path)
    p=dict(preparation=old['preparation'],train_indices=old['train_indices'],source_sha256=sources,pinned_files=pins,
        parameters=check['parameters'],max_updates=32,maximum_training_seconds=2700,seed=0,
        effective_batch=4,microbatch=1,learning_rate=5e-4,weight_decay=1e-4,clip=1.,
        observations=[0,1,4,8,16,24,32],crop_samples=63872,two_sided_bins=512,
        blocks=6,embedding=128,bidirectional_lstm_hidden=192,attention_heads=4,
        loss='unchanged PIT NMSE+coherence+inactive/background +0.1 construction count CE',
        success='NMSE2/3 decrease from own initialization; report SI-SDR and weak/all source errors; no model superiority claim',
        incumbent_replaced=False,validation_read=False,heldout_read=False,
        control='reuse historical fresh U-Net fit receipts; different parameter count and initialization disclosed; no new control',
        registered_at=time.time())
    w.write(root/'PROTOCOL.json',p);ph=w.digest(root/'PROTOCOL.json');verify(root,p)
    state(root,'WAITING_GPU_LOCK')
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert torch.cuda.is_available()
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=worker.NativeMixtures(p['preparation'],'train_pack',1)
        items=[worker.fit.base.batch([data[i]]) for i in p['train_indices']]
        net=build().cuda();assert sum(x.numel() for x in net.parameters())==p['parameters']
        opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],foreach=False)
        began=time.time();history=[];torch.cuda.reset_peak_memory_stats()
        for step in range(p['max_updates']+1):
            if step:
                net.train();opt.zero_grad(set_to_none=True);total=0.
                for case,item in enumerate(items):
                    state(root,'TRAINING',updates=step-1,working_update=step,case=case+1,total=32,seconds=time.time()-began)
                    output,logits=worker.predict(net,item)
                    loss=convergence.original.prior.pit_waveform_loss(output,item['references'],item['active'],item['mixture'])['loss']
                    loss=loss+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
                    assert torch.isfinite(loss);(loss/4).backward();total+=float(loss.detach())/4
                norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True));opt.step()
            reached_time=time.time()-began>=p['maximum_training_seconds']
            if step not in p['observations'] and not reached_time:continue
            state(root,'EVALUATING_TRAIN4',updates=step,total=32,seconds=time.time()-began)
            point=dict(step=step,**convergence.evaluate(net,items,data,p['train_indices']),seconds=time.time()-began)
            if step:
                assert {int(s['step']) for s in opt.state.values()}=={step}
                worker.atomic_torch(root/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),updates=step,
                    protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
                point['checkpoint_sha256']=w.digest(root/'LAST.pt');point['training_loss']=total
                point['gradient_norm_before_clip']=norm
            history.append(point);w.write(root/f'STEP_{step:03d}.json',point)
            report=dict(status='TRAIN_DIAGNOSIS',updates=step,history=history,protocol_sha256=ph,
                validation_read=False,heldout_read=False,incumbent_replaced=False)
            w.write(root/'HISTORY.json',report);w.write(ROOT/'reports/2026-10-11/TFGRIDNET_FIT_PROGRESS.json',report)
            print(dict(step=step,by_count=point['by_count']),flush=True)
            if reached_time or point['all_sources_below_005']:break
        verify(root,p)
        reason='FIT_TARGET' if point['all_sources_below_005'] else ('TIME_BUDGET_REVIEW' if reached_time else 'MAX32_REVIEW')
        result=dict(status='COMPLETE',updates=step,reason=reason,final=point,history=history,
            parameters=p['parameters'],seconds=time.time()-began,peak_memory_bytes=torch.cuda.max_memory_allocated(),
            protocol_sha256=ph,heldout_read=False,validation_read=False,incumbent_replaced=False,time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-11/TFGRIDNET_FIT_RESULT.json',result)
        state(root,'COMPLETED',updates=step,reason=reason)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    try:run(args.run.resolve())
    except Exception:
        w.write(args.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        state(args.run,'FAILED');raise
