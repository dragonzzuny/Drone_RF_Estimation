"""Bounded resume of full TF-GridNet TRAIN4; no development-set selection."""
import copy
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tfgridnet_rf_20261011'))
import fit
from model import build,ChunkedLSTM
w,study,worker=fit.w,fit.convergence,fit.worker
PARENT=ROOT/'local/tfgridnet_fit_20261011_v1'
OUT=ROOT/'local/tfgridnet_continuation_20261011_v1'


def state(status,**fields):
    w.write(OUT/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**fields))


def accepted(previous,current):
    checks=[]
    for a,b in zip(previous['by_count'],current['by_count']):
        assert a['count']==b['count']
        checks.append(dict(count=a['count'],nmse=b['mean_nmse']<a['mean_nmse'],
            si=b['mean_si_sdr']>a['mean_si_sdr'],weak=b['weakest_nmse']<=a['weakest_nmse']))
    return all(r['nmse'] and r['si'] and r['weak'] for r in checks),checks


def main():
    OUT.mkdir(exist_ok=False)
    audit=w.read(ROOT/'reports/2026-10-11/TFGRIDNET_FIT_AUDIT.json');assert audit['status']=='PASS'
    speed=w.read(ROOT/'local/tfgridnet_throughput_20261011_v2/COMPLETE.json')
    assert speed['status']=='COMPLETE' and speed['results'][0]['numerical_pass']
    eligible=[r for r in speed['results'] if r['configuration']=='fp32_chunk64' and r.get('qualifies_for_followup')]
    chunk=64 if eligible else 32
    parent=w.read(PARENT/'PROTOCOL.json');fit.verify(PARENT,parent)
    parent_sha=w.digest(PARENT/'LAST.pt');assert parent_sha==speed['checkpoint_sha256']==audit['checkpoint_sha256']
    p=copy.deepcopy(parent)
    sources=p['source_sha256']
    sources[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    for rel,sha in sources.items():
        target=OUT/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target)
    for path in [PARENT/'LAST.pt',PARENT/'PROTOCOL.json',ROOT/'local/tfgridnet_throughput_20261011_v2/COMPLETE.json',
                 ROOT/'reports/2026-10-11/TFGRIDNET_CONTINUATION_PLAN_KO.md']:
        p['pinned_files'][str(path)]=w.digest(path)
    p.update(start_update=32,max_updates=64,maximum_training_seconds=2400,
        observations=[32,40,48,56,64],lstm_chunk=chunk,parent_checkpoint_sha256=parent_sha,
        resume_optimizer=True,resume_rng=True,precision='FP32; TF32 disabled',
        stop_rule='each group: NMSE2/3 down, SI2/3 up, weakestNMSE2/3 not worse versus last accepted point',
        registered_at=time.time())
    w.write(OUT/'PROTOCOL.json',p);ph=w.digest(OUT/'PROTOCOL.json');fit.verify(OUT,p)
    state('WAITING_GPU_LOCK',updates=32,chunk=chunk)
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert torch.cuda.is_available()
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=worker.NativeMixtures(p['preparation'],'train_pack',1)
        items=[worker.fit.base.batch([data[i]]) for i in p['train_indices']]
        saved=torch.load(PARENT/'LAST.pt',map_location='cpu',weights_only=False)
        assert saved['updates']==32
        net=build().cuda();net.load_state_dict(saved['model'])
        for module in net.modules():
            if isinstance(module,ChunkedLSTM):module.chunk=chunk
        assert sum(t.numel() for t in net.parameters())==p['parameters']
        opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],foreach=False)
        opt.load_state_dict(saved['optimizer']);assert {int(s['step']) for s in opt.state.values()}=={32}
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
        began=time.time();torch.cuda.reset_peak_memory_stats()
        initial=dict(step=32,**study.evaluate(net,items,data,p['train_indices']),seconds=time.time()-began,
            checkpoint_sha256=parent_sha,checkpoint=str(PARENT/'LAST.pt'))
        prior=w.read(PARENT/'COMPLETE.json')['final']
        for a,b in zip(initial['rows'],prior['rows']):
            np.testing.assert_allclose(a['nmse'],b['nmse'],rtol=1e-5,atol=1e-5)
            np.testing.assert_allclose(a['si_sdr'],b['si_sdr'],rtol=1e-5,atol=1e-3)
        history=[initial];best=initial;reason='MAX64_REVIEW'
        w.write(OUT/'STEP_032.json',initial)
        w.write(OUT/'HISTORY.json',dict(history=history,protocol_sha256=ph))
        w.write(OUT/'SELECTED.json',best)
        for step in range(33,65):
            net.train();opt.zero_grad(set_to_none=True);total=0.
            for case,item in enumerate(items):
                state('TRAINING',updates=step-1,working_update=step,case=case+1,total=64,
                    selected_update=best['step'],seconds=time.time()-began,chunk=chunk)
                out,logits=worker.predict(net,item)
                loss=study.original.prior.pit_waveform_loss(out,item['references'],item['active'],item['mixture'])['loss']
                loss=loss+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
                assert torch.isfinite(loss);(loss/4).backward();total+=float(loss.detach())/4
            norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True));opt.step()
            reached=time.time()-began>=p['maximum_training_seconds']
            if step not in p['observations'] and not reached:continue
            state('EVALUATING_TRAIN4',updates=step,total=64,selected_update=best['step'])
            point=dict(step=step,**study.evaluate(net,items,data,p['train_indices']),seconds=time.time()-began,
                training_loss=total,gradient_norm_before_clip=norm)
            assert {int(s['step']) for s in opt.state.values()}=={step}
            checkpoint=dict(model=net.state_dict(),optimizer=opt.state_dict(),updates=step,protocol_sha256=ph,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            worker.atomic_torch(OUT/'LAST.pt',checkpoint)
            point['checkpoint_sha256']=w.digest(OUT/'LAST.pt');point['checkpoint']=str(OUT/'LAST.pt')
            passed,checks=accepted(best,point);point['continuation_pass']=passed;point['checks']=checks
            if passed:
                worker.atomic_torch(OUT/'SELECTED.pt',checkpoint)
                best=copy.deepcopy(point);best['checkpoint']=str(OUT/'SELECTED.pt')
                best['checkpoint_sha256']=w.digest(OUT/'SELECTED.pt');w.write(OUT/'SELECTED.json',best)
            history.append(point);w.write(OUT/f'STEP_{step:03d}.json',point)
            report=dict(status='TRAIN_DIAGNOSIS',updates=step,history=history,selected=best,
                protocol_sha256=ph,validation_read=False,heldout_read=False,incumbent_replaced=False)
            w.write(OUT/'HISTORY.json',report);w.write(ROOT/'reports/2026-10-11/TFGRIDNET_CONTINUATION_PROGRESS.json',report)
            print(dict(step=step,by_count=point['by_count'],continuation_pass=passed),flush=True)
            if point['all_sources_below_005']:reason='FIT_TARGET';break
            if not passed:reason='NO_JOINT_IMPROVEMENT_REVIEW';break
            if reached:reason='TIME_BUDGET_REVIEW';break
        fit.verify(OUT,p)
        result=dict(status='COMPLETE',updates=step,reason=reason,history=history,final=point,selected=best,
            protocol_sha256=ph,parameters=p['parameters'],chunk=chunk,seconds=time.time()-began,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),validation_read=False,heldout_read=False,incumbent_replaced=False)
        w.write(OUT/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-11/TFGRIDNET_CONTINUATION_RESULT.json',result)
        state('COMPLETED',updates=step,selected_update=best['step'],reason=reason)


if __name__=='__main__':
    try:main()
    except Exception:
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
