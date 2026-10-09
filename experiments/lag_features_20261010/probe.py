"""Full-model TRAIN4 test: lagged power versus power plus complex relation."""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/source_interaction_20261010'))
import head_learning_rate_probe as previous
from lag_input import augment

worker,w=previous.worker,previous.w
ARMS=('lag_power','lag_power_phase')
CHECK=ROOT/'reports/2026-10-10/LAG_FEATURE_CPU_CHECK.json'
AUDIT=ROOT/'reports/2026-10-10/CANONICAL_PHASE_TRAIN_AUDIT.json'


def register(root,study,dependency):
    if (root/'PROTOCOL.json').exists():raise ValueError('Duplicate lag probe')
    base=w.read(study/'PROTOCOL.json');check=w.read(CHECK)
    assert check['status']=='PASS'
    sources=dict(base['source_sha256']);sources.update(check['source_sha256'])
    for path in (Path(__file__),Path(previous.__file__)):
        sources[str(path.relative_to(ROOT))]=w.digest(path)
    p=dict(status='REGISTERED_LAG_FEATURE_TRAIN4_PROBE',source_sha256=sources,
        dependency=str(dependency),dependency_protocol_sha256=w.digest(dependency/'PROTOCOL.json'),
        parent_checkpoint=base['parent_checkpoint'],parent_checkpoint_sha256=base['parent_checkpoint_sha256'],
        preparation=base['preparation'],preparation_sha256=base['preparation_sha256'],
        gpu_display_policy=base['gpu_display_policy'],cpu_check_sha256=w.digest(CHECK),
        train_indices=[4,5,2,11],arms=list(ARMS),updates_per_arm=64,observations=[0,1,8,16,32,64],
        effective_batch=4,microbatch=1,seed=0,original_parameters=32142859,
        parameters={'lag_power':32146315,'lag_power_phase':32149771},
        original_lr=1e-5,added_lr=1e-3,optimizer='fresh AdamW wd1e-4 clip1 FP32 TF32off',
        loss='unchanged whole-crop PIT NMSE+coherence+inactive/background plus0.1 count CE',
        input='same mixed IQ, power context and crop; no oracle source count/category/reference',
        lags_samples=check['lags_samples'],fs_hz=100000000,
        comparison='Shared exact-sample lag power channels; candidate adds six bounded complex cross-product channels at the first convolution',
        capacity_difference='Candidate has3456 more parameters than power control (about0.011% of full model); not exactly parameter-matched',
        initialization='Same retained parent, all additional input weights zero, exact same initial predictions',
        acceptance='Report final64 counts2/3 NMSE and complex SI-SDR; joint lower NMSE and higher SI-SDR against power control needed for a favorable directional screen',
        limitation='Previously examined TRAIN4, not generalization; delayed STFT relationships differ from raw-IQ autocorrelation diagnostics',
        validation_read=False,heldout_read=False,weights_discarded=True,registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dst=root/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
    w.write(root/'PROTOCOL.json',p);return p


def verify(root,p):
    for rel,sha in p['source_sha256'].items():
        assert w.digest(ROOT/rel)==sha and w.digest(root/'source_snapshot'/rel)==sha
    for path,sha in [(Path(p['parent_checkpoint']),p['parent_checkpoint_sha256']),
        (Path(p['preparation'])/'PREPARATION.json',p['preparation_sha256']),
        (Path(p['dependency'])/'PROTOCOL.json',p['dependency_protocol_sha256']),(CHECK,p['cpu_check_sha256'])]:
        assert w.digest(path)==sha


def run(root,p,public):
    dependency=Path(p['dependency'])
    while not (dependency/'COMPLETE.json').exists() or not AUDIT.exists():
        if (dependency/'FAILURE.json').exists():raise RuntimeError('Predecessor failed')
        if time.time()-p['registered_at']>24*3600:raise TimeoutError('Queue wait limit')
        w.write(root/'STATE.json',dict(status='WAITING_CANONICAL_AND_AUDIT',pid=os.getpid(),time=time.time()))
        time.sleep(30)
    audit=w.read(AUDIT)
    assert audit['status']=='PASS' and audit['study_protocol_sha256']==p['dependency_protocol_sha256']
    assert audit['complete_sha256']==w.digest(dependency/'COMPLETE.json')
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p)
        assert torch.cuda.is_available()
        w.write(root/'GPU_START_CHECK.json',previous.guard.wait_for_predecessor(w.read(dependency/'STATE.json')['pid'],p['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=worker.NativeMixtures(p['preparation'],'train_pack',1)
        items=[worker.fit.base.batch([data[i]]) for i in p['train_indices']]
        initial=None;results=[]
        for arm in ARMS:
            net=augment(worker.make_model('retained_unet',Path(p['parent_checkpoint'])),arm).cuda().eval()
            assert sum(q.numel() for q in net.parameters())==p['parameters'][arm]
            with torch.no_grad():values=[tuple(x.cpu() for x in worker.predict(net,item)) for item in items]
            if initial is None:initial=values
            else:
                for a,b in zip(initial,values):
                    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
            del values
            old,new=[],[]
            for name,q in net.named_parameters():
                (new if name.startswith(('down.0.0.power.','down.0.0.phase.')) else old).append(q)
            assert sum(q.numel() for q in old)==32142859
            assert sum(q.numel() for q in new)==p['parameters'][arm]-32142859
            assert len({id(q) for q in old+new})==len(list(net.parameters()))
            opt=torch.optim.AdamW([dict(params=old,lr=1e-5),dict(params=new,lr=1e-3)],weight_decay=1e-4,foreach=False)
            torch.manual_seed(0);history=[dict(step=0,**previous.output_fit.score(net,items))];started=time.time()
            for step in range(1,65):
                net.train();opt.zero_grad(set_to_none=True)
                for item in items:
                    estimates,logits=worker.predict(net,item)
                    loss=worker.pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])['loss']+.1*F.cross_entropy(logits,item['construction_count']-1)
                    assert torch.isfinite(loss);(loss/4).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
                if step in p['observations']:
                    history.append(dict(step=step,**previous.output_fit.score(net,items)))
                    w.write(root/(arm+'_HISTORY.json'),dict(history=history))
                    w.write(root/'STATE.json',dict(status='GPU_TRAIN4_LAG_PROBE',arm=arm,step=step,total=64,pid=os.getpid(),time=time.time()))
            assert {int(s['step']) for s in opt.state.values()}=={64}
            assert all(torch.isfinite(v).all() for v in net.state_dict().values())
            result=dict(arm=arm,parameters=p['parameters'][arm],updates=64,history=history,
                optimizer_groups=[dict(lr=g['lr'],parameters=sum(q.numel() for q in g['params'])) for g in opt.param_groups],seconds=time.time()-started)
            results.append(result);w.write(root/(arm+'_COMPLETE.json'),result)
            del net,opt,item,estimates,logits,loss,old,new,q;gc.collect();torch.cuda.empty_cache()
        verify(root,p)
        result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),results=results,
            initial_predictions_exactly_equal=True,validation_read=False,heldout_read=False,weights_discarded=True,time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(public,result)
        w.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()));print('COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('study','dependency','run','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    try:
        with (root/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            p=register(root,a.study.resolve(),a.dependency.resolve());run(root,p,a.public.resolve())
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
