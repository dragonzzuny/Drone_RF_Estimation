"""Full U-Net C8 mean supervision + normalized orbit consistency, ten epochs."""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from common import ROOT, OUT, PUBLIC, PARENT, PARENT_SHA, PREPARATION, worker, w
import objective as obj
import evaluate as ev

PLAN=PUBLIC/'PHASE_CONSISTENT_PLAN_KO.md'

def state(status, **kw):
    payload=dict(status=status,pid=os.getpid(),time=time.time(),**kw)
    w.write(OUT/'STATE.json',payload)
    w.write(PUBLIC/'PHASE_CONSISTENT_LIVE_KO.md',
        '# 8위상 일관성 U-Net 실제 상태\n\n'+
        f"상태 {status}, 추가epoch {kw.get('epoch',0)}/10, 누적업데이트 {kw.get('updates',0)}/750.\n\n"+
        f"현재학습혼합 {kw.get('examples',0)}/2400; 평가 {kw.get('case',0)}/630.\n\n"+
        '같은 전체 U-Net의8위상 평균 파형 손실 +0.1 위상 일관성; 자료 분할 유지. '
        '자동 채팅 보고가 아니며 실행과 성능 개선을 구분한다.\n')

def verify(p):
    for rel,h in p['source_sha256'].items():
        assert w.digest(ROOT/rel)==h and w.digest(OUT/'source_snapshot'/rel)==h,rel
    for path,h in p['pinned_files'].items():
        assert w.digest(Path(path))==h,path

def register():
    OUT.mkdir(exist_ok=False)
    assert w.digest(PARENT)==PARENT_SHA
    checks=[PUBLIC/'PHASE_CONSISTENT_CPU_CHECK.json',PUBLIC/'PHASE_CONSISTENT_GPU_CHECK.json']
    old=w.read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')
    sources=dict(old['source_sha256'])
    for path in Path(__file__).parent.glob('*.py'):
        sources[str(path.relative_to(ROOT))]=w.digest(path)
    for path in checks:
        check=w.read(path);assert check['status']=='PASS'
        for rel,h in check['source_sha256'].items():assert sources[rel]==h,rel
    train=[worker.NativeMixtures(PREPARATION,'train_pack',k) for k in range(1,6)]
    validation=worker.NativeMixtures(PREPARATION,'validation_pack',1)
    feature_paths=[PREPARATION/'features'/f'train_pack_{k:03d}.json' for k in range(1,6)]
    feature_paths += [PREPARATION/'features/validation_pack_001.json',PREPARATION/'PREPARATION.json',
                      PREPARATION/'NATIVE_MANIFEST.json']
    baseline=PUBLIC/'PHASE_EIGHT_RESULT.json'
    pinned=[PLAN,PARENT,baseline,*checks,*feature_paths]
    for rel,h in sources.items():
        assert w.digest(ROOT/rel)==h
        target=OUT/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target)
    p=dict(status='REGISTERED_C8_MEAN_AND_CONSISTENCY_TRAINING',source_sha256=sources,
        pinned_files={str(f):w.digest(f) for f in pinned},parent_checkpoint=str(PARENT),
        parent_checkpoint_sha256=PARENT_SHA,parent_selected_epoch=2,preparation=str(PREPARATION),
        train_rows_sha256={str(k+1):d.rows_hash for k,d in enumerate(train)},
        dev_rows_sha256=validation.rows_hash,baseline=str(baseline),seed=0,
        parameters=32142859,trainable_parameters=32142859,epochs=10,updates_per_epoch=75,
        train_schedules=[1,2,3,4,5,1,2,3,4,5],examples_per_epoch=2400,
        effective_batch=32,microbatch=1,learning_rate=1e-6,weight_decay=1e-4,clip_norm=1.,
        consistency_weight=.1,count_weight=.1,angles_degrees=list(obj.DEGREES),
        forward_per_example=16,backward_per_example=8,precision='FP32 TF32off',
        initialization='original full native e2; preflight update discarded',
        loss='original PIT waveform loss on C8 mean +0.1 normalized orbit variance +0.1 count CE',
        selection='joint improvement versus fixed C8 parent; among passing epochs minimum mean NMSE counts2/3',
        early_stopping='no one-epoch performance stop; ten epochs unless numeric/integrity/resource failure or user stop',
        inference_reference_access=False,heldout_read=False,independent_test=False,
        limitation='combined training-objective intervention and extra updates; no matched extra-budget ablation',
        registered_at=time.time())
    w.write(OUT/'PROTOCOL.json',p);verify(p)
    return p

def checkpoint(net,opt,epoch,updates,best,ph,**kw):
    return dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,updates=updates,
        best=best,protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),**kw)

def audit_epoch(epoch):
    subprocess.run(['taskset','-c','12,13',sys.executable,str(Path(__file__).with_name('audit.py')),
                    '--epoch',str(epoch)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)

def run(p,resume):
    ph=w.digest(OUT/'PROTOCOL.json'); baseline=w.read(Path(p['baseline']))['eight_phase']
    with worker.fit.base.LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);verify(p)
        assert torch.cuda.is_available()
        torch.set_num_threads(2);torch.manual_seed(0)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        net=worker.make_model('retained_unet',PARENT).cuda()
        assert sum(q.numel() for q in net.parameters())==p['parameters']
        assert all(q.requires_grad for q in net.parameters()) and not list(net.buffers())
        opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],foreach=False)
        validation=worker.NativeMixtures(PREPARATION,'validation_pack',1)
        assert validation.rows_hash==p['dev_rows_sha256']
        best=dict(epoch=0,metric=baseline['selection_nmse']); first=1;updates=0;partial=None
        if resume:
            options=[]
            for name in ('LAST.pt','RECOVERY.pt'):
                if (OUT/name).exists():
                    value=torch.load(OUT/name,map_location='cpu',weights_only=False)
                    assert value['protocol_sha256']==ph
                    options.append(value)
            if not options:raise RuntimeError('No optimizer recovery yet; do not overwrite registered run')
            old=max(options,key=lambda r:(r['updates'],r['epoch_evaluated']))
            net.load_state_dict(old['model']);opt.load_state_dict(old['optimizer'])
            first=old['epoch']+int(old['epoch_evaluated']);updates=old['updates'];best=old['best']
            torch.set_rng_state(old['torch_rng']);torch.cuda.set_rng_state_all(old['cuda_rng'])
            if not old['epoch_evaluated']:
                partial={k:old[k] for k in ('next_example','counts','losses','norms','train_seconds')}
            del old,options
            previous=w.read(OUT/f'EIGHT_{first-1:03d}.json')
        else:
            state('BASELINE_REPRODUCTION',epoch=0,updates=0)
            single,initial,diag=ev.evaluate(net,validation,baseline,0,state)
            for actual,expected in zip(initial['rows'],baseline['rows']):
                np.testing.assert_allclose(actual['nmse'],expected['nmse'],rtol=1e-5,atol=1e-6)
                np.testing.assert_allclose(actual['si_sdr'],expected['si_sdr'],rtol=1e-5,atol=1e-4)
            w.write(OUT/'SINGLE_000.json',single);w.write(OUT/'EIGHT_000.json',initial)
            w.write(OUT/'DIAGNOSTIC_000.json',diag)
            worker.atomic_torch(OUT/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
            previous=initial
        for epoch in range(first,p['epochs']+1):
            verify(p); schedule=p['train_schedules'][epoch-1]
            train=worker.NativeMixtures(PREPARATION,'train_pack',schedule)
            assert train.rows_hash==p['train_rows_sha256'][str(schedule)] and len(train)==2400
            net.train();opt.zero_grad(set_to_none=True);began=time.time();counts=[0,0,0];losses=[];norms=[]
            start_index=0;elapsed_offset=0.
            if partial is not None:
                start_index=partial['next_example'];counts=partial['counts'];losses=partial['losses'];norms=partial['norms']
                elapsed_offset=partial['train_seconds'];partial=None
                assert start_index%32==0 and updates==(epoch-1)*75+start_index//32
            torch.cuda.reset_peak_memory_stats()
            state('TRAINING',epoch=epoch,updates=updates,examples=start_index)
            for i in range(start_index,len(train)):
                item=worker.fit.base.batch([train[i]])
                receipt=obj.backward_replay(net,worker.predict,item,divisor=32,
                                           consistency_weight=p['consistency_weight'],check_replay=(i==start_index))
                losses.append({k:v for k,v in receipt.items() if k!='prediction_only_orders'})
                counts[int(item['construction_count'][0])-1]+=1
                if (i+1)%32==0:
                    norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
                    norms.append(norm);opt.step();opt.zero_grad(set_to_none=True);updates+=1
                    if len(norms)%5==0:
                        worker.atomic_torch(OUT/'RECOVERY.pt',checkpoint(net,opt,epoch,updates,best,ph,
                            next_example=i+1,epoch_evaluated=False,train_seconds=elapsed_offset+time.time()-began,
                            counts=counts,losses=losses,norms=norms))
                if (i+1)%8==0:
                    elapsed=elapsed_offset+time.time()-began
                    state('TRAINING',epoch=epoch,updates=updates,examples=i+1,seconds=elapsed,
                        estimated_training_seconds=elapsed*2400/(i+1),
                        recent_loss={k:float(np.mean([r[k] for r in losses[-32:]])) for k in losses[-1]})
            assert counts==[800,800,800] and len(norms)==75 and updates==epoch*75
            assert {int(v['step']) for v in opt.state.values()}=={updates}
            train_receipt=dict(epoch=epoch,schedule=schedule,updates=updates,examples=2400,counts=counts,
                seconds=elapsed_offset+time.time()-began,mean_loss={k:float(np.mean([r[k] for r in losses])) for k in losses[0]},
                gradient_norms=norms,peak_gpu_bytes=torch.cuda.max_memory_allocated(),protocol_sha256=ph)
            w.write(OUT/f'TRAIN_{epoch:03d}.json',train_receipt)
            # Epoch optimizer is saved separately so CPU audit can run without
            # racing the next epoch's in-memory parameters.
            epoch_path=OUT/f'ACTUAL_{epoch:03d}.pt'
            worker.atomic_torch(epoch_path,checkpoint(net,opt,epoch,updates,best,ph,epoch_evaluated=False))
            del train;gc.collect()
            started=time.time();single,eight,diagnostic=ev.evaluate(net,validation,baseline,epoch,state)
            w.write(OUT/f'SINGLE_{epoch:03d}.json',single);w.write(OUT/f'EIGHT_{epoch:03d}.json',eight)
            w.write(OUT/f'DIAGNOSTIC_{epoch:03d}.json',diagnostic)
            passed,checks=ev.acceptable(eight,baseline)
            selected=passed and eight['selection_nmse']<best['metric']
            if selected:
                best=dict(epoch=epoch,metric=eight['selection_nmse'])
                worker.atomic_torch(OUT/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
            event=dict(epoch=epoch,updates=updates,training=train_receipt,
                single={k:v for k,v in single.items() if k!='rows'},
                eight={k:v for k,v in eight.items() if k!='rows'},diagnostic=diagnostic,
                baseline_diagnostic=w.read(OUT/'DIAGNOSTIC_000.json'),
                preceding_epoch={k:v for k,v in previous.items() if k!='rows'},
                candidate=passed,checks=checks,selected_this_epoch=selected,best=best,
                evaluation_seconds=time.time()-started,protocol_sha256=ph,
                checkpoint_sha256=w.digest(epoch_path),heldout_read=False,independent_test=False,
                audit_status='PENDING',incumbent_replaced=False)
            w.write(OUT/f'EPOCH_{epoch:03d}.json',event)
            w.write(PUBLIC/f'PHASE_CONSISTENT_E{epoch:03d}.json',event)
            worker.atomic_torch(OUT/'LAST.pt',checkpoint(net,opt,epoch,updates,best,ph,epoch_evaluated=True))
            state('CPU_AUDIT',epoch=epoch,updates=updates,case=630,best=best)
            audit_epoch(epoch)
            print(f'EPOCH_COMPLETE {epoch} '+str(event['eight']['by_count']),flush=True)
            previous=eight
        verify(p)
        w.write(OUT/'COMPLETE.json',dict(status='COMPLETE_AUDITED',epochs=p['epochs'],updates=updates,
                best=best,protocol_sha256=ph,heldout_read=False,incumbent_replaced=False,time=time.time()))
        state('COMPLETE_AUDITED',epoch=p['epochs'],updates=updates,best=best)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:
        p=w.read(OUT/'PROTOCOL.json') if args.resume else register()
        with (OUT/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(p,args.resume)
    except BaseException:
        if OUT.exists():
            w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time(),pid=os.getpid()))
            state('FAILED')
        raise

