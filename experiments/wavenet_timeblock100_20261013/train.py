"""Fresh native WaveNet: fixed 100 epochs on the registered time-block diagnostic."""
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
from torch.nn import functional as F
from data import *
from models import predict
sys.path.insert(0,str(ROOT/'experiments/architecture_audit_20261009'))
from native_wavenet import build_native_wavenet
from study import atomic_torch, batch
from drone_rf.losses import pit_waveform_loss
from drone_rf.waveform import waveform_metrics

sys.path.insert(0, str(ROOT/'experiments/phase_eight_20261011'))
from core import predict_eight

PUBLIC = ROOT/'reports/2026-10-13'
PLAN = PUBLIC/'WAVENET100_PLAN_KO.md'
LOCK = Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock')
STOP = False


def state(status, **kw):
    write_json(OUT/'STATE.json', dict(status=status, pid=os.getpid(), time=time.time(), **kw))
    text = '# 동일 10 MHz 시간 분리 진단: 실행 상태\n\n'
    text += f"상태: {status}. Epoch {kw.get('epoch',0)}/100, 누적 업데이트 {kw.get('updates',0)}.\n\n"
    text += f"학습 혼합 {kw.get('examples',0)}/2400, 검증 {kw.get('case',0)}/630.\n\n"
    text += '새 초기화 WaveNet30층·128채널, 별도 B 분할. 같은 녹음의 시간 분리 진단이며 독립 녹음 일반화 결과가 아니다.\n'
    (PUBLIC/'WAVENET100_LIVE_KO.md').write_text(text)


def verify(p):
    for rel, h in p['source_sha256'].items():
        assert sha256(ROOT/rel) == h, rel
        assert sha256(OUT/'source_snapshot'/rel) == h, rel
    for path, h in p['pinned_files'].items():
        assert sha256(path) == h, path


def register():
    assert not (OUT/'PROTOCOL.json').exists(), 'Already registered'
    ready=read(OUT/'READY.json');assert ready['status']=='PASS'
    data=Mixtures(ROLES[0]);dev=Mixtures(ROLES[1])
    assert len(data)==2400 and len(dev)==630
    sources=[*HERE.glob('*.py'),*LEGACY.glob('*.py'),*(LEGACY/'vendor/drone_rf').glob('*.py'),
             ROOT/'experiments/architecture_audit_20261009/native_wavenet.py',
             ROOT/'experiments/phase_eight_20261011/core.py',
             ROOT/'experiments/rfuav_phase_average_20261009/phase_core.py']
    hashes={str(f.relative_to(ROOT)):sha256(f) for f in sources}
    for rel in hashes:
        target=OUT/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target)
    pins=[PLAN,MANIFEST,OUT/'READY.json',PREP/'PREPARATION.json',
          ROOT/'local/bandwidth_timeblock_20261011_v1/PROTOCOL.json',
          ROOT/'local/timeblock_extend100_20261012_v1/PROTOCOL.json']
    p=dict(status='REGISTERED_FRESH_WAVENET_B100',seed=0,parameters=4601867,
           epochs=100,performance_early_stopping=False,examples_per_epoch=2400,
           updates_per_epoch=75,effective_batch=32,microbatch=4,learning_rate=1e-4,
           weight_decay=1e-4,clip_norm=1.,count_weight=.1,
           lr_scheduler='ReduceLROnPlateau factor0.5 patience5 min1e-6, same rule not same adaptive LR values',
           model='NativeWaveNet:30 layers,128 channels,dilation cycle10;3 complex slots+background',
           convolutional_receptive_field_samples=6139,
           initialization='seed0 fresh random tensors; no project checkpoint loaded',
           training_loss='same PIT NMSE+coherence/inactive/background plus0.1 count CE',
           phase_training='same U-Net100 epoch schedules including independent source phases',
           inference='same reference-free C8',
           selection='minimum C8 counts2/3 macroNMSE over epochs0..100',
           precision='FP32,TF32off',source_sha256=hashes,
           pinned_files={str(f):sha256(f) for f in pins},
           old_checkpoint_reuse=False,independent_recording_validation=False,
           heldout_read=False,dev20_read=False,
           fixed_dev_schedule_sha256=dev.rows_hash,registered_at=time.time(),
           chat_report_every_completed_epochs=5,
           comparison='same B split,100 schedules,7500 updates; unequal params/compute; reused U-Net100',
           microbatch_equivalence='requires full-input GPU single vs batch4 gradient/forward check')
    write_json(OUT/'PROTOCOL.json',p);verify(p);state('REGISTERED');return p


def objective(net, item):
    estimates, logits = predict(net, item)
    wave = pit_waveform_loss(estimates, item['references'], item['active'], item['mixture'])['loss']
    count = F.cross_entropy(logits, item['construction_count']-1)
    return wave+.1*count, wave, count


def aggregate(rows):
    groups = []
    for count in (1,2,3):
        group = [r for r in rows if r['count']==count]
        assert len(group) == 210
        assert all(np.isfinite(r['nmse']).all() for r in group)
        values = [v for r in group for v in r['si_sdr']]
        groups.append(dict(count=count, cases=len(group),
            mean_nmse=float(np.mean([v for r in group for v in r['nmse']])),
            mean_si_sdr=float(np.mean(values)) if all(v is not None for v in values) else None,
            nonfinite_si_sdr=sum(v is None for v in values),
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in group])),
            count_accuracy=float(np.mean([r['predicted_count']==count for r in group]))))
    return dict(by_count=groups, selection_nmse=float(np.mean([g['mean_nmse'] for g in groups[1:]])))


@torch.no_grad()
def evaluate(net, dev, epoch, updates):
    net.eval()
    rows=[]
    for i in range(len(dev)):
        item = batch(dev[i], 'cuda')
        _, estimates, logits, _ = predict_eight(net, predict, item)
        m = waveform_metrics(estimates, item['references'], item['active'], item['mixture'])
        active=item['active'][0]
        r=dev.rows[i]
        count=int(r['count'])
        assert int(active.sum())==count
        cs=[dev.library.clips[int(j)] for j in r['indices'][:count]]
        power=m['reference_power'][0][active].cpu().tolist()
        si=m['si_sdr'][0][active].cpu().tolist()
        assert float(m['sum_relative_error'].max()) < 1e-9
        rows.append(dict(index=i, count=count, categories=[c['category'] for c in cs],
            clip_ids=[c['clip_id'] for c in cs], nominal_levels_db=r['levels'][:count].tolist(),
            reference_power=power, nmse=m['nmse'][0][active].cpu().tolist(),
            si_sdr=[v if np.isfinite(v) else None for v in si],
            si_status=['finite' if np.isfinite(v) else 'undefined_or_infinite' for v in si],
            weakest_index=int(np.argmin(power)), predicted_count=int(logits.argmax(-1)[0])+1,
            assignment=m['assignment'][0].cpu().tolist(),
            sum_relative_error=float(m['sum_relative_error'][0])))
        if i%25==0:
            state('VALIDATING', epoch=epoch, updates=updates, case=i+1)
    result=dict(epoch=epoch, updates=updates, rows_sha256=dev.rows_hash, rows=rows,
        **aggregate(rows), independent_recording_validation=False)
    write_json(OUT/f'VALIDATION_{epoch:03d}.json', result)
    return result


def save_checkpoint(path, net, opt, scheduler, epoch, updates, best, ph, **kw):
    atomic_torch(path, dict(model=net.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
        epoch=epoch, updates=updates, best=best, protocol_sha256=ph,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), **kw))


def feature_worker(epoch):
    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (OUT/'features.log').open('ab') as log:
        return subprocess.Popen(['nice','-n','5','taskset','-c','12,13',sys.executable,str(HERE/'data.py'),
            '--role',ROLES[0],'--epoch',str(epoch)], stdout=log,stderr=subprocess.STDOUT,env=env)


def grouped_batch(items):
    keys=('mixture','references','active','context_features','crop_start','construction_count')
    return {k:torch.as_tensor(np.stack([item[k] for item in items]),device='cuda') for k in keys}


def gpu_preflight(p):
    state('GPU_PREFLIGHT')
    net=build_native_wavenet().cuda().train()
    assert sum(q.numel() for q in net.parameters())==p['parameters']
    train=Mixtures(ROLES[0])
    indices=[int(np.flatnonzero(train.rows['count']==k)[0]) for k in (1,2,3)]
    indices.append(int(np.flatnonzero(train.rows['count']==3)[1]))
    items=[train[i] for i in indices]
    single_predictions=[];losses=[]
    for item in items:
        entry=batch(item,'cuda');prediction,_=predict(net,entry)
        single_predictions.append(prediction.detach().cpu())
        loss,_,_=objective(net,entry);(loss/4).backward();losses.append(float(loss.detach()))
    reference={n:q.grad.detach().cpu().clone() for n,q in net.named_parameters() if q.grad is not None}
    net.zero_grad(set_to_none=True)
    combined=grouped_batch(items)
    with torch.no_grad():prediction,_=predict(net,combined)
    torch.testing.assert_close(prediction.cpu(),torch.cat(single_predictions),rtol=2e-4,atol=2e-6)
    loss,_,_=objective(net,combined);loss.backward()
    numerator=denominator=0.
    for n,q in net.named_parameters():
        assert q.grad is not None and torch.isfinite(q.grad).all()
        current=q.grad.detach().cpu().double();prior=reference[n].double()
        numerator+=float((current-prior).square().sum());denominator+=float(prior.square().sum())
    relative_error=(numerator/max(denominator,1e-30))**.5
    assert relative_error<1e-3,relative_error
    assert abs(float(loss)-float(np.mean(losses)))<1e-4
    optimizer=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],foreach=False)
    norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
    optimizer.step()
    assert norm>0 and all(torch.isfinite(q).all() for q in net.parameters())
    write_json(OUT/'GPU_PREFLIGHT.json',dict(status='PASS',indices=indices,losses=losses,
        gradient_norm=norm,microbatch_relative_gradient_error=relative_error,
        single_vs_batch4_forward='PASS',discarded_updates=1,parameters=p['parameters'],
        max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),protocol_sha256=sha256(OUT/'PROTOCOL.json')))
    del net,optimizer,train,loss,reference,prediction,combined,single_predictions
    gc.collect();torch.cuda.empty_cache()


def run(p, resume):
    ph=sha256(OUT/'PROTOCOL.json')
    with LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK')
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify(p)
        assert torch.cuda.is_available()
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        if not resume:
            gpu_preflight(p)
        torch.manual_seed(0)
        net=build_native_wavenet().cuda()
        assert sum(q.numel() for q in net.parameters())==p['parameters']
        opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],foreach=False)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-6)
        dev=Mixtures(ROLES[1]); assert dev.rows_hash==p['fixed_dev_schedule_sha256']
        updates=0; first=1; partial=None
        if resume:
            choices=[f for f in (OUT/'LAST.pt',OUT/'RECOVERY.pt') if f.exists()]
            meta=[(f,torch.load(f,map_location='cpu',weights_only=False,mmap=True)) for f in choices]
            # A LAST written just before a crash in validation/audit is not a
            # committed epoch. Replay its full-epoch RECOVERY evaluation instead.
            meta=[(f,s) for f,s in meta if not s['epoch_evaluated'] or s['epoch']==0
                  or (OUT/f"AUDIT_{s['epoch']:03d}.json").exists()]
            assert meta, 'No audited LAST or valid optimizer RECOVERY available'
            path,saved=max(meta,key=lambda v:(v[1]['updates'],v[1]['epoch_evaluated']))
            assert saved['protocol_sha256']==ph
            net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            updates=saved['updates'];best=saved['best'];first=saved['epoch']+int(saved['epoch_evaluated'])
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            if not saved['epoch_evaluated']:
                partial={k:saved[k] for k in ('next_example','losses','gradient_norms','train_seconds')}
            del saved,meta
        else:
            atomic_torch(OUT/'INITIAL.pt',dict(model=net.state_dict(),epoch=0,updates=0,protocol_sha256=ph))
            measured=evaluate(net,dev,0,0)
            best=dict(epoch=0,metric=measured['selection_nmse'])
            shutil.copyfile(OUT/'INITIAL.pt',OUT/'BEST.pt')
            save_checkpoint(OUT/'LAST.pt',net,opt,scheduler,0,0,best,ph,epoch_evaluated=True)
        producer=feature_worker(first) if first<=100 else None
        for epoch in range(first,101):
            verify(p)
            assert shutil.disk_usage(OUT).free>1_500_000_000
            state('WAITING_CPU_FEATURES',epoch=epoch,updates=updates)
            assert producer.wait()==0, 'CPU feature generation failed'
            train=Mixtures(ROLES[0],epoch)
            producer=feature_worker(epoch+1) if epoch<100 else None
            net.train();opt.zero_grad(set_to_none=True)
            began=time.time();start=0;offset=0.;losses=[];norms=[]
            if partial is not None:
                start=partial['next_example'];losses=partial['losses'];norms=partial['gradient_norms'];offset=partial['train_seconds'];partial=None
            assert start%32==0 and updates==(epoch-1)*75+start//32
            for start_micro in range(start,len(train),4):
                i=start_micro+3
                item=grouped_batch([train[j] for j in range(start_micro,start_micro+4)])
                loss,wave,count=objective(net,item)
                assert torch.isfinite(loss)
                (loss/8).backward()
                losses.extend([[float(loss.detach()),float(wave.detach()),float(count.detach())]]*4)
                if (i+1)%32==0:
                    norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
                    opt.step();opt.zero_grad(set_to_none=True);updates+=1;norms.append(norm)
                    elapsed=offset+time.time()-began
                    if len(norms)%5==0 or STOP:
                        save_checkpoint(OUT/'RECOVERY.pt',net,opt,scheduler,epoch,updates,best,ph,
                            epoch_evaluated=False,next_example=i+1,losses=losses,gradient_norms=norms,train_seconds=elapsed)
                    state('TRAINING',epoch=epoch,updates=updates,examples=i+1,seconds=elapsed,
                        estimated_training_seconds=elapsed*2400/(i+1),recent_loss=np.mean(losses[-32:],axis=0).tolist(),
                        learning_rate=opt.param_groups[0]['lr'])
                    if STOP:
                        state('STOPPED_WITH_RECOVERY',epoch=epoch,updates=updates,examples=i+1)
                        return
            assert updates==epoch*75 and len(losses)==2400 and len(norms)==75
            train_seconds=offset+time.time()-began
            started=time.time();measured=evaluate(net,dev,epoch,updates)
            selected=measured['selection_nmse']<best['metric']
            if selected:
                best=dict(epoch=epoch,metric=measured['selection_nmse'])
                atomic_torch(OUT/'BEST.pt',dict(model=net.state_dict(),epoch=epoch,updates=updates,best=best,protocol_sha256=ph))
            scheduler.step(measured['selection_nmse'])
            save_checkpoint(OUT/'LAST.pt',net,opt,scheduler,epoch,updates,best,ph,epoch_evaluated=True)
            # WaveNet model-only epochs; no duplicated optimizer archives.
            atomic_torch(OUT/f'ACTUAL_{epoch:03d}.pt',dict(model=net.state_dict(),epoch=epoch,updates=updates,protocol_sha256=ph))
            event=dict(epoch=epoch,updates=updates,best=best,selected=selected,protocol_sha256=ph,
                train_seconds=train_seconds,evaluation_seconds=time.time()-started,
                mean_training_loss=np.mean(losses,axis=0).tolist(),gradient_norms=norms,
                learning_rate_next=opt.param_groups[0]['lr'],checkpoint_sha256=sha256(OUT/f'ACTUAL_{epoch:03d}.pt'),
                validation_sha256=sha256(OUT/f'VALIDATION_{epoch:03d}.json'),
                training_rows_sha256=train.rows_hash,validation={k:v for k,v in measured.items() if k!='rows'})
            write_json(OUT/f'EPOCH_{epoch:03d}.json',event)
            state('CPU_AUDIT',epoch=epoch,updates=updates)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
            subprocess.run(['taskset','-c','12,13',sys.executable,str(HERE/'audit.py'),'--epoch',str(epoch)],env=env,check=True)
            print(f'EPOCH_COMPLETE {epoch} '+str(measured['by_count']),flush=True)
            # Keep receipts; drop only regenerable completed-epoch feature arrays.
            old_feature=train.stem.with_suffix('.npy'); del train;gc.collect()
            old_receipt=old_feature.with_suffix('.json')
            write_json(OUT/f'FEATURE_CLEANUP_{epoch:03d}.json',dict(path=str(old_feature),
                sha256=sha256(old_feature),reason='completed epoch; reproducible from frozen schedule',time=time.time()))
            old_feature.unlink();old_receipt.rename(old_receipt.with_suffix('.archived.json'))
        state('COMPLETE_AUDITED',epoch=100,updates=updates,best=best)


def stop_request(*_):
    global STOP
    STOP=True


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,stop_request);signal.signal(signal.SIGINT,stop_request)
    try:
        with (OUT/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.register:
                register()
            else:
                run(read(OUT/'PROTOCOL.json'),args.resume)
    except BaseException:
        write_json(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        state('FAILED')
        raise
