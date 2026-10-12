"""Continuation of the same B U-Net from audited epoch50 through total epoch100."""
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
from models import build, predict
from study import atomic_torch, batch
from drone_rf.losses import pit_waveform_loss
from drone_rf.waveform import waveform_metrics

sys.path.insert(0, str(ROOT/'experiments/phase_eight_20261011'))
from core import predict_eight

PUBLIC = ROOT/'reports/2026-10-12'
PLAN = PUBLIC/'TIMEBLOCK100_PLAN_KO.md'
LOCK = Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock')
STOP = False


def state(status, **kw):
    write_json(OUT/'STATE.json', dict(status=status, pid=os.getpid(), time=time.time(), **kw))
    text = '# 동일 10 MHz 시간 분리 진단: 실행 상태\n\n'
    text += f"상태: {status}. Epoch {kw.get('epoch',0)}/100, 누적 업데이트 {kw.get('updates',0)}.\n\n"
    text += f"학습 혼합 {kw.get('examples',0)}/2400, 검증 {kw.get('case',0)}/630.\n\n"
    if 'parent_state' in kw:
        parent = kw['parent_state']
        text += (f"기존 실행: {parent['status']}, 학습 {parent.get('examples',0)}/2400, "
                 f"검증 {parent.get('case',0)}/630. 50epoch 감사 완료 후 자동 연결.\n\n")
    text += '동일 B 모델의 50→100epoch 연속 학습. 같은 녹음의 시간 분리 진단이며 독립 녹음 일반화 결과가 아니다.\n'
    (PUBLIC/'TIMEBLOCK100_LIVE_KO.md').write_text(text)


def verify(p):
    for rel, h in p['source_sha256'].items():
        assert sha256(ROOT/rel) == h, rel
        assert sha256(OUT/'source_snapshot'/rel) == h, rel
    for path, h in p['pinned_files'].items():
        assert sha256(path) == h, path


def register():
    from handoff import register_extension
    return register_extension()


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
        torch.manual_seed(0)
        net=build('unet_mean').cuda()
        assert sum(q.numel() for q in net.parameters())==p['parameters']
        opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],foreach=False)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-6)
        dev=Mixtures(ROLES[1]); assert dev.rows_hash==p['fixed_dev_schedule_sha256']
        updates=0; first=1; partial=None
        if resume:
            handoff = read(OUT/'HANDOFF.json')
            assert handoff['status']=='PASS' and handoff['protocol_sha256']==ph
            choices=[f for f in (OUT/'LAST.pt',OUT/'RECOVERY.pt') if f.exists()]
            meta=[(f,torch.load(f,map_location='cpu',weights_only=False,mmap=True)) for f in choices]
            # A LAST written just before a crash in validation/audit is not a
            # committed epoch. Replay its full-epoch RECOVERY evaluation instead.
            meta=[(f,s) for f,s in meta if not s['epoch_evaluated'] or s['epoch']==50
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
            from handoff import install_parent
            first, updates, best = install_parent(net, opt, scheduler, ph, save_checkpoint)
        assert 51 <= first <= 101 and 3750 <= updates <= 7500
        producer=feature_worker(first) if first <= 100 else None
        for epoch in range(first,101):
            verify(p)
            assert shutil.disk_usage(OUT).free > 1_500_000_000, 'Disk reserve below 1.5 GB'
            state('WAITING_CPU_FEATURES',epoch=epoch,updates=updates)
            assert producer.wait()==0, 'CPU feature generation failed'
            train=Mixtures(ROLES[0],epoch)
            producer=feature_worker(epoch+1) if epoch<100 else None
            net.train();opt.zero_grad(set_to_none=True)
            began=time.time();start=0;offset=0.;losses=[];norms=[]
            if partial is not None:
                start=partial['next_example'];losses=partial['losses'];norms=partial['gradient_norms'];offset=partial['train_seconds'];partial=None
            assert start%32==0 and updates==(epoch-1)*75+start//32
            for i in range(start,len(train)):
                item=batch(train[i],'cuda')
                loss,wave,count=objective(net,item)
                assert torch.isfinite(loss)
                (loss/32).backward()
                losses.append([float(loss.detach()),float(wave.detach()),float(count.detach())])
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
            # Model-only epochs cost ~129MB each; no duplicated optimizer archives.
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
    parser.add_argument('--wait-parent',action='store_true')
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,stop_request);signal.signal(signal.SIGINT,stop_request)
    try:
        with (OUT/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.register:
                register()
            else:
                if args.wait_parent:
                    from handoff import wait_parent
                    wait_parent(state, lambda: STOP)
                run(read(OUT/'PROTOCOL.json'),args.resume)
    except SystemExit:
        raise
    except BaseException:
        write_json(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        state('FAILED')
        raise
