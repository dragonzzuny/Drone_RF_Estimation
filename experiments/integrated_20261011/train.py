"""Integrated performance-first experiment; fixed backbone, guarded new layers.

Full architecture/native I/Q. Original fixed schedules and DEV. Each logged
epoch is 2400 examples/75 updates. No inference access to targets or true count.
"""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
from model import ROOT, worker, build, augment, configure, check_frozen, guard_components, module

w = worker.watch
PUBLIC = ROOT/'reports/2026-10-11'
validation = module('integrated_validation', 'experiments/recursive_20261010/validation.py')
Accumulator, correct = guard_components()


def state(root, status, **kwargs):
    w.write(root/'STATE.json', dict(status=status, pid=os.getpid(), time=time.time(), **kwargs))


def register(root):
    assert not (root/'PROTOCOL.json').exists(), 'Use --resume for a registered experiment'
    root.mkdir(parents=True, exist_ok=True)
    old = w.read(ROOT/'local/five_candidates_20261010_v2/PROTOCOL.json')
    sources = dict(old['source_sha256'])
    for path in Path(__file__).parent.glob('*.py'):
        sources[str(path.relative_to(ROOT))] = w.digest(path)
    for relative, sha in sources.items():
        assert w.digest(ROOT/relative) == sha, relative
        dest = root/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, dest)
    pins = {str(path): w.digest(path) for path in [Path(old['parent_checkpoint']), Path(old['baseline']),
            Path(old['preparation'])/'PREPARATION.json', Path(old['preparation'])/'NATIVE_MANIFEST.json',
            PUBLIC/'INTEGRATED_PLAN_KO.md']}
    for path in (Path(old['preparation'])/'features').glob('*pack_*.json'):
        pins[str(path)] = w.digest(path)
    p = dict(status='REGISTERED_INTEGRATED_FROZEN_GUARDED', source_sha256=sources, pinned_files=pins,
        parent_checkpoint=old['parent_checkpoint'], parent_checkpoint_sha256=old['parent_checkpoint_sha256'],
        baseline=old['baseline'], preparation=old['preparation'], parameters=37444428, trainable=5301569,
        seed=0, effective_batch=32, microbatch=1, precision='FP32, TF32 off',
        max_epochs=50, updates_per_epoch=75, examples_per_epoch=2400,
        schedule='(epoch-1)%5+1; same frozen12000 examples, not new recordings',
        optimizer='AdamW: order gate1e-2/wd0; TF1e-4/wd1e-4; source head1e-3/wd1e-4',
        frozen='All original parent parameters. New gate/TF/head trainable. No automatic unfreezing.',
        guard='Actual AdamW displacement on TRAIN count2/3 waveform gradients; subtract count CE; first-order constraint only',
        loss='Unmodified PIT NMSE+coherence+inactive/background and0.1 count CE',
        selection='Minimum DEV mean NMSE count2/3 including e0; success assessed separately on joint waveform criteria',
        decision='At e5 stop if no epoch improves both count2/3 NMSE and complex SI-SDR versus parent while weakest NMSE is nonworse. After e5 stop after5 epochs without >=1e-5 selection improvement. Maximum50. Early catastrophic stop if both NMSE exceed1.25x parent.',
        lr_rule='Half all learning rates after3 consecutive non-improvements; preserve optimizer moments',
        control='Reuse saved parent and prior control. No new matched-budget control; no factorial attribution.',
        heldout_read=False, independent_test=False, scope='RFUAV TRAIN8/DEV5, same band, native100MS/s, retained RF offsets; unseen nominal DEV bandwidth; product-label contributions not proven airframe-only',
        registered_at=time.time(), user_authorization='Prioritize combined model and fast performance; stop current non-improving training')
    w.write(root/'PROTOCOL.json', p)
    shutil.copyfile(p['baseline'], root/'VALIDATION_000.json')
    return p


def verify(root, p):
    for relative, sha in p['source_sha256'].items():
        assert w.digest(ROOT/relative) == w.digest(root/'source_snapshot'/relative) == sha, relative
    for path, sha in p['pinned_files'].items():
        assert w.digest(Path(path)) == sha, path


def backward_one(net, guard, item, divisor):
    output, logits = worker.predict(net, item)
    loss = worker.pit_waveform_loss(output, item['references'], item['active'], item['mixture'])['loss']
    ce = .1*torch.nn.functional.cross_entropy(logits, item['construction_count']-1)
    count = int(item['construction_count'])
    if count in (2,3):
        values = torch.autograd.grad(ce/divisor, guard.ce_parameters, retain_graph=True, allow_unused=True)
        guard.collect_ce(count, values)
    total = loss+ce
    assert torch.isfinite(total)
    (total/divisor).backward()
    value = float(total.detach())
    guard.collect(count)
    return value


def apply_update(net, opt, named, guard):
    params = [p for _,p in named]
    guard.assign(guard.groups.sum(0))
    protected = guard.waveform_groups()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert all(p.grad is None for p in net.parameters() if not p.requires_grad)
    norms = {prefix: float(torch.stack([p.grad.square().sum() for n,p in named
                if n.startswith(prefix)]).sum().sqrt())
             for prefix in ('context_encoder.', 'tf_axes.', 'output.')}
    norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
    before = torch.cat([p.detach().flatten() for p in params])
    opt.step()
    delta, detail = correct(params, before, protected)
    guard.reset()
    opt.zero_grad(set_to_none=True)
    return dict(preclip_norm=norm, component_gradient_norms=norms, **detail)


def preflight(root, p):
    state(root, 'GPU_PREFLIGHT')
    torch.manual_seed(0)
    train = worker.NativeMixtures(p['preparation'], 'train_pack', 1)
    indices = [next(i for i,r in enumerate(train.rows) if int(r['count'])==n) for n in (1,2,3)]
    net = worker.make_model('retained_unet').cuda().eval()
    saved = torch.load(p['parent_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    net.load_state_dict(saved['model']); del saved
    originals = []
    for index in indices:
        item = worker.fit.base.batch([train[index]])
        with torch.no_grad():
            out, logits = worker.predict(net, item)
        originals.append((out.cpu(), logits.cpu()))
    augment(net); net.cuda(); opt,named = configure(net)
    equality = []
    for index, (expected, counts) in zip(indices, originals):
        item = worker.fit.base.batch([train[index]])
        with torch.no_grad():
            out, logits = worker.predict(net, item)
        gap = float((out.cpu()-expected).abs().max())
        assert torch.equal(out.cpu(),expected) and torch.equal(logits.cpu(),counts), gap
        equality.append(dict(index=index, count=int(item['construction_count']), output_exact=True, max_error=gap))
    del originals, out, logits, item, expected, counts
    guard = Accumulator(named); net.train(); updates=[]; losses=[]
    for step in range(3):
        values=[]
        for index in indices:
            item = worker.fit.base.batch([train[index]])
            values.append(backward_one(net, guard, item, 3))
        updates.append(apply_update(net,opt,named,guard)); losses.append(values)
        state(root, 'GPU_PREFLIGHT', diagnostic_updates=step+1, total=3)
    check_frozen(net, p['parent_checkpoint'])
    assert all(updates[-1]['component_gradient_norms'][key]>0 for key in updates[-1]['component_gradient_norms'])
    assert net.tf_axes.blocks[0].time.weight_ih_l0.requires_grad
    assert all(torch.isfinite(t).all() for t in net.state_dict().values())
    result = dict(status='PASS', initial_predictions=equality, diagnostic_updates=3,
        diagnostic_losses=losses, updates=updates, parameters=p['parameters'], trainable=p['trainable'],
        original_parent_weights_unchanged=True, temporary_weights_discarded=True,
        full_model=True, input_samples=63872, peak_bytes=torch.cuda.max_memory_allocated(),
        protocol_sha256=w.digest(root/'PROTOCOL.json'), heldout_read=False, time=time.time(),
        meaning='Implementation and connectivity check on TRAIN3, not validation performance')
    w.write(root/'PREFLIGHT.json',result); w.write(PUBLIC/'INTEGRATED_PREFLIGHT.json',result)
    del net,opt,named,guard,item,train
    gc.collect(); torch.cuda.empty_cache()
    return result


def joint_success(actual, baseline):
    return all(a['mean_nmse']<b['mean_nmse'] and a['mean_si_sdr'] is not None and
               a['mean_si_sdr']>b['mean_si_sdr'] and a['weakest_nmse']<=b['weakest_nmse']
               for a,b in zip(actual['by_count'][1:],baseline['by_count'][1:]))


def publish(root, status):
    events = [w.read(path) for path in sorted(root.glob('EPOCH_*.json'))]
    result = dict(status=status, events=events, completed_epochs=len(events), max_epochs=50,
                  run=str(root), protocol_sha256=w.digest(root/'PROTOCOL.json'), time=time.time(), heldout_read=False)
    w.write(PUBLIC/'INTEGRATED_PROGRESS.json', result)
    lines=['# 통합 모델 학습 진행','',f'상태: {status}. 원 규모37,444,428파라미터, 추가5,301,569개 학습. '
           '1epoch=2,400합성·75업데이트. 동일 DEV630이며 독립 시험이 아니다.','',
           '|epoch|NMSE2/3 ↓|복소SI-SDR2/3 ↑ dB|최약NMSE2/3 ↓|선택|공동 기준 개선|',
           '|---:|---|---|---|---:|---|']
    for e in events:
        a,b=e['validation']['by_count'][1:]
        lines.append(f"|{e['epoch']}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|"
                     f"{w.fmt(a['mean_si_sdr'],3)}/{w.fmt(b['mean_si_sdr'],3)}|"
                     f"{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{e['best']['epoch']}|{e['joint_success']}|")
    lines += ['', '[실행 규약](INTEGRATED_PLAN_KO.md). 시간 순서·TF·성분 보정·업데이트 보호를 결합했다. '
              '개별 구성의 기여나 성능 향상은 자동으로 입증되지 않는다.','']
    w.write(PUBLIC/'INTEGRATED_PROGRESS_KO.md','\n'.join(lines))


def train(root,p):
    baseline, ids = w.validate(Path(p['baseline']),None)
    dev = worker.NativeMixtures(p['preparation'],'validation_pack',1)
    events = [w.read(path) for path in sorted(root.glob('EPOCH_*.json'))]
    assert [e['epoch'] for e in events] == list(range(1,len(events)+1))
    net,opt,named = build(p['parent_checkpoint'])
    best = dict(epoch=0,metric=baseline['selection_nmse'])
    monitor = dict(bad=0, lr_bad=0, ever_joint=False)
    ph=w.digest(root/'PROTOCOL.json')
    if events:
        saved=torch.load(root/'LAST.pt',map_location='cpu',weights_only=False,mmap=True)
        assert w.digest(root/'LAST.pt')==events[-1]['last_sha256']
        assert saved['epoch']==len(events) and saved['protocol_sha256']==ph
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        best,monitor=saved['best'],saved['monitor']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    else:
        worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
    for epoch in range(len(events)+1,p['max_epochs']+1):
        verify(root,p)
        train_data=worker.NativeMixtures(p['preparation'],'train_pack',(epoch-1)%5+1)
        net.train(); guard=Accumulator(named);loss_sum=0.;counts=[0,0,0];details=[]
        started=time.time();torch.cuda.reset_peak_memory_stats()
        for index in range(2400):
            item=worker.fit.base.batch([train_data[index]])
            loss_sum+=backward_one(net,guard,item,32)
            counts[int(item['construction_count'])-1]+=1
            if (index+1)%32==0:
                details.append(apply_update(net,opt,named,guard))
                state(root,'TRAINING',epoch=epoch,updates_in_epoch=len(details),
                      cumulative_updates=75*(epoch-1)+len(details),examples=index+1,
                      mean_training_loss_so_far=loss_sum/(index+1),seconds=time.time()-started)
        elapsed=time.time()-started
        assert counts==[800,800,800] and len(details)==75
        assert {int(s['step']) for s in opt.state.values()}=={epoch*75}
        check_frozen(net,p['parent_checkpoint'])
        del guard,item;gc.collect();torch.cuda.empty_cache()
        state(root,'VALIDATING',epoch=epoch,updates=75*epoch)
        vp=root/f'VALIDATION_{epoch:03d}.json'
        validation.validate(net,dev,vp,epoch,worker.predict,worker)
        actual,_=w.validate(vp,ids)
        joint=joint_success(actual,baseline);monitor['ever_joint']|=joint
        improved=actual['selection_nmse']<best['metric']-1e-5
        monitor['bad']=0 if improved else monitor['bad']+1
        monitor['lr_bad']=0 if improved else monitor['lr_bad']+1
        if actual['selection_nmse']<best['metric']:
            best=dict(epoch=epoch,metric=actual['selection_nmse'])
            worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
        used=[g['lr'] for g in opt.param_groups]
        if monitor['lr_bad']>=3:
            for g in opt.param_groups:g['lr']*=.5
            monitor['lr_bad']=0
        worker.atomic_torch(root/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
            updates=75*epoch,best=best,monitor=monitor,protocol_sha256=ph,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        w.write(root/f'GUARD_{epoch:03d}.json',dict(updates=details,protocol_sha256=ph))
        event=dict(candidate='integrated',epoch=epoch,updates=75*epoch,validation=actual,best=best,
            joint_success=joint,monitor=dict(monitor),count_examples=counts,mean_training_loss=loss_sum/2400,
            train_seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(),schedule_epoch=(epoch-1)%5+1,
            rows_sha256=train_data.rows_hash,used_learning_rates=used,next_learning_rates=[g['lr'] for g in opt.param_groups],
            validation_sha256=w.digest(vp),last_sha256=w.digest(root/'LAST.pt'),best_sha256=w.digest(root/'BEST.pt'),
            protocol_sha256=ph,parent_parameters_unchanged=True,heldout_read=False)
        w.write(root/f'EPOCH_{epoch:03d}.json',event);publish(root,'EPOCH_COMPLETE')
        print(dict(event='EPOCH_COMPLETE',**event),flush=True)
        catastrophic=all(a['mean_nmse']>1.25*b['mean_nmse'] for a,b in zip(actual['by_count'][1:],baseline['by_count'][1:]))
        reason=('CATASTROPHIC_DEV_REGRESSION' if catastrophic else
                'NO_JOINT_WAVEFORM_GAIN_BY_E5' if epoch>=5 and not monitor['ever_joint'] else
                'FIVE_EPOCH_PLATEAU' if epoch>=5 and monitor['bad']>=5 else
                'MAXIMUM_BUDGET' if epoch==50 else None)
        if reason:
            state(root,'STOPPED_AT_REGISTERED_REVIEW',epoch=epoch,reason=reason,best=best)
            w.write(root/'COMPLETE.json',dict(status='BUDGET_OR_FUTILITY_REVIEW_COMPLETE',reason=reason,
                epochs=epoch,updates=epoch*75,best=best,protocol_sha256=ph,time=time.time(),heldout_read=False))
            publish(root,'STOPPED_AT_REGISTERED_REVIEW: '+reason)
            return
        del train_data,details;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();root=args.run.resolve()
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p=w.read(root/'PROTOCOL.json') if args.resume else register(root)
            assert not (root/'COMPLETE.json').exists(),'Finished run; register an amendment to extend'
            state(root,'WAITING_GPU_LOCK')
            with worker.fit.base.LOCK.open('r') as gpu:
                fcntl.flock(gpu,fcntl.LOCK_EX)
                verify(root,p);assert torch.cuda.is_available()
                torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                torch.manual_seed(0)
                if not (root/'PREFLIGHT.json').exists():preflight(root,p)
                assert w.read(root/'PREFLIGHT.json')['status']=='PASS'
                train(root,p)
        except Exception:
            w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time(),pid=os.getpid()))
            state(root,'FAILED');raise


if __name__=='__main__':main()
