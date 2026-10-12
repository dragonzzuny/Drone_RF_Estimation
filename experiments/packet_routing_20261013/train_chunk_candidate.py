"""Conditional full-capacity cross-window model; explicit registration required.

Preparation is not a queued experiment. Reuses the completed matched study's
data and budget, but adds waveform-feature connections between neighbouring
windows. This is a composite capacity/context change, not a pure PIT ablation.
"""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
import numpy as np
import torch

from sequence_data import ROOT, OUT as SEQUENCE, Sequences, ROLES, read, sha256, write_json
from sequence_eval import summarize, predict_eight
from sequence_objective import assert_replay_safe, PERMUTATIONS
from models import build, predict
from study import batch, atomic_torch
from long_training_core import contiguous_loss
from checkpoint_blocks import enable_checkpointing
from chunk_temporal_candidate import ChunkTemporalUNet, predict_bundle, predict_primary
from chunk_evaluation import short_eval, long_eval, Mixtures, add_neighbours
from dual_context_candidate import DualContextUNet, ADDED_PREFIXES

HERE=Path(__file__).resolve().parent
PARENT=ROOT/'local/timeblock_extend100_20261012_v1'
LOCK=ROOT.parent/'uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock'
STOP=False
ARCHITECTURES={'chunk_temporal':(ChunkTemporalUNet,35820299,3677440),
               'dual_context':(DualContextUNet,36554507,4411648)}


def load(path, device, architecture, scope):
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    factory,total,added=ARCHITECTURES[architecture]
    torch.manual_seed(0);net=factory()
    if any(k.startswith('gates.') for k in saved['model']):net.load_state_dict(saved['model'])
    else:net.initialize_parent(saved['model'])
    assert sum(p.numel() for p in net.parameters())==total
    if scope not in ('all','added'):raise ValueError('Explicit all/added optimization scope required')
    for name,p in net.named_parameters():p.requires_grad_(scope=='all' or name.startswith(ADDED_PREFIXES))
    assert sum(p.numel() for p in net.parameters() if p.requires_grad)==(total if scope=='all' else added)
    assert_replay_safe(net);enable_checkpointing(net)
    return net.to(device),saved


def step_bundle(net,items,divisor):
    predictions,logits=predict_bundle(net,items)
    loss,values,diagnostic=contiguous_loss(predictions,logits,items)
    (loss/divisor).backward()
    count=int(items[0]['construction_count'][0])
    diagnostic=dict(diagnostic,construction_count=count,
        active_order_disagreement=len({PERMUTATIONS[j][:count] for j in diagnostic['local_indices']})>1)
    return values,diagnostic


class Run:
    def __init__(self,output):self.output=output

    def state(self,status,**kw):
        write_json(self.output/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**kw))

    def register(self,plan,scope,architecture,learning_rate):
        assert plan.is_file()
        assert scope in ('all','added') and architecture in ARCHITECTURES
        assert learning_rate is not None and 0 < learning_rate <= 1e-3
        _,total,added=ARCHITECTURES[architecture]
        # The next choice is made after all registered comparisons, not queued
        # in advance of evidence. This guard does not claim their success.
        dependencies=(SEQUENCE,ROOT/'local/b99_on_c_20261013_v1',ROOT/'local/sequence_packets_20261013_v1',
                      ROOT/'local/full_context_parent_20261013_v1')
        assert all(read(d/'STATE.json')['status']=='COMPLETE_AUDITED' for d in dependencies)
        self.output.mkdir(parents=True,exist_ok=True)
        assert not (self.output/'PROTOCOL.json').exists()
        sources=set(read(SEQUENCE/'PROTOCOL.json')['source_sha256'])
        for name in ('train_chunk_candidate.py','chunk_temporal_candidate.py','chunk_evaluation.py',
                     'long_temporal_candidate.py','long_training_core.py','checkpoint_blocks.py',
                     'test_chunk_temporal_candidate.py','test_chunk_observations.py','test_chunk_frozen.py','test_long_objective.py',
                     'dual_context_candidate.py','test_dual_context.py','test_selective_decode.py'):
            sources.add(str((HERE/name).relative_to(ROOT)))
        sources.add(str(plan.resolve().relative_to(ROOT)))
        hashes={name:sha256(ROOT/name) for name in sorted(sources)}
        parent=read(PARENT/'EPOCH_100.json');audit=read(PARENT/'AUDIT_100.json')
        assert audit['status']=='PASS' and audit['event_sha256']==sha256(PARENT/'EPOCH_100.json')
        assert parent['best']['epoch']==99
        checkpoint=PARENT/'ACTUAL_099.pt'
        assert sha256(checkpoint)==read(PARENT/'EPOCH_099.json')['checkpoint_sha256']
        protocol=dict(registered=time.time(),architecture=architecture,source_sha256=hashes,
            sequence_protocol_sha256=sha256(SEQUENCE/'PROTOCOL.json'),
            data_plan_sha256=sha256(SEQUENCE/'DATA_PLAN.json'),
            initialization=dict(epoch=99,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint)),
            epochs=3,updates_per_epoch=96,bundles_per_epoch=768,effective_bundles=8,
            samples_per_bundle=191616,parameters=total,optimization_scope=scope,
            trainable_parameters=total if scope=='all' else added,
            learning_rate=learning_rate,weight_decay=1e-4,gradient_clip=1.,seed=0,
            objective='Mean of three original window-normalized losses, shared permutation, plus0.1 countCE',
            precision='FP32 TF32 disabled',activation_checkpointing=True,
            evaluation='Original630 short targets with observed neighbours; fixed24 long targets; C8 primary-only alignment',
            comparisons='Same parent99 and completed independent/shared three-epoch trajectories; no new control training',
            composite_capacity_context_change=True,early_performance_stop=False,
            no_heldout_access=True,independent_recording_validation=False)
        write_json(self.output/'PROTOCOL.json',protocol)
        for rel,digest in hashes.items():
            dst=self.output/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/rel,dst);assert sha256(dst)==digest
        self.state('REGISTERED_NOT_RUNNING')
        print('REGISTERED_CHUNK',sha256(self.output/'PROTOCOL.json'),flush=True)

    def verify(self,p):
        assert sha256(SEQUENCE/'PROTOCOL.json')==p['sequence_protocol_sha256']
        assert sha256(SEQUENCE/'DATA_PLAN.json')==p['data_plan_sha256']
        assert sha256(p['initialization']['checkpoint'])==p['initialization']['checkpoint_sha256']
        for rel,digest in p['source_sha256'].items():
            assert sha256(ROOT/rel)==digest,rel
            assert sha256(self.output/'source_snapshot'/rel)==digest,rel

    def preflight(self,p):
        path=self.output/'PREFLIGHT.json'
        if path.exists():assert read(path)['status']=='PASS';return
        self.state('GPU_PREFLIGHT')
        net,saved=load(p['initialization']['checkpoint'],'cuda',p['architecture'],p['optimization_scope'])
        parent=build('unet_mean').to('cuda');parent.load_state_dict(saved['model']);del saved
        net.eval();parent.eval();train=Sequences(ROLES[0],1)
        indices=[next(i for i,r in enumerate(train.rows) if r['count']==k) for k in (1,2,3)]
        differences=[]
        with torch.no_grad():
            for i in indices:
                items=[batch(v,'cuda') for v in train[i]];ys,logits=predict_bundle(net,items)
                for y,item in zip(ys,items):
                    expected,count=predict(parent,item)
                    torch.testing.assert_close(y,expected,rtol=0,atol=0)
                    torch.testing.assert_close(logits,count,rtol=0,atol=0)
                    differences.append(float((y-expected).abs().max()))
            dev=Mixtures(ROLES[1]);phase_checks=[]
            for k in (1,2,3):
                i=next(i for i,r in enumerate(dev.rows) if r['count']==k)
                item=add_neighbours(dev,i,dev[i],'cuda')
                _,new,count,_=predict_eight(net,predict_primary,item)
                _,old,count0,_=predict_eight(parent,predict,item)
                torch.testing.assert_close(new,old,rtol=0,atol=0)
                torch.testing.assert_close(count,count0,rtol=0,atol=0)
                phase_checks.append(dict(count=k,max_difference=float((new-old).abs().max())))
        del parent,ys,expected,new,old,items,dev;gc.collect();torch.cuda.empty_cache()
        net.train();opt=torch.optim.AdamW([q for q in net.parameters() if q.requires_grad],
            lr=p['learning_rate'],weight_decay=1e-4,foreach=False);rows=[]
        for i in indices:
            opt.zero_grad(set_to_none=True);before=net.gates.expand.weight.detach().clone()
            values,diagnostic=step_bundle(net,[batch(v,'cuda') for v in train[i]],1)
            norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
            gate_gradient=float(net.gates.expand.weight.grad.norm());assert gate_gradient>0
            if p['optimization_scope']=='all':
                assert any(q.grad is not None and bool(q.grad.abs().max()>0) for q in net.down.parameters())
            else:assert all(q.grad is None for name,q in net.named_parameters() if not name.startswith(ADDED_PREFIXES))
            ordered_norm=float(net.ordered_projection.weight.grad.norm()) if hasattr(net,'ordered_projection') else None
            opt.step();assert bool((before-net.gates.expand.weight).abs().max()>0)
            rows.append(dict(count=int(train.rows[i]['count']),gradient_norm=norm,
                             adapter_projection_gradient=gate_gradient,ordered_projection_gradient=ordered_norm,
                             loss=np.mean(values,axis=0).tolist()))
        if p['optimization_scope']=='added':
            original=torch.load(p['initialization']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)['model']
            for name,value in net.state_dict().items():
                if not name.startswith(ADDED_PREFIXES):assert torch.equal(value.cpu(),original[name]),name
            del original
        write_json(path,dict(status='PASS',rows=rows,initial_parent_differences=differences,
            primary_c8_checks=phase_checks,weights_discarded=True,updates_reused_in_experiment=0,
            max_gpu_allocated=torch.cuda.max_memory_allocated(),protocol_sha256=sha256(self.output/'PROTOCOL.json')))
        del net,opt,train;gc.collect();torch.cuda.empty_cache()

    def save(self,net,opt,epoch,updates,next_bundle,losses,diagnostics,seconds,evaluated):
        atomic_torch(self.output/'RECOVERY.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),
            epoch=epoch,updates=updates,next_bundle=next_bundle,losses=losses,diagnostics=diagnostics,
            seconds=seconds,epoch_evaluated=evaluated,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),protocol_sha256=sha256(self.output/'PROTOCOL.json')))

    def audit(self,epoch):
        d=self.output;event=read(d/f'EPOCH_{epoch:03d}.json');p=read(d/'PROTOCOL.json')
        self.verify(p)
        for prefix,key in (('ACTUAL','checkpoint_sha256'),('SHORT','short_sha256'),('LONG','long_sha256')):
            assert sha256(d/f'{prefix}_{epoch:03d}.{"pt" if prefix=="ACTUAL" else "json"}')==event[key]
        for value,n in ((read(d/f'SHORT_{epoch:03d}.json'),630),(read(d/f'LONG_{epoch:03d}.json')['merged'],24)):
            assert len(value['rows'])==n and summarize(value['rows'])=={k:value[k] for k in ('by_count','selection_nmse')}
        saved=torch.load(d/'RECOVERY.pt',map_location='cpu',weights_only=False,mmap=True)
        actual=torch.load(d/f'ACTUAL_{epoch:03d}.pt',map_location='cpu',weights_only=False,mmap=True)
        assert set(actual['model'])==set(saved['model'])
        assert all(torch.equal(value,actual['model'][name]) for name,value in saved['model'].items())
        assert saved['updates']==epoch*96 and saved['next_bundle']==768 and saved['epoch_evaluated']
        steps=[int(v['step']) for v in saved['optimizer']['state'].values()]
        optimized=[name for name in saved['model'] if p['optimization_scope']=='all' or name.startswith(ADDED_PREFIXES)]
        assert len(steps)==len(optimized) and min(steps)==max(steps)==epoch*96
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
        frozen=0
        if p['optimization_scope']=='added':
            original=torch.load(p['initialization']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)['model']
            for name,value in saved['model'].items():
                if not name.startswith(ADDED_PREFIXES):
                    assert torch.equal(value,original[name]),name
                    frozen+=1
        write_json(d/f'AUDIT_{epoch:03d}.json',dict(status='PASS',epoch=epoch,
            event_sha256=sha256(d/f'EPOCH_{epoch:03d}.json'),optimizer_steps=epoch*96,frozen_tensors_verified=frozen,
            checks=['Pinned sources and checkpoint hashes','Saved row aggregates','All parameter optimizer steps','Finite weights']))

    def epoch(self,p,epoch):
        d=self.output
        if (d/f'AUDIT_{epoch:03d}.json').exists():
            assert read(d/f'AUDIT_{epoch:03d}.json')['status']=='PASS';return True
        recovery=d/'RECOVERY.pt';net,saved=load(recovery if recovery.exists() else p['initialization']['checkpoint'],
                                             'cuda',p['architecture'],p['optimization_scope'])
        opt=torch.optim.AdamW([q for q in net.parameters() if q.requires_grad],
            lr=p['learning_rate'],weight_decay=1e-4,foreach=False)
        start,updates,losses,diagnostics,offset=0,(epoch-1)*96,[],[],0.
        if 'optimizer' in saved:
            assert saved['protocol_sha256']==sha256(d/'PROTOCOL.json')
            opt.load_state_dict(saved['optimizer']);updates=saved['updates']
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            if saved['epoch']==epoch and saved['epoch_evaluated']:
                del net,opt,saved;gc.collect();torch.cuda.empty_cache();self.audit(epoch);return True
            if saved['epoch']==epoch:
                start=saved['next_bundle'];losses=saved['losses'];diagnostics=saved['diagnostics'];offset=saved['seconds']
            else:assert saved['epoch']==epoch-1 and saved['epoch_evaluated']
        else:assert epoch==1
        del saved
        data=Sequences(ROLES[0],epoch);net.train();opt.zero_grad(set_to_none=True)
        assert start%8==0 and updates==(epoch-1)*96+start//8
        began=time.time()
        for i in range(start,len(data)):
            values,diag=step_bundle(net,[batch(v,'cuda') for v in data[i]],8)
            losses.extend(values);diagnostics.append(diag)
            if (i+1)%8==0:
                norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
                opt.step();opt.zero_grad(set_to_none=True);updates+=1;elapsed=offset+time.time()-began
                if updates%12==0 or STOP:self.save(net,opt,epoch,updates,i+1,losses,diagnostics,elapsed,False)
                self.state('TRAINING',epoch=epoch,updates=updates,bundles=i+1,total=len(data),seconds=elapsed,
                    gradient_norm=norm,recent_loss=np.mean(losses[-24:],axis=0).tolist())
                if STOP:self.state('STOPPED_WITH_RECOVERY',epoch=epoch,updates=updates);return False
        seconds=offset+time.time()-began
        assert updates==epoch*96 and len(losses)==2304 and len(diagnostics)==768
        progress=lambda s,**kw:self.state(s,epoch=epoch,updates=updates,**kw)
        t=time.time();short=short_eval(net,'cuda',progress);long=long_eval(net,'cuda',progress)
        write_json(d/f'SHORT_{epoch:03d}.json',short);write_json(d/f'LONG_{epoch:03d}.json',long)
        cp=d/f'ACTUAL_{epoch:03d}.pt'
        atomic_torch(cp,dict(model=net.state_dict(),epoch=epoch,updates=updates,architecture=p['architecture'],
                            protocol_sha256=sha256(d/'PROTOCOL.json')))
        event=dict(epoch=epoch,architecture=p['architecture'],updates=updates,parameters=p['parameters'],
            train_seconds=seconds,evaluation_seconds=time.time()-t,optimization_scope=p['optimization_scope'],
            trainable_parameters=p['trainable_parameters'],
            short={k:v for k,v in short.items() if k!='rows'},
            long_merged={k:v for k,v in long['merged'].items() if k!='rows'},
            mean_training_loss=np.mean(losses,axis=0).tolist(),
            training_local_disagreement_fraction=float(np.mean([v['local_disagreement'] for v in diagnostics])),
            training_active_order_disagreement_fraction=float(np.mean([v['active_order_disagreement'] for v in diagnostics])),
            disagreement_definition='Active-order statistic excludes swaps solely between inactive slots; full-permutation statistic retained separately',
            checkpoint_sha256=sha256(cp),short_sha256=sha256(d/f'SHORT_{epoch:03d}.json'),
            long_sha256=sha256(d/f'LONG_{epoch:03d}.json'),initial_epoch=99,
            protocol_sha256=sha256(d/'PROTOCOL.json'),independent_recording_validation=False)
        write_json(d/f'EPOCH_{epoch:03d}.json',event);self.save(net,opt,epoch,updates,768,losses,diagnostics,seconds,True)
        del net,opt,data;gc.collect();torch.cuda.empty_cache();self.audit(epoch)
        write_json(ROOT/'reports/2026-10-13'/f'{p["architecture"].upper()}_E{epoch:03d}.json',event)
        print('CHUNK_EPOCH_COMPLETE',epoch,event['short'],flush=True);return True

    def run(self):
        p=read(self.output/'PROTOCOL.json');self.verify(p)
        with LOCK.open('r') as lock:
            self.state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
            self.verify(p);assert torch.cuda.is_available();torch.set_num_threads(2);torch.manual_seed(0)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            self.preflight(p)
            for epoch in (1,2,3):
                self.verify(p)
                if STOP or not self.epoch(p,epoch):return
            self.state('COMPLETE_AUDITED',epochs=3,updates=288)


def stop(*_):
    global STOP
    STOP=True


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan',type=Path)
    parser.add_argument('--scope',choices=('all','added'))
    parser.add_argument('--architecture',choices=tuple(ARCHITECTURES))
    parser.add_argument('--learning-rate',type=float)
    a=parser.parse_args();runner=Run(a.output)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if a.action=='register':runner.register(a.plan,a.scope,a.architecture,a.learning_rate)
        else:
            with (a.output/'.run.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runner.run()
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time(),pid=os.getpid()))
        runner.state('FAILED');raise
