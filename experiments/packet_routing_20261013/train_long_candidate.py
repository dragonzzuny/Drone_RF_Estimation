"""Explicitly register/run ONE conditional long-input candidate after diagnosis.

Nothing in this file auto-selects or queues a candidate. The entry point needs
a new output path and reviewed plan; its own protocol must be registered first.
The full-capacity backbone, data, source-normalized losses and update budget are
retained. Results compare against the previously trained shared/control paths.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
import numpy as np
import torch

from sequence_data import ROOT, OUT as SEQUENCE, Sequences, ROLES, read, sha256, write_json
from sequence_eval import short_eval, long_eval, summarize
from models import predict
from study import batch, atomic_torch
from long_training_core import load_model, predict_contiguous, train_bundle
from long_native_eval import evaluate as native_eval

HERE = Path(__file__).resolve().parent
PARENT = ROOT/'local/timeblock_extend100_20261012_v1'
LOCK = ROOT.parent/'uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock'
STOP = False


class Run:
    def __init__(self, output):
        self.output = output

    def state(self, status, **kw):
        write_json(self.output/'STATE.json', dict(status=status, pid=os.getpid(), time=time.time(), **kw))

    def register(self, architecture, plan):
        assert architecture in ('long_plain', 'long_gated') and plan.is_file()
        self.output.mkdir(parents=True, exist_ok=True)
        assert not (self.output/'PROTOCOL.json').exists()
        original = read(SEQUENCE/'PROTOCOL.json')
        sources = set(original['source_sha256'])
        for name in ('long_native_eval.py', 'long_training_core.py', 'train_long_candidate.py',
                     'long_temporal_candidate.py', 'checkpoint_blocks.py', 'test_long_objective.py',
                     'test_long_temporal_candidate.py', 'test_checkpoint_blocks.py'):
            sources.add(str((HERE/name).relative_to(ROOT)))
        sources.add(str(plan.resolve().relative_to(ROOT)))
        hashes = {name: sha256(ROOT/name) for name in sorted(sources)}
        final = read(PARENT/'EPOCH_100.json')
        audit = read(PARENT/'AUDIT_100.json')
        assert audit['status'] == 'PASS' and audit['event_sha256'] == sha256(PARENT/'EPOCH_100.json')
        assert final['best']['epoch'] == 99
        checkpoint = PARENT/'ACTUAL_099.pt'
        selected = read(PARENT/'EPOCH_099.json')
        assert sha256(checkpoint) == selected['checkpoint_sha256']
        protocol = dict(registered=time.time(), architecture=architecture, source_sha256=hashes,
            sequence_protocol_sha256=sha256(SEQUENCE/'PROTOCOL.json'),
            data_plan_sha256=sha256(SEQUENCE/'DATA_PLAN.json'),
            initialization=dict(epoch=99, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint)),
            epochs=3, updates_per_epoch=96, bundles_per_epoch=768, effective_bundles=8,
            samples_per_bundle=191616, parameters=32142859 if architecture=='long_plain' else 35820299,
            learning_rate=1e-5, weight_decay=1e-4, gradient_clip=1., seed=0,
            objective='Same three window-normalized original losses, one shared permutation, mean plus0.1 countCE',
            precision='FP32 TF32 disabled', activation_checkpointing=True,
            evaluation='Original short630 C8 and fixed merged long24; native long24 additional diagnostic',
            comparisons='Same parent and existing independent/shared final3epoch, no repeated control training',
            composite_change=architecture=='long_gated', early_performance_stop=False,
            no_heldout_access=True, independent_recording_validation=False)
        write_json(self.output/'PROTOCOL.json', protocol)
        for rel, digest in hashes.items():
            dst = self.output/'source_snapshot'/rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/rel, dst)
            assert sha256(dst) == digest
        self.state('REGISTERED_NOT_RUNNING', architecture=architecture)
        print('REGISTERED_LONG', architecture, sha256(self.output/'PROTOCOL.json'), flush=True)

    def verify(self, p):
        assert sha256(SEQUENCE/'PROTOCOL.json') == p['sequence_protocol_sha256']
        assert sha256(SEQUENCE/'DATA_PLAN.json') == p['data_plan_sha256']
        assert sha256(p['initialization']['checkpoint']) == p['initialization']['checkpoint_sha256']
        for rel, digest in p['source_sha256'].items():
            assert sha256(ROOT/rel) == digest, rel
            assert sha256(self.output/'source_snapshot'/rel) == digest, rel

    def preflight(self, p):
        receipt = self.output/'PREFLIGHT.json'
        if receipt.exists():
            assert read(receipt)['status'] == 'PASS'
            return
        self.state('GPU_PREFLIGHT', architecture=p['architecture'])
        net, saved = load_model(p['initialization']['checkpoint'], p['architecture'], 'cuda')
        del saved
        parent, saved = load_model(p['initialization']['checkpoint'], 'long_plain', 'cuda')
        del saved
        train = Sequences(ROLES[0], 1)
        indices = [next(i for i,r in enumerate(train.rows) if r['count']==k) for k in (1,2,3)]
        net.eval(); parent.eval(); differences = []
        with torch.no_grad():
            for i in indices:
                items = [batch(v, 'cuda') for v in train[i]]
                a, ca = predict_contiguous(net, predict, items)
                b, cb = predict_contiguous(parent, predict, items)
                maximum = max(float((x-y).abs().max()) for x,y in zip(a,b))
                for x,y in zip(a,b):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
                torch.testing.assert_close(ca, cb, rtol=0, atol=0)
                differences.append(maximum)
        del parent, a, b, ca, cb; gc.collect(); torch.cuda.empty_cache()
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=1e-4, foreach=False)
        rows = []
        for i in indices:
            opt.zero_grad(set_to_none=True)
            before = net.output.weight.detach().clone()
            losses, diag = train_bundle(net, predict, [batch(v, 'cuda') for v in train[i]], 1)
            norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
            assert any(q.grad is not None and bool(q.grad.abs().max()>0) for q in net.down.parameters())
            adapter_norm = float(net.gates.expand.weight.grad.norm()) if hasattr(net, 'gates') else None
            opt.step()
            assert bool((before-net.output.weight).abs().max()>0)
            rows.append(dict(count=int(train.rows[i]['count']), gradient_norm=norm,
                             adapter_projection_gradient=adapter_norm, loss=np.mean(losses,axis=0).tolist()))
        write_json(receipt, dict(status='PASS', rows=rows, weights_discarded=True,
            initial_full_input_difference=differences, max_gpu_allocated=torch.cuda.max_memory_allocated(),
            protocol_sha256=sha256(self.output/'PROTOCOL.json')))
        del net, opt, train, items; gc.collect(); torch.cuda.empty_cache()

    def save(self, net, opt, epoch, updates, next_bundle, losses, diagnostics, seconds, evaluated):
        atomic_torch(self.output/'RECOVERY.pt', dict(model=net.state_dict(), optimizer=opt.state_dict(),
            epoch=epoch, updates=updates, next_bundle=next_bundle, losses=losses, diagnostics=diagnostics,
            seconds=seconds, epoch_evaluated=evaluated, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), protocol_sha256=sha256(self.output/'PROTOCOL.json')))

    def audit(self, epoch):
        d=self.output; event=read(d/f'EPOCH_{epoch:03d}.json')
        for prefix,key in (('ACTUAL','checkpoint_sha256'),('SHORT','short_sha256'),
                           ('LONG','long_sha256'),('NATIVE_LONG','native_long_sha256')):
            assert sha256(d/f'{prefix}_{epoch:03d}.{"pt" if prefix=="ACTUAL" else "json"}') == event[key]
        short=read(d/f'SHORT_{epoch:03d}.json'); long=read(d/f'LONG_{epoch:03d}.json')
        native=read(d/f'NATIVE_LONG_{epoch:03d}.json')
        for x,n in ((short,630),(long['merged'],24),(native,24)):
            assert len(x['rows'])==n
            assert summarize(x['rows'])=={k:x[k] for k in ('by_count','selection_nmse')}
        saved=torch.load(d/'RECOVERY.pt',map_location='cpu',weights_only=False,mmap=True)
        assert saved['updates']==epoch*96 and saved['next_bundle']==768 and saved['epoch_evaluated']
        steps=[int(v['step']) for v in saved['optimizer']['state'].values()]
        assert steps and min(steps)==max(steps)==epoch*96
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
        write_json(d/f'AUDIT_{epoch:03d}.json', dict(status='PASS', epoch=epoch,
            event_sha256=sha256(d/f'EPOCH_{epoch:03d}.json'), optimizer_steps=epoch*96,
            checks=['source/row/checkpoint hashes','saved aggregates','optimizer updates','finite parameters']))

    def epoch(self, p, epoch):
        d=self.output
        if (d/f'AUDIT_{epoch:03d}.json').exists():
            assert read(d/f'AUDIT_{epoch:03d}.json')['status']=='PASS'
            return True
        recovery=d/'RECOVERY.pt'
        path=recovery if recovery.exists() else Path(p['initialization']['checkpoint'])
        net,saved=load_model(path,p['architecture'],'cuda')
        opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=1e-4,foreach=False)
        start,updates,losses,diagnostics,offset=0,(epoch-1)*96,[],[],0.
        if 'optimizer' in saved:
            assert saved['protocol_sha256']==sha256(d/'PROTOCOL.json')
            opt.load_state_dict(saved['optimizer']);updates=saved['updates']
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            if saved['epoch']==epoch and saved['epoch_evaluated']:
                del net,opt,saved;gc.collect();torch.cuda.empty_cache();self.audit(epoch)
                return True
            if saved['epoch']==epoch:
                start=saved['next_bundle'];losses=saved['losses'];diagnostics=saved['diagnostics'];offset=saved['seconds']
            else:
                assert saved['epoch']==epoch-1 and saved['epoch_evaluated']
        else:
            assert epoch==1
        del saved
        data=Sequences(ROLES[0],epoch);net.train();opt.zero_grad(set_to_none=True)
        assert start%8==0 and updates==(epoch-1)*96+start//8
        began=time.time()
        for i in range(start,len(data)):
            values,diag=train_bundle(net,predict,[batch(v,'cuda') for v in data[i]],8)
            losses.extend(values);diagnostics.append(diag)
            if (i+1)%8==0:
                norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
                opt.step();opt.zero_grad(set_to_none=True);updates+=1;elapsed=offset+time.time()-began
                if updates%12==0 or STOP:
                    self.save(net,opt,epoch,updates,i+1,losses,diagnostics,elapsed,False)
                self.state('TRAINING',architecture=p['architecture'],epoch=epoch,updates=updates,
                    bundles=i+1,total=len(data),seconds=elapsed,gradient_norm=norm,
                    recent_loss=np.mean(losses[-24:],axis=0).tolist())
                if STOP:
                    self.state('STOPPED_WITH_RECOVERY',epoch=epoch,updates=updates)
                    return False
        seconds=offset+time.time()-began
        assert updates==epoch*96 and len(losses)==2304 and len(diagnostics)==768
        progress=lambda s,**kw:self.state(s,architecture=p['architecture'],epoch=epoch,updates=updates,**kw)
        t=time.time();short=short_eval(net,'cuda',progress);long=long_eval(net,'cuda',progress)
        native=native_eval(net,'cuda',progress)
        for prefix,value in (('SHORT',short),('LONG',long),('NATIVE_LONG',native)):
            write_json(d/f'{prefix}_{epoch:03d}.json',value)
        cp=d/f'ACTUAL_{epoch:03d}.pt'
        atomic_torch(cp,dict(model=net.state_dict(),epoch=epoch,updates=updates,
                            architecture=p['architecture'],protocol_sha256=sha256(d/'PROTOCOL.json')))
        event=dict(epoch=epoch,architecture=p['architecture'],updates=updates,parameters=p['parameters'],
            train_seconds=seconds,evaluation_seconds=time.time()-t,
            short={k:v for k,v in short.items() if k!='rows'},
            long_merged={k:v for k,v in long['merged'].items() if k!='rows'},
            native_long={k:v for k,v in native.items() if k!='rows'},
            mean_training_loss=np.mean(losses,axis=0).tolist(),
            training_local_disagreement_fraction=float(np.mean([v['local_disagreement'] for v in diagnostics])),
            checkpoint_sha256=sha256(cp),short_sha256=sha256(d/f'SHORT_{epoch:03d}.json'),
            long_sha256=sha256(d/f'LONG_{epoch:03d}.json'),native_long_sha256=sha256(d/f'NATIVE_LONG_{epoch:03d}.json'),
            initial_epoch=99,protocol_sha256=sha256(d/'PROTOCOL.json'),independent_recording_validation=False)
        write_json(d/f'EPOCH_{epoch:03d}.json',event)
        self.save(net,opt,epoch,updates,768,losses,diagnostics,seconds,True)
        del net,opt,data;gc.collect();torch.cuda.empty_cache();self.audit(epoch)
        write_json(ROOT/'reports/2026-10-13'/f'{p["architecture"].upper()}_E{epoch:03d}.json',event)
        print('LONG_EPOCH_COMPLETE',epoch,json.dumps(event['short']),flush=True)
        return True

    def run(self):
        p=read(self.output/'PROTOCOL.json');self.verify(p)
        # Registration follows diagnosis; never jump ahead of the matched study.
        s=read(SEQUENCE/'STATE.json')
        assert s['status']=='COMPLETE_AUDITED',s
        with LOCK.open('r') as lock:
            self.state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
            self.verify(p);assert torch.cuda.is_available()
            torch.set_num_threads(2);torch.manual_seed(0)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cuda.matmul.allow_tf32=False
            self.preflight(p)
            if not (self.output/'INITIAL_NATIVE_LONG.json').exists():
                net,saved=load_model(p['initialization']['checkpoint'],p['architecture'],'cuda');del saved
                write_json(self.output/'INITIAL_NATIVE_LONG.json',native_eval(net,'cuda',lambda s,**kw:self.state(s,epoch=0,**kw)))
                del net;gc.collect();torch.cuda.empty_cache()
            for epoch in (1,2,3):
                self.verify(p)
                if STOP or not self.epoch(p,epoch):
                    return
            self.state('COMPLETE_AUDITED',architecture=p['architecture'],epochs=3,updates=288)


def stop(*_):
    global STOP
    STOP=True


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--architecture',choices=('long_plain','long_gated'))
    parser.add_argument('--plan',type=Path)
    a=parser.parse_args();runner=Run(a.output)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if a.action=='register':runner.register(a.architecture,a.plan)
        else:
            with (a.output/'.run.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runner.run()
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time(),pid=os.getpid()))
        runner.state('FAILED');raise
