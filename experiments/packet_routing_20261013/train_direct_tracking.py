"""Fixed50 direct discrete routing: global-PIT expected waveform NMSE."""
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
from evaluate_parent_full_context import ROOT,read,write,sha,LOCK
from tracking_cache import feature_input
from routing_statistics import score as cached_score
from spectral_routing_constrained import fit
from spectral_routing import PERMS
from diagnose_count_routing import summary

HERE=Path(__file__).resolve().parent
DEV=ROOT/'local/native_tracking_cache_20261013_v1'
STOP=False


def configure():
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False


def new_model():
    import torch
    from direct_tracking_model import DirectTracking as TrackingEmbedding
    torch.manual_seed(0);net=TrackingEmbedding()
    assert sum(p.numel() for p in net.parameters())==875905
    return net.cuda()


def load_train(items):
    data=[]
    for item in items:
        assert item['training']
        observed,_,_=feature_input(item)
        from direct_tracking_model import waveform_cost
        assert sha(item['scoring'])==item['scoring_sha256']
        with np.load(item['scoring'],allow_pickle=False) as z:cost=waveform_cost(z,item['count'])
        assert observed.shape==(33,3,513) and cost.shape==(33,6,6)
        data.append((observed,cost))
    return data


def training_batch(data,indices):
    import torch
    return [torch.from_numpy(np.stack([data[int(i)][j] for i in indices])).cuda() for j in range(2)]


def load_evaluation(items, training_diagnostic=False):
    cases=[]
    for item in items:
        assert item['training'] == training_diagnostic
        if not training_diagnostic: assert 'labels' not in item
        observed,powers,bounds=feature_input(item)
        assert sha(item['scoring'])==item['scoring_sha256']
        with np.load(item['scoring'],allow_pickle=False) as z:statistics={k:z[k].copy() for k in z.files}
        cases.append(dict(item=item,observed=observed,powers=powers,bounds=bounds,statistics=statistics))
    return cases


def evaluate(net,cases):
    import torch
    net.eval();rows=[];receipts=[]
    with torch.no_grad():
        for cached in cases:
            item=cached['item']
            # Only observed-only features enter the network. Counts, category
            # names and all reference statistics are deliberately inaccessible
            # through its forward signature.
            from direct_tracking_model import decode
            logits=net(torch.from_numpy(cached['observed'])[None].cuda())[0]
            orders,receipt=decode(logits)
            metrics=cached_score(cached['statistics'],orders,item['count'],item['predicted_count'])
            assert metrics['sum_relative_error']<1e-9
            rows.append(dict(index=item['index'],categories=item['metadata']['categories'],
                nominal_levels_db=item['metadata']['nominal_levels_db'],**metrics))
            receipts.append(dict(index=item['index'],orders=orders.tolist(),**receipt))
    groups={str(k):summary([r for r in rows if r['count']==k]) for k in (1,2,3)}
    return dict(rows=rows,by_count=groups,selection_nmse=float(np.mean([groups[str(k)]['mean_nmse'] for k in (2,3)])),
                routing_receipts=receipts)


class Training:
    def __init__(self,out):self.out=out
    def state(self,status,**kw):write(self.out/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**kw))

    def register(self,teacher,plan):
        teacher=teacher.resolve();p=read(teacher/'PLAN.json')
        assert p['cases']==48 and p['training_role']=='fit_block'
        assert read(DEV/'STATE.json')['status']=='COMPLETE_AUDITED'
        assert plan.is_file() and not (self.out/'PROTOCOL.json').exists()
        check=read(ROOT/'reports/2026-10-13/DIRECT_TRACKING_CPU_CHECK.json');assert check['status']=='PASS'
        assert check['source_sha256']==sha(HERE/'direct_tracking_model.py')
        sources=dict(p['source_sha256'])
        for name in ('train_direct_tracking.py','direct_tracking_model.py','test_direct_tracking.py','learned_tracking_model.py','test_learned_tracking_model.py',
                     'spectral_routing.py','spectral_routing_constrained.py','tracking_cache.py','routing_statistics.py',
                     'audit_direct_tracking.py','cache_native_tracking.py','numpy_waveform_score.py','native_full_context.py'):
            sources[str((HERE/name).relative_to(ROOT))]=sha(HERE/name)
        for name,h in read(DEV/'RESULT.json')['source_sha256'].items():
            if name in sources: assert sources[name]==h
            sources[name]=h
        sources[str(plan.resolve().relative_to(ROOT))]=sha(plan)
        files={str(q):sha(q) for q in (teacher/'PLAN.json',DEV/'RESULT.json',DEV/'AUDIT.json')}
        protocol=dict(registered=time.time(),source_sha256=sources,input_sha256=files,teacher=str(teacher),
            epochs=50,updates_per_epoch=12,batch_size=4,training_cases=48,learning_rate=.00025,
            weight_decay=.0001,gradient_clip=1.,seed=0,parameters=875905,
            optimizer='AdamW fresh; no scheduler',precision='FP32 TF32 disabled',
            objective='Global-PIT expected categorical block-permutation waveform NMSE +0.1 inactive output power; softmax temperature1',
            waveform_model_frozen=True,teacher_epoch=2,early_stopping=False,final_epoch_fixed=50,
            FIT48_every_epoch=True,B24_evaluated=False,C24_final_only=True,no_heldout_access=True,
            no_device_identity_or_count_in_encoder=True,predicted_count_used_in_routing=False,extra_head_parameters=384,reference_cost_in_training_only=True,
            references_in_training_labels_and_scoring_only=True)
        self.out.mkdir(exist_ok=True);write(self.out/'PROTOCOL.json',protocol)
        for name,h in sources.items():
            dest=self.out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dest);assert sha(dest)==h
        self.state('REGISTERED_NOT_RUNNING')

    def verify(self,p):
        assert all(sha(ROOT/name)==h and sha(self.out/'source_snapshot'/name)==h for name,h in p['source_sha256'].items())
        assert all(sha(name)==h for name,h in p['input_sha256'].items())

    def preflight(self,p,data,teacher):
        import torch
        from direct_tracking_model import expected_waveform_loss as affinity_loss
        path=self.out/'PREFLIGHT.json'
        if path.exists():assert read(path)['status']=='PASS';return
        self.state('GPU_PREFLIGHT');net=new_model();net.train()
        index=next(i for i,r in enumerate(teacher['cases']) if r['count']==3)
        x,y=training_batch(data,[index]);opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
        losses=[]
        for _ in range(16):
            opt.zero_grad(set_to_none=True);loss=affinity_loss(net(x),y);losses.append(float(loss.detach()))
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        with torch.no_grad():last=float(affinity_loss(net(x),y))
        assert last<losses[0],(losses,last)
        write(path,dict(status='PASS',initial_expected_waveform_loss=losses[0],final_expected_waveform_loss=last,losses=losses,
            discarded_updates=16,updates_reused=0,parameters=875905,case_index=teacher['cases'][index]['index'],
            max_gpu_allocated=torch.cuda.max_memory_allocated(),protocol_sha256=sha(self.out/'PROTOCOL.json')))
        del net,opt,x,y;gc.collect();torch.cuda.empty_cache()

    def run(self):
        p=read(self.out/'PROTOCOL.json');teacher_dir=Path(p['teacher']);self.verify(p)
        while True:
            s=read(teacher_dir/'STATE.json')
            if s['status']=='COMPLETE_AUDITED':break
            if s['status']=='FAILED':raise RuntimeError('FIT teacher preparation failed')
            self.state('WAITING_FIT_TEACHER',previous=s['status'],case=s.get('case'));time.sleep(15)
        assert sha(teacher_dir/'RESULT.json')==read(teacher_dir/'AUDIT.json')['result_sha256']
        dependency=dict(teacher_result_sha256=sha(teacher_dir/'RESULT.json'),teacher_audit_sha256=sha(teacher_dir/'AUDIT.json'))
        write(self.out/'DEPENDENCY.json',dependency)
        with LOCK.open('r') as lock:
            self.state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);self.verify(p);configure()
            import torch
            from direct_tracking_model import expected_waveform_loss as affinity_loss
            from study import atomic_torch
            teacher=read(teacher_dir/'RESULT.json');assert teacher['training_role']=='fit_block'
            data=load_train(teacher['cases']);dev=read(DEV/'RESULT.json');fitcases=load_evaluation(teacher['cases'],training_diagnostic=True)
            self.preflight(p,data,teacher)
            net=new_model();opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],foreach=False)
            completed=0;recovery=self.out/'RECOVERY.pt'
            if recovery.exists():
                saved=torch.load(recovery,map_location='cuda',weights_only=False)
                assert saved['protocol_sha256']==sha(self.out/'PROTOCOL.json')
                net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);completed=saved['epoch']
                assert saved['updates']==completed*12;del saved
            if not (self.out/'FIT_000.json').exists():
                self.state('FIT_TRAINING_DIAGNOSTIC',epoch=0);write(self.out/'FIT_000.json',evaluate(net,fitcases))
            for epoch in range(completed+1,51):
                if STOP:self.state('STOPPED_AFTER_EPOCH',epoch=epoch-1);return
                self.verify(p);net.train();started=time.time();losses=[]
                order=np.random.default_rng(np.random.SeedSequence([0,101350,epoch])).permutation(48)
                for start in range(0,48,4):
                    x,y=training_batch(data,order[start:start+4]);opt.zero_grad(set_to_none=True)
                    loss=affinity_loss(net(x),y);loss.backward()
                    norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
                    opt.step();losses.append(float(loss.detach()))
                    self.state('TRAINING',epoch=epoch,updates=(epoch-1)*12+start//4+1,gradient_norm=norm)
                seconds=time.time()-started;self.state('FIT_TRAINING_DIAGNOSTIC',epoch=epoch,updates=epoch*12)
                result=evaluate(net,fitcases);write(self.out/f'FIT_{epoch:03d}.json',result)
                cp=self.out/f'ACTUAL_{epoch:03d}.pt'
                atomic_torch(cp,dict(model=net.state_dict(),epoch=epoch,updates=epoch*12,protocol_sha256=sha(self.out/'PROTOCOL.json')))
                atomic_torch(recovery,dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                    updates=epoch*12,protocol_sha256=sha(self.out/'PROTOCOL.json')))
                steps=[int(v['step']) for v in opt.state.values()];assert steps and min(steps)==max(steps)==epoch*12
                assert all(torch.isfinite(v).all() for v in net.state_dict().values())
                event=dict(epoch=epoch,updates=epoch*12,training_loss=float(np.mean(losses)),training_seconds=seconds,
                    evaluation_scope='FIT48 training diagnostic, not validation',evaluation_seconds=time.time()-started-seconds,by_count=result['by_count'],selection_nmse=result['selection_nmse'],
                    checkpoint_sha256=sha(cp),FIT_result_sha256=sha(self.out/f'FIT_{epoch:03d}.json'),
                    protocol_sha256=sha(self.out/'PROTOCOL.json'),fixed_final_epoch=50,C_evaluated=False)
                write(self.out/f'EPOCH_{epoch:03d}.json',event)
                saved=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
                assert all(torch.equal(v.detach().cpu(),saved['model'][name]) for name,v in net.state_dict().items())
                del saved
                write(self.out/f'AUDIT_{epoch:03d}.json',dict(status='PASS',event_sha256=sha(self.out/f'EPOCH_{epoch:03d}.json'),
                    optimizer_steps=epoch*12,saved_weights_exact=True,source_input_hashes=True))
                write(ROOT/'reports/2026-10-13'/f'DIRECT_TRACKING_E{epoch:03d}.json',event)
                print('TRACKING_EPOCH_COMPLETE',epoch,result['by_count'],flush=True)
            self.verify(p);self.state('C_DEVELOPMENT_FINAL',epoch=50)
            ccases=load_evaluation(dev['groups']['C24']);cresult=evaluate(net,ccases);write(self.out/'C_050.json',cresult)
            result=dict(status='VERIFIED',epoch=50,FIT=read(self.out/'FIT_050.json'),C=cresult,
                parameters=875905,optimizer_updates=600,teacher_updates=0,
                protocol_sha256=sha(self.out/'PROTOCOL.json'),dependency_sha256=sha(self.out/'DEPENDENCY.json'),
                inference_reference_free=True,scope='Direct-routing expected waveform objective; FIT48 training only; exposed C24 final50; native e2 frozen')
            write(self.out/'RESULT.json',result);write(self.out/'AUDIT.json',dict(status='PASS',result_sha256=sha(self.out/'RESULT.json'),
                final_epoch50=True,optimizer_updates=600,source_input_hashes=True))
            write(ROOT/'reports/2026-10-13/DIRECT_TRACKING_RESULT.json',result)
            self.state('COMPLETE_AUDITED',epochs=50,updates=600)


def stop(*_):
    global STOP
    STOP=True


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','run'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--teacher',type=Path);parser.add_argument('--plan',type=Path)
    a=parser.parse_args();obj=Training(a.output);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if a.action=='register':obj.register(a.teacher,a.plan)
        else:obj.run()
    except BaseException:
        obj.out.mkdir(exist_ok=True);write(obj.out/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        obj.state('FAILED');raise
