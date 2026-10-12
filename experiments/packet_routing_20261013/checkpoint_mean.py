"""One predeclared equal average of parent96..100, zero optimizer updates."""
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'local/timeblock_extend100_20261012_v1'
OUT=ROOT/'local/checkpoint_mean_20261013_v1'
LOCK=ROOT.parent/'uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock'
PLAN=ROOT/'reports/2026-10-13/CHECKPOINT_MEAN_PLAN_KO.md'
EPOCHS=tuple(range(96,101))


def read(p):return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**kw))


def run():
    OUT.mkdir(exist_ok=True);assert not (OUT/'PROTOCOL.json').exists()
    hashes={str(ROOT/p):h for p,h in read(PARENT/'PROTOCOL.json')['source_sha256'].items()}
    for p in [Path(__file__),PLAN,Path(__file__).with_name('sequence_eval.py'),
              Path(__file__).with_name('sequence_data.py'),Path(__file__).with_name('sequence_objective.py')]:hashes[str(p)]=sha(p)
    protocol=dict(registered=time.time(),source_sha256=hashes,epochs=list(EPOCHS),weight_each=.2,
        optimizer_updates=0,parameters=32142859,accumulation='float64 then cast to originalfloat32',
        inference='single averaged model, unchanged C8',comparisons='Original selected parent and saved97, B DEV630 and fixedlong24',
        no_mean_window_search=True,no_heldout=True,changes_sequence_initialization=False)
    write(OUT/'PROTOCOL.json',protocol)
    with LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
        s=read(PARENT/'STATE.json');assert s['status']=='COMPLETE_AUDITED' and s['epoch']==100
        assert all(sha(p)==h for p,h in hashes.items())
        import torch
        from sequence_eval import short_eval,long_eval,summarize
        from models import build
        from study import atomic_torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.manual_seed(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        average={};sources=[];dtypes={}
        for e in EPOCHS:
            event=read(PARENT/f'EPOCH_{e:03d}.json');audit=read(PARENT/f'AUDIT_{e:03d}.json')
            assert audit['status']=='PASS' and audit['event_sha256']==sha(PARENT/f'EPOCH_{e:03d}.json')
            cp=PARENT/f'ACTUAL_{e:03d}.pt';assert sha(cp)==event['checkpoint_sha256']
            saved=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)['model']
            if average:assert list(average)==list(saved)
            for key,tensor in saved.items():
                assert tensor.dtype==torch.float32 and torch.isfinite(tensor).all()
                if key not in average:average[key]=torch.zeros_like(tensor,dtype=torch.float64);dtypes[key]=tensor.dtype
                average[key].add_(tensor.double())
            sources.append(dict(epoch=e,checkpoint_sha256=event['checkpoint_sha256']))
            del saved;gc.collect();state('AVERAGING',epoch=e)
        for key in average:average[key]=(average[key]/len(EPOCHS)).to(dtypes[key])
        net=build('unet_mean').cuda();net.load_state_dict(average)
        assert sum(p.numel() for p in net.parameters())==32142859
        assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in net.modules())
        cp=OUT/'ACTUAL.pt';atomic_torch(cp,dict(model=average,source_epochs=EPOCHS,
            optimizer_updates=0,protocol_sha256=sha(OUT/'PROTOCOL.json')))
        del average;gc.collect()
        began=time.time();short=short_eval(net,'cuda',lambda status,**kw:state(status,**kw))
        long=long_eval(net,'cuda',lambda status,**kw:state(status,**kw))
        write(OUT/'SHORT.json',short);write(OUT/'LONG.json',long)
        selected=read(PARENT/'EPOCH_100.json')['best']['epoch'];parent_dir=PARENT if selected>50 else ROOT/'local/bandwidth_timeblock_20261011_v1'
        base=read(parent_dir/f'EPOCH_{selected:03d}.json')['validation'];old=read(PARENT/'EPOCH_097.json')['validation']
        def dominates(reference):
            return all(x['mean_nmse']<y['mean_nmse'] and x['weakest_nmse']<y['weakest_nmse'] and x['mean_si_sdr']>y['mean_si_sdr']
                       for x,y in zip(short['by_count'][1:],reference['by_count'][1:]))
        result=dict(status='COMPLETE',sources=sources,checkpoint_sha256=sha(cp),
            short={k:v for k,v in short.items() if k!='rows'},long_merged={k:v for k,v in long['merged'].items() if k!='rows'},
            selected_parent_epoch=selected,parent=base,epoch97=old,
            all_six_metrics_better_than_selected_parent=dominates(base),all_six_metrics_better_than_epoch97=dominates(old),
            evaluation_seconds=time.time()-began,optimizer_updates=0,independent_recording_validation=False,
            current_best_manifest_changed=False)
        write(OUT/'RESULT.json',result)
        assert summarize(short['rows'])=={k:short[k] for k in ('by_count','selection_nmse')}
        assert summarize(long['merged']['rows'])=={k:long['merged'][k] for k in ('by_count','selection_nmse')}
        assert len(short['rows'])==630 and len(long['merged']['rows'])==24
        assert all(sha(p)==h for p,h in hashes.items())
        write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),checkpoint_sha256=sha(cp),
            short_sha256=sha(OUT/'SHORT.json'),long_sha256=sha(OUT/'LONG.json'),protocol_sha256=sha(OUT/'PROTOCOL.json'),
            source_epochs_audited=list(EPOCHS),saved_row_aggregates_verified=True))
        write(ROOT/'reports/2026-10-13/CHECKPOINT_MEAN_RESULT.json',result)
        state('COMPLETE_AUDITED');print('CHECKPOINT_MEAN_COMPLETE',json.dumps(result['short']),flush=True)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
