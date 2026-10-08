"""TRAIN-only gradient diagnosis of shared context learning; zero updates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from native_data import ROOT, NativeMixtures, sha256, write_json
from phase_native import engine, native_inference_batch
from drone_rf.losses import pit_waveform_loss


def read(path):
    return json.loads(path.read_text())


def geometry(wave, count):
    wn=float(torch.linalg.vector_norm(wave));cn=float(torch.linalg.vector_norm(count))
    dot=float(torch.dot(wave,count))
    return dict(wave_norm=wn, weighted_count_norm=cn, weighted_count_to_wave_norm=cn/max(wn,1e-30),
                dot=dot, cosine=dot/max(wn*cn,1e-30))


def main(parent):
    root=parent/'gradient'
    root.mkdir(exist_ok=True)
    if (root/'PROTOCOL.json').exists():raise ValueError('Refuse duplicate gradient diagnosis')
    if not (parent/'fit_gap/COMPLETE.json').exists():raise ValueError('Prior diagnosis incomplete')
    base=read(parent/'fit_gap/PROTOCOL.json');sources=dict(base['source_sha256'])
    sources[str(Path(__file__).resolve().relative_to(ROOT))]=sha256(__file__)
    data=NativeMixtures(parent/'preparation','train_pack',1)
    candidates=base['subsets']['1']
    indices=sorted(i for count in (1,2,3) for i in [j for j in candidates if int(data.rows[j]['count'])==count][:32])
    checkpoint=parent/'gpu/SELECTED_005.pt'
    plan=dict(status='REGISTERED_TRAIN_GRADIENT_DIAGNOSIS',source_sha256=sources,
        data_sha256=base['data_sha256'],indices=indices,checkpoint_sha256=sha256(checkpoint),
        cases=96,parameters='shared context_encoder only',model_mode='eval with autograd, no optimizer',
        wave_loss='existing PIT waveform loss',count_loss='0.1 times construction-count cross entropy',
        aggregates='per case; equal means by count and all96',model_updates=0,validation_read=False,heldout_read=False,
        interpretation='Local optimization diagnosis; negative cosine does not establish validation causality',
        next_test_rule='prioritize identical-budget count-gradient-detached comparison only if overall cosine<0 and weighted-count gradient norm >= waveform gradient norm')
    if len(indices)!=96:raise ValueError('Wrong probe budget')
    for rel,digest in sources.items():
        if sha256(ROOT/rel)!=digest:raise ValueError('Changed source')
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    write_json(root/'PROTOCOL.json',plan)
    with engine.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        engine.gpu_check();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        net=engine.stft_build('unet_mean').cuda().eval()
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        net.load_state_dict(saved['model']);selected=saved['best']['epoch'];del saved
        params=list(net.context_encoder.parameters());size=sum(p.numel() for p in params)
        sums={k:[torch.zeros(size,device='cuda',dtype=torch.float64) for _ in range(2)] for k in (1,2,3)}
        rows=[];started=time.time()
        write_json(root/'STATE.json',dict(status='GPU_DIAGNOSIS_RUNNING',pid=os.getpid(),time=time.time()))
        for index in indices:
            item=native_inference_batch(data[index],'cuda');count=int(item['construction_count'][0])
            prediction,logits=engine.stft_predict(net,{k:item[k] for k in ('mixture','context_features','crop_start')})
            waveform=pit_waveform_loss(prediction,item['references'],item['active'],item['mixture'])['loss']
            counting=.1*F.cross_entropy(logits,item['construction_count']-1)
            gc=torch.autograd.grad(counting,params,retain_graph=True)
            gw=torch.autograd.grad(waveform,params)
            w=torch.cat([g.reshape(-1) for g in gw]).to(torch.float64)
            c=torch.cat([g.reshape(-1) for g in gc]).to(torch.float64)
            if not bool(torch.isfinite(w).all() and torch.isfinite(c).all()):raise ValueError('Nonfinite gradient')
            sums[count][0]+=w;sums[count][1]+=c
            rows.append(dict(index=index,count=count,wave_loss=float(waveform),weighted_count_loss=float(counting),**geometry(w,c)))
            del prediction,logits,waveform,counting,gw,gc,w,c
            if len(rows)%16==0:
                write_json(root/'PROGRESS.json',dict(cases=len(rows),total=96,seconds=time.time()-started,time=time.time(),pid=os.getpid()))
        groups=[]
        for count in (1,2,3):
            subset=[r for r in rows if r['count']==count]
            groups.append(dict(count=count,cases=len(subset),negative_case_fraction=float(np.mean([r['cosine']<0 for r in subset])),
                **geometry(sums[count][0]/32,sums[count][1]/32)))
        overall=geometry(sum(v[0] for v in sums.values())/96,sum(v[1] for v in sums.values())/96)
        for path,digest in plan['data_sha256'].items():
            if sha256(path)!=digest:raise ValueError('Probe metadata changed')
        if sha256(checkpoint)!=plan['checkpoint_sha256']:raise ValueError('Probe checkpoint changed')
        for rel,digest in sources.items():
            if sha256(ROOT/rel)!=digest:raise ValueError('Probe source changed')
        result=dict(status='COMPLETE',protocol_sha256=sha256(root/'PROTOCOL.json'),selected_epoch=selected,
            shared_parameters=size,by_count=groups,overall=overall,rows=rows,seconds=time.time()-started,
            next_test_triggered=overall['cosine']<0 and overall['weighted_count_to_wave_norm']>=1,
            model_updates=0,heldout_read=False,validation_read=False)
        write_json(root/'COMPLETE.json',result)
        write_json(root/'STATE.json',dict(status='COMPLETED',time=time.time()))
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    os.sched_setaffinity(0,{14,15});os.nice(10)
    try:main(args.run.resolve())
    except Exception:
        write_json(args.run/'gradient/FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write_json(args.run/'gradient/STATE.json',dict(status='FAILED',time=time.time()));raise
