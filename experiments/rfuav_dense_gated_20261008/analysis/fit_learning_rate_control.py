"""Matched LR-only continuation diagnostic; fitted TRAIN examples, not a trial.

Both arms restart AdamW from the same step-300 full-model weights, using the
same four original examples. No deployment or validation-score substitution.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_mix_fit import (LOCK, PREPARATION, admitted_dataset, atomic_torch, batch,
                           evaluate, make_model, objective, sha256, write_json)


def run(fit, root):
    if any(root.iterdir()):
        raise RuntimeError('Fresh run directory required')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    process_lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                           '--format=csv,noheader,nounits'],text=True).splitlines()
    for line in process_lines:
        pid,memory=line.split(','); pid=int(pid)
        if pid==os.getpid():
            continue
        path=Path('/proc',str(pid))
        if (path.joinpath('exe').resolve()!=Path('/usr/share/rustdesk/rustdesk')
                or path.stat().st_uid!=os.getuid() or not 0<=float(memory)<=256):
            raise RuntimeError('Other GPU task active')
    original=json.loads((fit/'PROTOCOL.json').read_text())
    for name,digest in original['source_sha256'].items():
        if sha256(Path(name))!=digest:
            raise RuntimeError('Original frozen source changed')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config=json.loads((PREPARATION/'PREPARATION.json').read_text())
    data=admitted_dataset(PREPARATION,'train_pack',1)
    items=[batch(data[e['schedule_index']],'cuda') for e in original['examples']]
    arms=[('lr_1e4',1e-4),('lr_1e5',1e-5)]
    checkpoint=fit/'BEST.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['step']==300
    protocol=dict(status='MATCHED_FIXED_TRAIN_LR_DIAGNOSTIC', no_validation_or_heldout_iq=True,
        generalization_claim=False, seed=0,full_model_parameters=32142859,
        initialization_sha256=sha256(checkpoint), original_fit_protocol_sha256=sha256(fit/'PROTOCOL.json'),
        original_fit_result_sha256=sha256(fit/'STEP_300.json'), examples=original['examples'],
        updates_per_arm=150,effective_batch=4,microbatch=1,weight_decay=0.,
        optimizer='fresh AdamW in BOTH arms; preceding optimizer state unavailable',
        arms=[dict(name=name,learning_rate=lr) for name,lr in arms],
        loss='unchanged PIT NMSE+coherence/inactive/background plus count CE',
        selection='report BOTH terminal results, no selecting the best evaluation step',
        fit_target='all ten components NMSE<0.05 and complex SI-SDR>10dB',
        source_sha256={**original['source_sha256'],str(Path(__file__).resolve()):sha256(Path(__file__))})
    write_json(root/'PROTOCOL.json',protocol)
    start=time.time()
    for name,lr in arms:
        torch.manual_seed(0); np.random.seed(0)
        directory=root/name; directory.mkdir()
        net=make_model('unet_mean',config).cuda()
        net.load_state_dict(saved['model'],strict=True)
        assert sum(p.numel() for p in net.parameters())==32142859
        optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0.,foreach=False)
        initial=evaluate(net,items,0)
        expected=json.loads((fit/'STEP_300.json').read_text())
        np.testing.assert_allclose(initial['mean_nmse'],expected['mean_nmse'],rtol=1e-6,atol=1e-8)
        write_json(directory/'STEP_000.json',initial)
        for step in range(1,151):
            net.train(); optimizer.zero_grad(set_to_none=True)
            total=0.
            for item in items:
                loss=objective(net,item,config)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                (loss/4).backward(); total+=float(loss.detach())/4
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if step%10==0:
                progress=dict(stage='FIT_LR_CONTROL_RUNNING',arm=name,step=step,max_updates=150,
                              seconds=time.time()-start,loss=total,pid=os.getpid(),time=time.time())
                write_json(root/'PROGRESS.json',progress)
                print(json.dumps(progress),flush=True)
            if step in (50,150):
                write_json(directory/f'STEP_{step:03d}.json',evaluate(net,items,step))
        atomic_torch(directory/'FINAL.pt',dict(model=net.state_dict(),step=150,original_fit_steps=300))
        del net,optimizer,loss
        gc.collect(); torch.cuda.empty_cache()
    result=dict(status='FIT_LR_CONTROL_COMPLETE',wall_seconds=time.time()-start,
        generalization_claim=False,no_validation_or_heldout_iq=True,
        arms={name:json.loads((root/name/'STEP_150.json').read_text()) for name,_ in arms})
    write_json(root/'COMPLETE.json',result)
    print('FIT_LR_CONTROL_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-2:]))
    os.nice(10)
    args.run.mkdir(parents=True,exist_ok=True)
    try:
        with LOCK.open('r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.fit,args.run)
    except Exception:
        write_json(args.run/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
