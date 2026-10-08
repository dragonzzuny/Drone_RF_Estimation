"""Full native-input learnability check, not a held-out architecture contest.

Wait for the existing registered study AND its four-phase evaluation. Compare
fresh full WaveNet and full U-Net on the same four TRAIN mixtures for 32 updates.
No production checkpoint is replaced and no training-data score selects a paper
model. Different capacities and lack of hyperparameter tuning are explicit.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE = ROOT/'experiments/rfuav_native_frequency_20261009'
sys.path.insert(0, str(NATIVE))
from native_data import NativeMixtures, write_json, sha256
import run_native as base
from native_wavenet import build_native_wavenet
from models import build, predict
from drone_rf.losses import pit_waveform_loss
from drone_rf.waveform import waveform_metrics


def read(path):
    return json.loads(path.read_text())


def register(root, preparation, dependency):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Refuse duplicate registration')
    data = NativeMixtures(preparation, 'train_pack', 1)
    indices = [int(i) for count in (2,3) for i in np.flatnonzero(data.rows['count']==count)[:2]]
    sources = dict(read(ROOT/'local/native_frequency_20261009_v1/GPU_PROTOCOL.json')['source_sha256'])
    sources.update({str(p.relative_to(ROOT)):sha256(p) for p in HERE.glob('*.py')})
    for rel, digest in sources.items():
        if sha256(ROOT/rel)!=digest:
            raise ValueError('Prior frozen source changed')
        target = root/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
    protocol = dict(status='REGISTERED_FULL_CAPACITY_TRAIN_ONLY_DIAGNOSTIC', source_sha256=sources,
        preparation=str(preparation), preparation_sha256=sha256(preparation/'PREPARATION.json'),
        manifest_sha256=sha256(preparation/'NATIVE_MANIFEST.json'), dependency=str(dependency),
        train_indices=indices, role='train_pack', epoch=1, seed=0, steps=32, effective_batch=4,
        microbatch=1, crop_samples=63872, context_tokens=255, sample_rate_hz=100_000_000,
        optimizer='fresh AdamW lr5e-4 wd1e-4 clip1, identical in both diagnostic arms',
        loss='whole-window PIT NMSE+coherence+inactive/background + 0.1 count CE',
        arms={'wavenet_native':4601867,'unet_mean_fresh':32142859},
        inference='mixture I/Q, mixture context features, crop position; no references or true counts',
        purpose='fit and gradient/memory/throughput check before a prospective full comparison',
        acceptance='finite gradients and lower fixed TRAIN NMSE after 32 updates; not architecture superiority',
        constraints='full model and full input; shared original RF geometry; no heldout or validation I/Q read',
        initialization='both fresh, no pretrained U-Net advantage; capacities differ and optimizer is not tuned per family',
        maximum_wait_seconds=7200)
    write_json(root/'PROTOCOL.json', protocol)
    return protocol


@torch.no_grad()
def score(net, items):
    net.eval()
    values = []
    for item in items:
        output, _ = predict(net, item)
        measured = waveform_metrics(output, item['references'], item['active'], item['mixture'])
        if float(measured['sum_relative_error'].max()) > 1e-9:
            raise ValueError('Mixture-sum failure')
        values += measured['nmse'][item['active']].tolist()
    return float(np.mean(values))


def run(root, protocol):
    started = time.time()
    dependency = Path(protocol['dependency'])
    while not (dependency/'phase/COMPLETE.json').exists():
        if (dependency/'GPU_FAILURE.json').exists() or (dependency/'phase/FAILURE.json').exists():
            raise RuntimeError('Dependency failed; preserve registered order')
        if time.time()-started > protocol['maximum_wait_seconds']:
            raise TimeoutError('Dependency wait exceeded')
        write_json(root/'STATE.json', dict(status='WAITING_FOR_REGISTERED_STUDY_AND_PHASE', pid=os.getpid(), time=time.time()))
        time.sleep(10)
    for rel, digest in protocol['source_sha256'].items():
        if sha256(ROOT/rel)!=digest:
            raise ValueError('Frozen diagnostic source changed')
    with base.LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).split()
        if any(int(p)!=os.getpid() for p in pids):
            raise RuntimeError('Unexpected other GPU process')
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        data = NativeMixtures(protocol['preparation'], 'train_pack', 1)
        items = [base.batch([data[i]]) for i in protocol['train_indices']]
        results = {}
        for arm in protocol['arms']:
            torch.manual_seed(0)
            net = (build_native_wavenet() if arm=='wavenet_native' else build('unet_mean')).cuda()
            if sum(p.numel() for p in net.parameters())!=protocol['arms'][arm]:
                raise ValueError('Model capacity differs')
            opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4, foreach=False)
            torch.cuda.reset_peak_memory_stats()
            before = score(net, items)
            curve = [dict(step=0,nmse=before)]
            begin = time.time()
            for step in range(1,33):
                net.train();opt.zero_grad(set_to_none=True)
                for item in items:
                    output, logits = predict(net, item)
                    loss = pit_waveform_loss(output,item['references'],item['active'],item['mixture'])['loss']
                    loss = loss+.1*F.cross_entropy(logits,item['construction_count']-1)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite diagnostic loss')
                    (loss/len(items)).backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                if not float(norm)>0:
                    raise ValueError('Zero gradient')
                opt.step()
                if step in (1,8,16,32):
                    curve.append(dict(step=step,nmse=score(net,items)))
                    print(json.dumps(dict(arm=arm,curve=curve[-1])),flush=True)
                write_json(root/'STATE.json', dict(status='TRAIN_ONLY_DIAGNOSTIC',arm=arm,step=step,total=32,
                    seconds=time.time()-begin,pid=os.getpid(),time=time.time()))
            result = dict(arm=arm,parameters=protocol['arms'][arm],curve=curve,
                seconds=time.time()-begin,peak_bytes=torch.cuda.max_memory_allocated(),
                finite_gradients=True,improved=curve[-1]['nmse']<before,heldout_read=False,
                validation_read=False,full_input_samples=63872,production_checkpoint_changed=False)
            write_json(root/f'{arm}.json',result);results[arm]=result
            del net,opt,output,logits,loss,norm
            gc.collect();torch.cuda.empty_cache()
        write_json(root/'COMPLETE.json',dict(status='COMPLETE',results=results,time=time.time(),
            protocol_sha256=sha256(root/'PROTOCOL.json'),interpretation='TRAIN fit only, not generalization ranking'))
        write_json(root/'STATE.json',dict(status='COMPLETED',time=time.time(),pid=os.getpid()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--dependency',type=Path,required=True)
    args=parser.parse_args()
    root=args.run.resolve()
    try:
        protocol=register(root,args.preparation.resolve(),args.dependency.resolve())
        run(root,protocol)
    except Exception:
        root.mkdir(parents=True,exist_ok=True)
        write_json(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write_json(root/'STATE.json',dict(status='FAILED',time=time.time(),pid=os.getpid()))
        raise
