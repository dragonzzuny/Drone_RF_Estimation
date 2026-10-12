"""GPU packet-event diagnosis queued BEFORE the matched follow-up training.

The immutable parent holds the same flock until its audited100 completion.
An additional fixed 2ms whole-input inference is a diagnostic, not a retrained
long-context model. It is not selected by looking at the reference waveform.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'local/packet_final100_20261013_v1'
PARENT=ROOT/'local/timeblock_extend100_20261012_v1'
OLD=ROOT.parent/'uav_analysis/rf_detection/separator_detector_20261012'
LOCK=ROOT.parent/'uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock'


def read(p):return json.loads(p.read_text())


def write(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def run():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'PLAN.json').exists()
    write(OUT/'PLAN.json',dict(registered=time.time(),source_sha256=sha(__file__),
        selection='Original short DEV630 macro NMSE best after audited100, before new updates',
        inputs='Fixed previously exposed FPV/Mavic3Pro2ms event, exact original components',
        comparisons=['existing C8 overlapping short windows/Hann','one fixed full2ms C8 input'],
        longer_input_is_distribution_shift=True,no_new_training=True,no_heldout_access=True))
    write(OUT/'STATE.json',dict(status='WAITING_GPU_LOCK',pid=os.getpid()))
    with LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        parent=read(PARENT/'STATE.json')
        assert parent['status']=='COMPLETE_AUDITED' and parent['epoch']==100,parent
        event=read(PARENT/'EPOCH_100.json');audit=read(PARENT/'AUDIT_100.json')
        assert audit['status']=='PASS' and audit['event_sha256']==sha(PARENT/'EPOCH_100.json')
        selected=event['best']['epoch'];run_dir=PARENT if selected>50 else ROOT/'local/bandwidth_timeblock_20261011_v1'
        checkpoint=run_dir/f'ACTUAL_{selected:03d}.pt'
        assert sha(checkpoint)==read(run_dir/f'EPOCH_{selected:03d}.json')['checkpoint_sha256']
        import numpy as np
        import torch
        from sequence_eval import merge,score
        from sequence_data import WINDOW
        from models import build,predict
        from core import predict_eight
        from drone_rf.context_data import component_gains,mixture_context_features
        from study import batch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        p=read(OLD/'raw_diagnosis/SEPARATION_PLAN.json')
        first_path=OLD/'raw_diagnosis/FIRST_EVENT.npz';second_path=Path(p['second_clip']['cache_path'])
        assert sha(first_path)==p['first_event_hash'] and sha(second_path)==p['second_clip']['cache_sha256']
        first=np.load(first_path)['filtered'];second=np.load(second_path)
        gains=component_gains([float(np.mean(abs(x.astype(np.complex128))**2)) for x in (first,second)],[0,0],[0,.7])
        assert np.array_equal([[g.real,g.imag] for g in gains],p['gains_real_imag'])
        refs=np.stack([(x*g).astype(np.complex64) for x,g in zip((first,second),gains)])
        mix=refs.sum(0);features=mixture_context_features(mix)['context_features'];lo,hi=p['roi']
        prior=np.load(OLD/'retest_best_e079/SEPARATED_EVENT.npz')
        assert np.array_equal(refs[:,lo:hi],prior['references'][:2])
        assert np.array_equal(mix[lo:hi],prior['mixture']) and np.array_equal(features,prior['context_features'])
        target=np.zeros((3,hi-lo),np.complex64);target[:2]=refs[:,lo:hi]
        write(OUT/'INITIALIZATION.json',dict(epoch=selected,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            parent_event_sha256=sha(PARENT/'EPOCH_100.json'),input_verified_exact=True))
        net=build('unet_mean').cuda().eval().requires_grad_(False)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);net.load_state_dict(saved['model']);del saved
        def item(start,length):
            # Inference whitelist in predict ignores the placeholders for scoring.
            return batch(dict(mixture=mix[start:start+length].copy(),context_features=features,
                crop_start=start,references=np.zeros((3,length),np.complex64),
                active=np.zeros(3,bool),construction_count=2),'cuda')
        ys=[]
        with torch.no_grad():
            for j,start in enumerate(p['starts']):
                _,estimate,logits,_=predict_eight(net,predict,item(start,WINDOW))
                ys.append(estimate[0].cpu().numpy())
                write(OUT/'STATE.json',dict(status='SHORT_WINDOWS',epoch=selected,windows=j+1,pid=os.getpid()))
            beginning=p['starts'][0];length=p['starts'][-1]+WINDOW-beginning
            output,receipts=merge(ys,[s-beginning for s in p['starts']],length)
            output=output[:,lo-beginning:hi-beginning]
            y=torch.from_numpy(output)[None].cuda();r=torch.from_numpy(target)[None].cuda();m=torch.from_numpy(mix[lo:hi])[None].cuda()
            baseline=score(y,r,m,2,logits)
            arrays={'C8_hann':output};stages={'C8_hann':dict(metrics=baseline,overlap_matching=receipts)}
            try:
                write(OUT/'STATE.json',dict(status='FULL_2MS_INPUT',epoch=selected,pid=os.getpid()))
                _,full,full_logits,_=predict_eight(net,predict,item(lo,hi-lo))
                arrays['full_2ms_C8']=full[0].cpu().numpy()
                stages['full_2ms_C8']=dict(metrics=score(full,r,m,2,full_logits))
            except torch.cuda.OutOfMemoryError:
                stages['full_2ms_C8']=dict(status='GPU_OOM',adopted=False)
                torch.cuda.empty_cache()
        np.savez_compressed(OUT/'ASSEMBLED.npz',**arrays)
        write(OUT/'RESULT.json',dict(status='COMPLETE',epoch=selected,stages=stages,independent_recordings=False,
            no_training_updates=True,full_input_diagnostic='Fixed length2ms, model trained on0.63872ms; not an architecture-training claim'))
        write(OUT/'VERIFICATION.json',dict(status='PASS',input_hashes_unchanged=
            sha(first_path)==p['first_event_hash'] and sha(second_path)==p['second_clip']['cache_sha256'],
            artifacts={name:sha(OUT/name) for name in ('ASSEMBLED.npz','RESULT.json','PLAN.json','INITIALIZATION.json')}))
        del net;torch.cuda.empty_cache()
        print('FINAL_MODEL_EVENT',selected,json.dumps(stages),flush=True)
        # Read-only CPU receiver runs before releasing the GPU lock, completing
        # the packet-preservation check before the follow-up trainer proceeds.
        from check_packets import run as check_packets
        check_packets(OUT)
        write(OUT/'STATE.json',dict(status='COMPLETE_WITH_PACKET_CHECK',epoch=selected,pid=os.getpid()))


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        raise
