"""Registered five fixed equal-power combinations, evaluated after queued jobs."""
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'local/full_context_parent_20261013_v1'
SEQ=ROOT/'local/sequence_assignment_20261013_v1'
PARENT=ROOT/'local/timeblock_extend100_20261012_v1'
LOCK=ROOT.parent/'uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock'


def read(p):return json.loads(p.read_text())
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def state(status,**kw):write(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def run():
    OUT.mkdir(exist_ok=True)
    hashes=dict(read(SEQ/'PROTOCOL.json')['source_sha256'])
    for name in ('evaluate_parent_full_context.py','full_context_eval.py'):
        p=Path(__file__).with_name(name);hashes[str(p.relative_to(ROOT))]=sha(p)
    paper=ROOT/'reports/2026-10-13/FULL_CONTEXT_PLAN_KO.md';hashes[str(paper.relative_to(ROOT))]=sha(paper)
    checkpoint=PARENT/'ACTUAL_099.pt';event=read(PARENT/'EPOCH_099.json')
    assert sha(checkpoint)==event['checkpoint_sha256']
    import numpy as np
    schedule=SEQ/'preparation/within_record_dev_001.npy';rows=np.load(schedule,allow_pickle=False)
    selected=[i for i,r in enumerate(rows) if r['count']>1 and np.all(r['levels'][:int(r['count'])]==0)]
    assert selected==[0,7,14,15,20]
    plan=dict(registered=time.time(),indices=selected,source_sha256=hashes,schedule_sha256=sha(schedule),
        selection='All five available category combinations with equal nominal source powers in the pre-existing24 cases',
        checkpoint_epoch=99,checkpoint_sha256=event['checkpoint_sha256'],samples=2088960,fs_hz=100000000,
        inference='Original C8 per63872sample window; prediction-only Hann overlap assembly',
        source_center_offsets_preserved=True,whole_context_power_normalization=True,
        cpu_inference=False,optimizer_updates=0,heldout_access=False,independent_recording_validation=False)
    assert not (OUT/'PLAN.json').exists();write(OUT/'PLAN.json',plan)
    for rel,digest in hashes.items():
        dst=OUT/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
        assert sha(dst)==digest
    dependency=ROOT/'local/sequence_packets_20261013_v1'
    while True:
        previous=read(dependency/'STATE.json')
        if previous['status']=='COMPLETE_AUDITED':break
        if previous['status']=='FAILED':raise RuntimeError('Earlier packet evaluation requires attention')
        state('WAITING_MATCHED_AND_PACKET',previous_stage=previous['status']);time.sleep(20)
    with LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
        assert all(sha(ROOT/p)==h for p,h in hashes.items())
        import torch
        from sequence_data import Sequences,ROLES
        from full_context_eval import evaluate,check_streaming
        from models import build,predict
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        check=check_streaming();write(OUT/'STREAMING_CHECK.json',check)
        net=build('unet_mean').cuda().eval().requires_grad_(False)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);net.load_state_dict(saved['model']);del saved
        data=Sequences(ROLES[1]);results=[]
        for position,index in enumerate(selected):
            raw=data.rows[index];count=int(raw['count']);refs=data.components(index,True)
            features=np.array(data.features[index])
            def progress(done,total):state('FULL_CONTEXT_INFERENCE',case=position+1,total_cases=5,
                windows=done,total_windows=total)
            output,result=evaluate(net,predict,refs,features,count,'cuda',progress)
            np.save(OUT/f'OUTPUT_{index:03d}.npy',output,allow_pickle=False)
            result.update(index=index,categories=[data.library.clips[int(i)]['category'] for i in raw['indices'][:count]],
                nominal_levels_db=raw['levels'][:count].tolist(),output_sha256=sha(OUT/f'OUTPUT_{index:03d}.npy'))
            write(OUT/f'CASE_{index:03d}.json',result);results.append(result)
            del output,refs;gc.collect();torch.cuda.empty_cache()
        assert all(sha(ROOT/p)==h for p,h in hashes.items()) and sha(checkpoint)==event['checkpoint_sha256']
        result=dict(status='VERIFIED',checkpoint_epoch=99,checkpoint_sha256=event['checkpoint_sha256'],
            cases=results,plan_sha256=sha(OUT/'PLAN.json'),optimizer_updates=0,
            scope='Five fixed equal-power B DEV mixtures over20.8896ms; no independent recording or detection claim')
        write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),
            streaming_equivalence=True,cases=5,source_and_checkpoint_hashes=True))
        write(ROOT/'reports/2026-10-13/PARENT_FULL_CONTEXT_RESULT.json',result)
        state('COMPLETE_AUDITED',cases=5)
        print('FULL_CONTEXT_PARENT_COMPLETE',[(v['index'],v['metrics']['nmse'],v['active_order_changes']) for v in results],flush=True)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
