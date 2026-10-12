"""Same known event for BOTH final matched arms, regardless of short DEV scores."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
SEQ=ROOT/'local/sequence_assignment_20261013_v1'
OUT=ROOT/'local/sequence_packets_20261013_v1'
OLD=ROOT.parent/'uav_analysis/rf_detection/separator_detector_20261012'
LOCK=ROOT.parent/'uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock'


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
    OUT.mkdir(exist_ok=True);assert not (OUT/'PLAN.json').exists()
    hashes=dict(read(SEQ/'PROTOCOL.json')['source_sha256'])
    for p in (Path(__file__),Path(__file__).with_name('check_packets.py'),
              ROOT/'reports/2026-10-13/SEQUENCE_PACKET_PLAN_KO.md'):
        hashes[str(p.relative_to(ROOT))]=sha(p)
    plan=dict(registered=time.time(),arms=['independent','shared'],additional_epoch=3,parent_epoch=99,
        source_sha256=hashes,sequence_protocol_sha256=sha(SEQ/'PROTOCOL.json'),
        parent_evidence_sha256=sha(ROOT/'reports/2026-10-13/PACKET_EVIDENCE_FINAL_E099.json'),
        event='Previously exposed FPV/Mavic3Pro2ms event, exact same full-context gains and native frequency offsets',
        inference='Original8short-window C8, reference-free overlap correspondence and Hann averaging',
        receiver='Fixed original+15MHz pilot receiver; no tuning or reference-aided synchronization',
        all_final_arms_evaluated=True,optimizer_updates=0,heldout_access=False,
        independent_recording_validation=False,no_payload_fields_stored=True,
        caveat='One known packet; mixture already decodes; CRC alone does not establish exclusive separation or detector gains')
    write(OUT/'PLAN.json',plan)
    for rel,h in hashes.items():
        assert sha(ROOT/rel)==h
        dst=OUT/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
    while True:
        s=read(SEQ/'STATE.json')
        if s['status']=='COMPLETE_AUDITED':break
        if s['status'] in ('FAILED','STOPPED_WITH_RECOVERY'):raise RuntimeError('Matched training needs attention')
        state('WAITING_MATCHED_STUDY',arm=s.get('arm'),epoch=s.get('epoch'));time.sleep(20)
    while True:
        s=read(ROOT/'local/b99_on_c_20261013_v1/STATE.json')
        if s['status']=='COMPLETE_AUDITED':break
        if s['status']=='FAILED':raise RuntimeError('C evaluation needs attention')
        state('WAITING_C_EVALUATION');time.sleep(15)
    with LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
        assert all(sha(ROOT/p)==h for p,h in hashes.items())
        import numpy as np
        import torch
        from sequence_eval import merge,score
        from sequence_data import WINDOW
        from models import build,predict
        from core import predict_eight
        from study import batch
        from drone_rf.context_data import component_gains,mixture_context_features
        from check_packets import run as receiver_check
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
        mixture=refs.sum(0);features=mixture_context_features(mixture)['context_features'];lo,hi=p['roi']
        original=np.load(OLD/'retest_best_e079/SEPARATED_EVENT.npz')
        assert np.array_equal(refs[:,lo:hi],original['references'][:2])
        assert np.array_equal(mixture[lo:hi],original['mixture'])
        target=np.zeros((3,hi-lo),np.complex64);target[:2]=refs[:,lo:hi]
        comparison=[]
        for arm in plan['arms']:
            source=SEQ/arm;e=read(source/'EPOCH_003.json');a=read(source/'AUDIT_003.json')
            assert a['status']=='PASS' and a['event_sha256']==sha(source/'EPOCH_003.json')
            checkpoint=source/'ACTUAL_003.pt';assert sha(checkpoint)==e['checkpoint_sha256']
            directory=OUT/arm;directory.mkdir(exist_ok=True)
            net=build('unet_mean').cuda().eval().requires_grad_(False)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);net.load_state_dict(saved['model']);del saved
            ys=[]
            with torch.no_grad():
                for j,start in enumerate(p['starts']):
                    item=batch(dict(mixture=mixture[start:start+WINDOW].copy(),context_features=features,crop_start=start,
                        references=np.zeros((3,WINDOW),np.complex64),active=np.zeros(3,bool),construction_count=2),'cuda')
                    _,y,logits,_=predict_eight(net,predict,item);ys.append(y[0].cpu().numpy())
                    state('PACKET_INFERENCE',arm=arm,windows=j+1,total=8)
                begin=p['starts'][0];length=p['starts'][-1]+WINDOW-begin
                output,receipt=merge(ys,[s-begin for s in p['starts']],length);output=output[:,lo-begin:hi-begin]
                metrics=score(torch.from_numpy(output)[None].cuda(),torch.from_numpy(target)[None].cuda(),
                    torch.from_numpy(mixture[lo:hi])[None].cuda(),2,logits)
            np.savez_compressed(directory/'ASSEMBLED.npz',C8_hann=output)
            result=dict(status='COMPLETE',epoch=3,parent_epoch=99,arm=arm,
                checkpoint_sha256=sha(checkpoint),optimizer_updates_in_diagnosis=0,
                stages={'C8_hann':dict(metrics=metrics,overlap_matching=receipt)})
            write(directory/'RESULT.json',result)
            write(directory/'VERIFICATION.json',dict(status='PASS',input_hashes_unchanged=
                sha(first_path)==p['first_event_hash'] and sha(second_path)==p['second_clip']['cache_sha256'],
                artifacts={name:sha(directory/name) for name in ('RESULT.json','ASSEMBLED.npz')}))
            del net;torch.cuda.empty_cache();state('PACKET_RECEIVER',arm=arm)
            receiver_check(directory)
            check=read(directory/'PACKET_AUDIT.json');assert check['status']=='PASS'
            assert check['results_sha256']==sha(directory/'PACKET_RESULTS.json')
            comparison.append(dict(arm=arm,additional_epoch=3,checkpoint_sha256=e['checkpoint_sha256'],
                metrics=metrics,packet=read(directory/'PACKET_RESULTS.json'),packet_audit=check,
                diagnosis_sha256=sha(directory/'RESULT.json')))
        assert all(sha(ROOT/p)==h for p,h in hashes.items())
        assert sha(first_path)==p['first_event_hash'] and sha(second_path)==p['second_clip']['cache_sha256']
        assert sha(ROOT/'reports/2026-10-13/PACKET_EVIDENCE_FINAL_E099.json')==plan['parent_evidence_sha256']
        public=dict(status='VERIFIED',parent_epoch=99,comparisons=comparison,
            original_parent=read(ROOT/'reports/2026-10-13/PACKET_EVIDENCE_FINAL_E099.json'),
            plan_sha256=sha(OUT/'PLAN.json'),independent_recording_validation=False,
            note='All final3epoch arms, one known event, unchanged receiver, not detector accuracy')
        write(OUT/'RESULT.json',public)
        write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),plan_sha256=sha(OUT/'PLAN.json')))
        write(ROOT/'reports/2026-10-13/SEQUENCE_PACKET_RESULT.json',public)
        state('COMPLETE_AUDITED',arms=plan['arms'])


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
