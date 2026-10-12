"""One frozen B-selected checkpoint on previously exposed separate-record C DEV.

No C scores select the tested epoch:99 was selected using the original B rule.
No learning, threshold tuning, heldout access or incumbent replacement occurs.
Imports of Torch and data caches wait until the GPU lock is acquired.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'local/b99_on_c_20261013_v1'
B=ROOT/'local/timeblock_extend100_20261012_v1'
C=ROOT/'local/phase_eight_20261011_v1'
MANIFEST_B=ROOT/'local/bandwidth_evaluation_20261011_v1/TIME_BLOCK_MANIFEST.json'
LOCK=ROOT.parent/'uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock'
PLAN=ROOT/'reports/2026-10-13/B99_ON_C_PLAN_KO.md'


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    q=path.with_suffix(path.suffix+'.tmp')
    q.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(path)


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**kw))


def run():
    OUT.mkdir(exist_ok=True);assert not (OUT/'PROTOCOL.json').exists()
    parent=read(B/'EPOCH_100.json');audit=read(B/'AUDIT_100.json')
    assert audit['status']=='PASS' and audit['event_sha256']==sha(B/'EPOCH_100.json')
    assert parent['best']['epoch']==99
    checkpoint=B/'ACTUAL_099.pt';selected=read(B/'EPOCH_099.json')
    assert sha(checkpoint)==selected['checkpoint_sha256']
    old=read(C/'PROTOCOL.json');preparation=Path(old['preparation'])
    native=read(preparation/'NATIVE_MANIFEST.json')
    fit=[x for x in read(MANIFEST_B)['clips'] if x['diagnostic_role']=='fit_block']
    dev=[x for x in native['clips'] if x['role']=='validation_pack']
    fit_packs={x['pack_id'] for x in fit};dev_packs={x['pack_id'] for x in dev}
    assert len(fit_packs)==len(dev_packs)==5 and not fit_packs&dev_packs
    assert all('VTSBW=10/' in x['pack_id'] for x in fit)
    assert all('VTSBW=20/' in x['pack_id'] for x in dev)
    assert not {x['source_sha256'] for x in fit}&{x['source_sha256'] for x in dev}
    sources=dict(old['source_sha256'])
    for rel,digest in read(B/'PROTOCOL.json')['source_sha256'].items():
        if rel in sources:assert sources[rel]==digest,rel
        sources[rel]=digest
    for path in (Path(__file__),PLAN,Path(__file__).with_name('sequence_eval.py'),
                 Path(__file__).with_name('sequence_data.py'),Path(__file__).with_name('sequence_objective.py')):
        sources[str(path.relative_to(ROOT))]=sha(path)
    baseline=ROOT/'reports/2026-10-11/PHASE_EIGHT_RESULT.json'
    selection=ROOT/'reports/2026-10-11/CURRENT_BEST_SELECTION.json'
    assert sha(baseline)==read(selection)['result_sha256']
    files=[checkpoint,MANIFEST_B,preparation/'PREPARATION.json',preparation/'NATIVE_MANIFEST.json',
           baseline,selection,C/'PROTOCOL.json',B/'EPOCH_100.json']
    protocol=dict(registered=time.time(),source_sha256=sources,pinned_files={str(p):sha(p) for p in files},
        checkpoint_epoch=99,checkpoint_sha256=sha(checkpoint),parameters=32142859,
        selection='Original B macroNMSE after100, fixed before C inference',cases=630,
        c_rows_sha256=old['dev_rows_sha256'],optimizer_updates=0,
        inference='Unchanged8phase aligned mean; mixture/context only',
        fit_recording_groups=sorted(fit_packs),c_recording_groups=sorted(dev_packs),
        pack_sets_disjoint=True,raw_file_hash_sets_disjoint=True,fit_bandwidth_mhz=10,c_bandwidth_mhz=20,
        precision='FP32 TF32off',no_threshold_tuning=True,no_heldout_access=True,
        already_exposed_development=True,new_independent_final_test=False,changes_current_best=False,
        caveat='Separate recording groups AND capture-bandwidth shift; earlier model has different training budget, not a method-only comparison')
    write(OUT/'PROTOCOL.json',protocol)
    for rel,digest in sources.items():
        assert sha(ROOT/rel)==digest,rel
        dst=OUT/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
        assert sha(dst)==digest
    def verify():
        for rel,h in sources.items():assert sha(ROOT/rel)==h and sha(OUT/'source_snapshot'/rel)==h,rel
        for p,h in protocol['pinned_files'].items():assert sha(p)==h,p
    with LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);verify()
        import numpy as np
        import torch
        from sequence_eval import score,summarize
        from models import build,predict
        from study import batch
        from core import predict_eight
        sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
        from native_data import NativeMixtures
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        data=NativeMixtures(preparation,'validation_pack',1)
        assert data.rows_hash==protocol['c_rows_sha256'] and len(data)==630
        original=read(baseline)['eight_phase'];old_rows=original['rows']
        net=build('unet_mean').cuda().eval().requires_grad_(False)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        net.load_state_dict(saved['model']);del saved
        rows=[];started=time.time()
        with torch.no_grad():
            for i in range(len(data)):
                item=batch(data[i],'cuda');raw=data.rows[i];k=int(raw['count'])
                categories=[data.library.clips[int(j)]['category'] for j in raw['indices'][:k]]
                assert old_rows[i]['index']==i and old_rows[i]['count']==k and old_rows[i]['categories']==categories
                _,y,logits,_=predict_eight(net,predict,item)
                r=score(y,item['references'],item['mixture'],k,logits)
                assert r['reference_power']==old_rows[i]['reference_power']
                r.update(index=i,categories=categories,nominal_levels_db=raw['levels'][:k].tolist())
                rows.append(r)
                if i%50==0:state('C_DEV_INFERENCE',case=i+1,total=630,seconds=time.time()-started)
        result=dict(status='COMPLETE',checkpoint_epoch=99,checkpoint_sha256=sha(checkpoint),
            rows=rows,rows_sha256=data.rows_hash,**summarize(rows),
            baseline=[{k:v for k,v in g.items()} for g in original['by_count']],
            optimizer_updates=0,seconds=time.time()-started,source_recording_groups_disjoint=True,
            nominal_capture_bandwidth_changed=True,heldout_access=False,independent_final_test=False,
            current_best_changed=False,protocol_sha256=sha(OUT/'PROTOCOL.json'))
        write(OUT/'RESULT.json',result);verify()
        assert summarize(read(OUT/'RESULT.json')['rows'])=={k:result[k] for k in ('by_count','selection_nmse')}
        write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),
            protocol_sha256=sha(OUT/'PROTOCOL.json'),rows=630,reference_powers_exactly_match_prior_C=True,
            pack_sets_disjoint=True,source_hashes_unchanged=True))
        write(ROOT/'reports/2026-10-13/B99_ON_C_RESULT.json',{k:v for k,v in result.items() if k!='rows'})
        state('COMPLETE_AUDITED',cases=630)
        print('C_EVALUATION_COMPLETE',json.dumps(result['by_count']),flush=True)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
