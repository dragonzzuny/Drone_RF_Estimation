"""Fixed posthoc development experiment on all five saved full observations."""
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'local/spectral_routing_20261013_v1'
PARENT=ROOT/'local/full_context_parent_20261013_v1'
FIT=ROOT/'local/b_timeblock_spectral_20261013_v1'
LOCK=ROOT.parent/'uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock'


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def state(status,**kw):write(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def run():
    assert read(PARENT/'STATE.json')['status']=='COMPLETE_AUDITED'
    original=read(PARENT/'RESULT.json');assert sha(PARENT/'RESULT.json')==read(PARENT/'AUDIT.json')['result_sha256']
    assert sha(FIT/'TEMPLATES.json')==read(FIT/'AUDIT.json')['templates_sha256']
    sources=dict(read(PARENT/'PLAN.json')['source_sha256'])
    for p in (Path(__file__),HERE/'spectral_routing.py',HERE/'test_spectral_routing.py',
              ROOT/'reports/2026-10-13/SPECTRAL_ROUTING_PLAN_KO.md'):
        sources[str(p.relative_to(ROOT))]=sha(p)
    assert read(ROOT/'reports/2026-10-13/SPECTRAL_ROUTING_CPU_CHECK.json')['status']=='PASS'
    OUT.mkdir(exist_ok=True);assert not (OUT/'PLAN.json').exists()
    files={str(p):sha(p) for p in (PARENT/'RESULT.json',PARENT/'PLAN.json',FIT/'TEMPLATES.json')}
    for case in original['cases']:
        p=PARENT/f"OUTPUT_{case['index']:03d}.npy";assert sha(p)==case['output_sha256'];files[str(p)]=sha(p)
    plan=dict(registered=time.time(),source_sha256=sources,input_sha256=files,
        methods=['fit_dictionary','observation_clusters'],cases=[0,7,14,15,20],block_samples=63872,
        synthetic_check_sha256=sha(ROOT/'reports/2026-10-13/SPECTRAL_ROUTING_CPU_CHECK.json'),
        optimizer_updates=0,references_in_routing=False,known_category_dictionary_arm=True,
        clustering_count='Parent predicted count, not construction count',posthoc_development=True,
        independent_recording_validation=False,heldout_access=False)
    write(OUT/'PLAN.json',plan)
    for name,digest in sources.items():
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p);assert sha(p)==digest
    def verify():
        assert all(sha(ROOT/p)==h and sha(OUT/'source_snapshot'/p)==h for p,h in sources.items())
        assert all(sha(p)==h for p,h in files.items())
    with LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);verify()
        import numpy as np
        import torch
        from sequence_data import Sequences,ROLES
        from sequence_eval import score
        from spectral_routing import route
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        data=Sequences(ROLES[1]);banks=read(FIT/'TEMPLATES.json')['banks'];rows=[]
        for i,case in enumerate(original['cases']):
            index=case['index'];raw=data.rows[index];count=int(raw['count']);refs=data.components(index,True)
            output=np.load(PARENT/f'OUTPUT_{index:03d}.npy',allow_pickle=False);mixture=refs.sum(0)
            ids=raw['indices'][:count];centers={data.library.clips[int(j)]['common_center_hz'] for j in ids}
            assert len(centers)==1;center=int(next(iter(centers)));bank=banks[str(center)]
            ref=torch.from_numpy(refs)[None].cuda();mix=torch.from_numpy(mixture)[None].cuda()
            predicted_count=case['metrics']['predicted_count'];logits=torch.zeros(1,3,device='cuda');logits[0,predicted_count-1]=1
            def evaluate(value):return score(torch.from_numpy(value)[None].cuda(),ref,mix,count,logits)
            with torch.no_grad():
                baseline=evaluate(output)
                for key in ('nmse','si_sdr','reference_power'):
                    np.testing.assert_allclose(baseline[key],case['metrics'][key],rtol=1e-11,atol=1e-11)
                assert baseline['assignment']==case['metrics']['assignment']
                methods={}
                for method in plan['methods']:
                    state('ROUTING_AND_SCORING',case=i+1,total=5,method=method)
                    routed,receipt=route(output,bank,method,predicted_count)
                    metrics=evaluate(routed);assert metrics['sum_relative_error']<1e-9
                    receipt_path=OUT/f'{method}_{index:03d}_ORDERS.json';write(receipt_path,receipt)
                    methods[method]=dict(metrics=metrics,orders_sha256=sha(receipt_path))
                    del routed;gc.collect();torch.cuda.empty_cache()
            total=case['local_scoring'][-1]['stop'];local=[]
            for j in range(count):
                numerator=sum(w['local_pit_nmse'][j]*w['reference_power'][j]*(w['stop']-w['start']) for w in case['local_scoring'])
                local.append(numerator/(case['metrics']['reference_power'][j]*total))
            result=dict(index=index,categories=case['categories'],parent=baseline,methods=methods,
                local_reference_mapping_nmse_diagnostic_only=local)
            rows.append(result);write(OUT/f'CASE_{index:03d}.json',result)
            del output,refs,mixture,ref,mix;gc.collect();torch.cuda.empty_cache()
        verify()
        result=dict(status='VERIFIED',cases=rows,plan_sha256=sha(OUT/'PLAN.json'),optimizer_updates=0,
            references_in_routing=False,posthoc_development=True,heldout_access=False,
            scope='All five fixed equal-power same-record observations; not weak-source sweep or new-recording proof')
        write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),
            baseline_reproduced=True,pinned_sources_and_inputs_unchanged=True))
        write(ROOT/'reports/2026-10-13/SPECTRAL_ROUTING_RESULT.json',result)
        state('COMPLETE_AUDITED',cases=5,methods=plan['methods'])
        print('SPECTRAL_ROUTING_COMPLETE',[(v['index'],{m:r['metrics']['nmse'] for m,r in v['methods'].items()}) for v in rows],flush=True)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
