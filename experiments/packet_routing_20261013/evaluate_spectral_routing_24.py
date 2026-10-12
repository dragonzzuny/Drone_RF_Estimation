"""Frozen four routing rules, all24 previously specified B observations."""
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from evaluate_parent_full_context import read,write,sha,LOCK

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'local/spectral_routing_24_20261013_v1'
PARENT=ROOT/'local/full_context_parent_20261013_v1'
V2=ROOT/'local/spectral_routing_20261013_v2'
FIT=ROOT/'local/b_timeblock_spectral_20261013_v1'
METHODS=['fit_dictionary','observation_clusters','count_dictionary','slot_initialized_clusters']


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def run():
    import numpy as np
    from sequence_data import Sequences,ROLES
    assert read(V2/'STATE.json')['status']=='COMPLETE_AUDITED'
    assert sha(V2/'RESULT.json')==read(V2/'AUDIT.json')['result_sha256']
    sources=dict(read(V2/'PLAN.json')['source_sha256'])
    for p in (Path(__file__),ROOT/'reports/2026-10-13/SPECTRAL_ROUTING_24_PLAN_KO.md'):
        sources[str(p.relative_to(ROOT))]=sha(p)
    files=dict(read(V2/'PLAN.json')['input_sha256'])
    files[str(V2/'RESULT.json')]=sha(V2/'RESULT.json')
    checkpoint=ROOT/'local/timeblock_extend100_20261012_v1/ACTUAL_099.pt'
    parent_result=read(PARENT/'RESULT.json');files[str(checkpoint)]=parent_result['checkpoint_sha256']
    schedule=ROOT/'local/sequence_assignment_20261013_v1/preparation/within_record_dev_001.npy'
    files[str(schedule)]=sha(schedule)
    data=Sequences(ROLES[1]);assert len(data)==24
    counts=[int(r['count']) for r in data.rows];assert [counts.count(k) for k in (1,2,3)]==[5,12,7]
    plan=dict(registered=time.time(),source_sha256=sources,input_sha256=files,indices=list(range(24)),
        methods=METHODS,primary='slot_initialized_clusters',reused_indices=[0,7,14,15,20],
        samples=2088960,fs_hz=100000000,block_samples=63872,optimizer_updates=0,
        references_in_routing=False,count_from_model=True,heldout_access=False,
        independent_recording_validation=False,posthoc_development=True,
        scope='Fixed24 B same-record DEV observations; five previously inspected,19 additional full observations')
    OUT.mkdir(exist_ok=True);assert not (OUT/'PLAN.json').exists();write(OUT/'PLAN.json',plan)
    for name,digest in sources.items():
        dst=OUT/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dst);assert sha(dst)==digest
    def verify():
        assert all(sha(ROOT/p)==h and sha(OUT/'source_snapshot'/p)==h for p,h in sources.items())
        assert all(sha(p)==h for p,h in files.items())
    with LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);verify()
        import torch
        from models import build,predict
        from full_context_eval import evaluate
        from sequence_eval import score,summarize
        from spectral_routing import route as route_v1
        from spectral_routing_v2 import route as route_v2
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        net=build('unet_mean').cuda().eval().requires_grad_(False)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);net.load_state_dict(saved['model']);del saved
        banks=read(FIT/'TEMPLATES.json')['banks'];previous={r['index']:r for r in parent_result['cases']};rows=[]
        for index in range(24):
            raw=data.rows[index];count=int(raw['count']);refs=data.components(index,True)
            mixture=refs.sum(0);features=np.array(data.features[index])
            if index in previous:
                case=previous[index];path=PARENT/f'OUTPUT_{index:03d}.npy'
                assert sha(path)==case['output_sha256'];output=np.load(path,allow_pickle=False)
                reused=True
            else:
                def progress(done,total):state('FULL_CONTEXT_INFERENCE',case=index+1,total_cases=24,windows=done,total_windows=total)
                output,case=evaluate(net,predict,refs,features,count,'cuda',progress)
                path=OUT/f'OUTPUT_{index:03d}.npy';np.save(path,output,allow_pickle=False)
                case.update(index=index,categories=[data.library.clips[int(i)]['category'] for i in raw['indices'][:count]],
                    nominal_levels_db=raw['levels'][:count].tolist(),output_sha256=sha(path))
                write(OUT/f'PARENT_{index:03d}.json',case);reused=False
            ids=raw['indices'][:count];centers={data.library.clips[int(j)]['common_center_hz'] for j in ids};assert len(centers)==1
            bank=banks[str(int(next(iter(centers))))]
            ref=torch.from_numpy(refs)[None].cuda();mix=torch.from_numpy(mixture)[None].cuda()
            predicted=case['metrics']['predicted_count'];logits=torch.zeros(1,3,device='cuda');logits[0,predicted-1]=1
            def scoring(value):return score(torch.from_numpy(value)[None].cuda(),ref,mix,count,logits)
            with torch.no_grad():
                baseline=scoring(output)
                for key in ('nmse','si_sdr','reference_power'):
                    assert baseline[key]==case['metrics'][key] or np.allclose(baseline[key],case['metrics'][key],rtol=1e-11,atol=1e-11)
                methods={}
                for method in METHODS:
                    state('ROUTING_AND_SCORING',case=index+1,total_cases=24,method=method)
                    if method in METHODS[:2]:routed,receipt=route_v1(output,bank,method,predicted)
                    else:routed,receipt=route_v2(output,bank,method,predicted,mixture)
                    metrics=scoring(routed);assert metrics['sum_relative_error']<1e-9
                    p=OUT/f'{method}_{index:03d}_ORDERS.json';write(p,receipt)
                    methods[method]=dict(metrics=metrics,orders_sha256=sha(p));del routed;gc.collect();torch.cuda.empty_cache()
            result=dict(index=index,categories=case['categories'],nominal_levels_db=case['nominal_levels_db'],
                reused_parent_output=reused,parent_output_sha256=sha(path),parent=baseline,methods=methods)
            write(OUT/f'CASE_{index:03d}.json',result);rows.append(result)
            print('CASE_COMPLETE',index,baseline['nmse'],methods[plan['primary']]['metrics']['nmse'],flush=True)
            del output,refs,mixture,ref,mix;gc.collect();torch.cuda.empty_cache()
        summaries={'parent':summarize([r['parent'] for r in rows])}
        for method in METHODS:summaries[method]=summarize([r['methods'][method]['metrics'] for r in rows])
        parent=summaries['parent']['by_count'];primary=summaries[plan['primary']]['by_count']
        checks={}
        for k in (2,3):
            a,b=parent[k-1],primary[k-1]
            checks[f'count{k}_nmse_lower']=b['mean_nmse']<a['mean_nmse']
            checks[f'count{k}_si_higher']=b['mean_si_sdr'] is not None and a['mean_si_sdr'] is not None and b['mean_si_sdr']>a['mean_si_sdr']
            checks[f'count{k}_weak_nmse_lower']=b['weakest_nmse']<a['weakest_nmse']
        checks['single_nmse_not_worse']=primary[0]['mean_nmse']<=parent[0]['mean_nmse']
        verify();result=dict(status='VERIFIED',cases=rows,summaries=summaries,primary_checks=checks,
            plan_sha256=sha(OUT/'PLAN.json'),optimizer_updates=0,references_in_routing=False,
            heldout_access=False,scope=plan['scope'])
        write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),
            source_input_hashes=True,parent_metrics_reproduced=True,case_count=24))
        write(ROOT/'reports/2026-10-13/SPECTRAL_ROUTING_24_RESULT.json',result)
        state('COMPLETE_AUDITED',cases=24,primary_checks=checks);print('COMPLETE',summaries,checks,flush=True)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
