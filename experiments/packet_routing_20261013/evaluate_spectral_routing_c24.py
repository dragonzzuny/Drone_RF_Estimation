"""Unchanged routing on24 predefined strata of already-exposed separate-record C."""
import fcntl
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from evaluate_parent_full_context import ROOT,read,write,sha,LOCK

HERE=Path(__file__).resolve().parent
OUT=ROOT/'local/spectral_routing_c24_20261013_v1'
B24=ROOT/'local/spectral_routing_24_20261013_v1'
OLD_C=ROOT/'local/b99_on_c_20261013_v1'
FIT=ROOT/'local/b_timeblock_spectral_20261013_v1'
METHODS=['fit_dictionary','observation_clusters','count_dictionary','slot_initialized_clusters']


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def run():
    assert read(B24/'STATE.json')['status']=='COMPLETE_AUDITED'
    assert sha(B24/'RESULT.json')==read(B24/'AUDIT.json')['result_sha256']
    check=ROOT/'reports/2026-10-13/NATIVE_FULL_CONTEXT_CPU_CHECK.json';test=read(check);assert test['status']=='PASS'
    for name,digest in test['source_sha256'].items():assert sha(HERE/name)==digest
    sources=dict(read(B24/'PLAN.json')['source_sha256'])
    for name,digest in read(OLD_C/'PROTOCOL.json')['source_sha256'].items():
        if name in sources:assert sources[name]==digest
        sources[name]=digest
    for p in (Path(__file__),HERE/'native_full_context.py',HERE/'test_native_full_context.py',
              ROOT/'reports/2026-10-13/SPECTRAL_ROUTING_C24_PLAN_KO.md'):
        sources[str(p.relative_to(ROOT))]=sha(p)
    files=dict(read(OLD_C/'PROTOCOL.json')['pinned_files'])
    for p in (B24/'RESULT.json',FIT/'TEMPLATES.json',check,OLD_C/'RESULT.json'):files[str(p)]=sha(p)
    checkpoint=ROOT/'local/timeblock_extend100_20261012_v1/ACTUAL_099.pt'
    prep=Path(read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')['preparation'])
    plan=dict(registered=time.time(),source_sha256=sources,input_sha256=files,indices=test['indices'],
        methods=METHODS,primary='slot_initialized_clusters',checkpoint_epoch=99,
        samples=2088960,fs_hz=100000000,optimizer_updates=0,references_in_routing=False,
        count_from_model=True,heldout_access=False,posthoc_development=True,
        independent_recording_groups_from_B_fit=True,already_exposed_C_development=True,
        bandwidth_shift_mhz=[10,20],new_independent_final_test=False,current_best_changed=False,
        scope='All24 predefined C category/count/power strata, separate recordings and changed bandwidth; no tuning')
    OUT.mkdir(exist_ok=True);assert not (OUT/'PLAN.json').exists();write(OUT/'PLAN.json',plan)
    for name,digest in sources.items():
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p);assert sha(p)==digest
    def verify():
        assert all(sha(ROOT/p)==h and sha(OUT/'source_snapshot'/p)==h for p,h in sources.items())
        assert all(sha(p)==h for p,h in files.items())
    with LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);verify()
        import numpy as np
        import torch
        from sequence_data import ROLES
        from models import build,predict
        from full_context_eval import evaluate
        from sequence_eval import score,summarize
        from spectral_routing import route as route_v1
        from spectral_routing_v2 import route as route_v2
        from native_full_context import components,selected_indices
        sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
        from native_data import NativeMixtures
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        data=NativeMixtures(prep,'validation_pack',1);assert selected_indices(data)==plan['indices']
        prior=read(OLD_C/'RESULT.json');assert data.rows_hash==prior['rows_sha256']
        net=build('unet_mean').cuda().eval().requires_grad_(False)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);net.load_state_dict(saved['model']);del saved
        banks=read(FIT/'TEMPLATES.json')['banks'];rows=[]
        for position,index in enumerate(plan['indices']):
            raw=data.rows[index];count=int(raw['count']);refs=components(data,index);mixture=refs.sum(0)
            def progress(done,total):state('C_FULL_INFERENCE',case=position+1,total_cases=24,index=index,windows=done,total_windows=total)
            output,case=evaluate(net,predict,refs,np.array(data.features[index]),count,'cuda',progress)
            predicted=case['metrics']['predicted_count'];assert predicted==prior['rows'][index]['predicted_count']
            path=OUT/f'OUTPUT_{index:03d}.npy';np.save(path,output,allow_pickle=False)
            case.update(index=index,categories=[data.library.clips[int(i)]['category'] for i in raw['indices'][:count]],
                nominal_levels_db=raw['levels'][:count].tolist(),output_sha256=sha(path))
            write(OUT/f'PARENT_{index:03d}.json',case)
            centers={data.library.clips[int(j)]['common_center_hz'] for j in raw['indices'][:count]};assert len(centers)==1
            bank=banks[str(int(next(iter(centers))))]
            ref=torch.from_numpy(refs)[None].cuda();mix=torch.from_numpy(mixture)[None].cuda()
            logits=torch.zeros(1,3,device='cuda');logits[0,predicted-1]=1
            def scoring(value):return score(torch.from_numpy(value)[None].cuda(),ref,mix,count,logits)
            with torch.no_grad():
                baseline=scoring(output);assert baseline==case['metrics'];methods={}
                for method in METHODS:
                    state('C_ROUTING',case=position+1,total_cases=24,index=index,method=method)
                    if method in METHODS[:2]:routed,receipt=route_v1(output,bank,method,predicted)
                    else:routed,receipt=route_v2(output,bank,method,predicted,mixture)
                    metrics=scoring(routed);assert metrics['sum_relative_error']<1e-9
                    p=OUT/f'{method}_{index:03d}_ORDERS.json';write(p,receipt)
                    methods[method]=dict(metrics=metrics,orders_sha256=sha(p));del routed;gc.collect();torch.cuda.empty_cache()
            result=dict(index=index,categories=case['categories'],nominal_levels_db=case['nominal_levels_db'],
                parent_output_sha256=sha(path),parent=baseline,methods=methods)
            write(OUT/f'CASE_{index:03d}.json',result);rows.append(result)
            print('C_CASE_COMPLETE',index,count,predicted,baseline['nmse'],methods[plan['primary']]['metrics']['nmse'],flush=True)
            del output,refs,mixture,ref,mix;gc.collect();torch.cuda.empty_cache()
        summaries={'parent':summarize([r['parent'] for r in rows])}
        for method in METHODS:summaries[method]=summarize([r['methods'][method]['metrics'] for r in rows])
        parent=summaries['parent']['by_count'];primary=summaries[plan['primary']]['by_count'];checks={}
        for k in (2,3):
            a,b=parent[k-1],primary[k-1]
            checks[f'count{k}_nmse_lower']=b['mean_nmse']<a['mean_nmse']
            checks[f'count{k}_si_higher']=b['mean_si_sdr'] is not None and a['mean_si_sdr'] is not None and b['mean_si_sdr']>a['mean_si_sdr']
            checks[f'count{k}_weak_nmse_lower']=b['weakest_nmse']<a['weakest_nmse']
        checks['single_nmse_not_worse']=primary[0]['mean_nmse']<=parent[0]['mean_nmse']
        verify();result=dict(status='VERIFIED',cases=rows,summaries=summaries,primary_checks=checks,
            plan_sha256=sha(OUT/'PLAN.json'),optimizer_updates=0,references_in_routing=False,
            heldout_access=False,current_best_changed=False,scope=plan['scope'])
        write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),
            pinned_sources_and_inputs=True,case_count=24,unchanged_count_predictions=True,original_target_crops_exact=True))
        write(ROOT/'reports/2026-10-13/SPECTRAL_ROUTING_C24_RESULT.json',result)
        state('COMPLETE_AUDITED',cases=24,primary_checks=checks);print('C_COMPLETE',summaries,checks,flush=True)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
