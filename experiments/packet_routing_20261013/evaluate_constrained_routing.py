"""One new fixed assignment algorithm after B diagnostics, before C scores."""
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
OUT=ROOT/'local/constrained_routing_20261013_v1'
B=ROOT/'local/spectral_routing_24_20261013_v1'
C=ROOT/'local/spectral_routing_c24_20261013_v1'


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def checks(previous,candidate):
    result={}
    for k in (2,3):
        a,b=previous['by_count'][k-1],candidate['by_count'][k-1]
        result[f'count{k}_nmse_lower']=b['mean_nmse']<a['mean_nmse']
        result[f'count{k}_si_higher']=a['mean_si_sdr'] is not None and b['mean_si_sdr'] is not None and b['mean_si_sdr']>a['mean_si_sdr']
        result[f'count{k}_weak_nmse_lower']=b['weakest_nmse']<a['weakest_nmse']
    result['single_nmse_not_worse']=candidate['by_count'][0]['mean_nmse']<=previous['by_count'][0]['mean_nmse']
    return result


def run():
    assert read(B/'STATE.json')['status']=='COMPLETE_AUDITED'
    sources=dict(read(C/'PLAN.json')['source_sha256'])
    check=ROOT/'reports/2026-10-13/CONSTRAINED_ROUTING_CPU_CHECK.json';test=read(check);assert test['status']=='PASS'
    for name,digest in test['source_sha256'].items():assert sha(HERE/name)==digest
    for p in (Path(__file__),HERE/'spectral_routing_constrained.py',HERE/'test_spectral_routing_constrained.py',
              ROOT/'reports/2026-10-13/CONSTRAINED_ROUTING_PLAN_KO.md'):
        sources[str(p.relative_to(ROOT))]=sha(p)
    files={str(p):sha(p) for p in (check,B/'RESULT.json',B/'PLAN.json',C/'PLAN.json')}
    plan=dict(registered=time.time(),source_sha256=sources,input_sha256=files,method='constrained_slot_clusters',
        comparator='slot_initialized_clusters',optimizer_updates=0,heldout_access=False,
        groups=['B24','C24'],cases_per_group=24,references_in_routing=False,count_from_model=True,
        B_scores_seen_before_design=True,C_scores_not_inspected_before_registration=True,
        proxy_objective_not_reconstruction_loss=True)
    OUT.mkdir(exist_ok=True);assert not (OUT/'PLAN.json').exists();write(OUT/'PLAN.json',plan)
    for name,digest in sources.items():
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p);assert sha(p)==digest
    def verify():
        assert all(sha(ROOT/p)==h and sha(OUT/'source_snapshot'/p)==h for p,h in sources.items())
        assert all(sha(p)==h for p,h in files.items())
    while True:
        s=read(C/'STATE.json')
        if s['status']=='COMPLETE_AUDITED':break
        if s['status']=='FAILED':raise RuntimeError('C evaluation failed')
        state('WAITING_C24',case=s.get('case'));time.sleep(15)
    dependency=dict(C_result_sha256=sha(C/'RESULT.json'),C_audit_sha256=sha(C/'AUDIT.json'))
    assert dependency['C_result_sha256']==read(C/'AUDIT.json')['result_sha256'];write(OUT/'DEPENDENCY.json',dependency)
    with LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);verify()
        import numpy as np
        import torch
        from sequence_data import Sequences,ROLES
        from sequence_eval import score,summarize
        from spectral_routing_v2 import route as previous_route
        from spectral_routing_constrained import route
        from native_full_context import components
        sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
        from native_data import NativeMixtures
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        groups={}
        for name,directory in (('B24',B),('C24',C)):
            original=read(directory/'RESULT.json');assert sha(directory/'RESULT.json')==read(directory/'AUDIT.json')['result_sha256']
            if name=='B24':data=Sequences(ROLES[1])
            else:data=NativeMixtures(Path(read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')['preparation']),'validation_pack',1)
            rows=[];out=OUT/name;out.mkdir(exist_ok=True)
            for position,case in enumerate(original['cases']):
                index=case['index'];refs=data.components(index,True) if name=='B24' else components(data,index)
                path=directory/f'OUTPUT_{index:03d}.npy'
                if case.get('reused_parent_output'):path=ROOT/'local/full_context_parent_20261013_v1'/path.name
                assert sha(path)==case['parent_output_sha256'];output=np.load(path,allow_pickle=False);mixture=refs.sum(0)
                count=case['parent']['count'];predicted=case['parent']['predicted_count']
                ref=torch.from_numpy(refs)[None].cuda();mix=torch.from_numpy(mixture)[None].cuda()
                logits=torch.zeros(1,3,device='cuda');logits[0,predicted-1]=1
                def scoring(x):return score(torch.from_numpy(x)[None].cuda(),ref,mix,count,logits)
                with torch.no_grad():
                    baseline=scoring(output);assert baseline==case['parent']
                    old,_=previous_route(output,{},'slot_initialized_clusters',predicted,mixture)
                    old_score=scoring(old);assert old_score==case['methods']['slot_initialized_clusters']['metrics'];del old
                    state('ROUTING_AND_SCORING',group=name,case=position+1,total=24)
                    new,receipt=route(output,predicted);metrics=scoring(new);assert metrics['sum_relative_error']<1e-9
                    p=out/f'ORDERS_{index:03d}.json';write(p,receipt)
                    row=dict(index=index,categories=case['categories'],nominal_levels_db=case['nominal_levels_db'],
                        parent=baseline,previous=old_score,candidate=metrics,orders_sha256=sha(p),
                        proxy_objective=receipt['weighted_cosine_objective'])
                    write(out/f'CASE_{index:03d}.json',row);rows.append(row)
                del new,output,refs,mixture,ref,mix;gc.collect();torch.cuda.empty_cache()
            summary={key:summarize([r[key] for r in rows]) for key in ('parent','previous','candidate')}
            groups[name]=dict(cases=rows,summaries=summary,checks_against_previous=checks(summary['previous'],summary['candidate']))
            del data;gc.collect();print('GROUP_COMPLETE',name,summary,flush=True)
        verify();assert sha(C/'RESULT.json')==dependency['C_result_sha256']
        result=dict(status='VERIFIED',groups=groups,optimizer_updates=0,plan_sha256=sha(OUT/'PLAN.json'),
            dependency_sha256=sha(OUT/'DEPENDENCY.json'),references_in_routing=False,heldout_access=False,
            scope='Posthoc B development method and fixed C development transfer; not a new final test')
        write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),
            parent_and_previous_reproduced=True,sources_and_input_hashes_unchanged=True,cases=48))
        write(ROOT/'reports/2026-10-13/CONSTRAINED_ROUTING_RESULT.json',result)
        state('COMPLETE_AUDITED',cases=48)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
