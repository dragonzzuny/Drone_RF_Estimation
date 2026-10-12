"""Fixed three-slot tracking; predicted/true counts do not enter routing."""
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha
from diagnose_count_routing import verify_metrics,summary
from numpy_waveform_score import score
from spectral_routing_constrained import route
from native_full_context import components
from sequence_data import Sequences,ROLES

HERE=Path(__file__).resolve().parent
B=ROOT/'local/spectral_routing_24_20261013_v1'
C=ROOT/'local/spectral_routing_c24_20261013_v1'
PREVIOUS=ROOT/'local/constrained_routing_20261013_v1'
OUT=ROOT/'local/count_free_routing_20261013_v1'


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def run():
    old=read(PREVIOUS/'RESULT.json')
    assert all(read(d/'STATE.json')['status']=='COMPLETE_AUDITED' for d in (B,C,PREVIOUS))
    assert read(ROOT/'local/count_routing_diagnosis_20261013_v1/STATE.json')['status']=='COMPLETE_AUDITED'
    sources=dict(read(ROOT/'local/count_routing_diagnosis_20261013_v1/PLAN.json')['source_sha256'])
    for p in (Path(__file__),ROOT/'reports/2026-10-13/COUNT_FREE_ROUTING_PLAN_KO.md'):
        sources[str(p.relative_to(ROOT))]=sha(p)
    files={str(p):sha(p) for d in (B,C,PREVIOUS) for p in (d/'RESULT.json',d/'AUDIT.json')}
    for d in (B,C,PREVIOUS):assert files[str(d/'RESULT.json')]==read(d/'AUDIT.json')['result_sha256']
    assert not (OUT/'PLAN.json').exists()
    plan=dict(registered=time.time(),source_sha256=sources,input_sha256=files,
        groups=['B24','C24'],cases_per_group=24,fixed_clusters=3,oracle_count_used=False,
        predicted_count_in_routing=False,optimizer_updates=0,heldout_access=False,posthoc_development=True)
    write(OUT/'PLAN.json',plan)
    for rel,digest in sources.items():
        p=OUT/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,p);assert sha(p)==digest
    def verify():
        assert all(sha(ROOT/p)==h and sha(OUT/'source_snapshot'/p)==h for p,h in sources.items())
        assert all(sha(p)==h for p,h in files.items())
    verify();groups={}
    sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
    from native_data import NativeMixtures
    for group,directory in (('B24',B),('C24',C)):
        original=read(directory/'RESULT.json');rows=[];out=OUT/group;out.mkdir()
        previous={v['index']:v for v in old['groups'][group]['cases']}
        if group=='B24':data=Sequences(ROLES[1])
        else:data=NativeMixtures(Path(read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')['preparation']),
                                'validation_pack',1)
        for position,case in enumerate(original['cases']):
            i=case['index'];state('CPU_ROUTING',group=group,case=position+1,total=24)
            path=directory/f'OUTPUT_{i:03d}.npy'
            if case.get('reused_parent_output'):path=ROOT/'local/full_context_parent_20261013_v1'/path.name
            assert sha(path)==case['parent_output_sha256']
            output=np.load(path,mmap_mode='r',allow_pickle=False)
            refs=data.components(i,True) if group=='B24' else components(data,i);mix=refs.sum(0)
            count=case['parent']['count'];predicted=case['parent']['predicted_count']
            parent=score(output,refs,mix,count,predicted);verify_metrics(parent,case['parent'])
            routed,_=route(output,predicted)
            before=score(routed,refs,mix,count,predicted);verify_metrics(before,previous[i]['candidate']);del routed
            # All three slots always enter the algorithm, including low-energy
            # placeholders. The ground-truth/predicted counts are NOT supplied.
            changed,receipt=route(output,3)
            after=score(changed,refs,mix,count,predicted);assert after['sum_relative_error']<1e-9
            if predicted==3:assert before==after
            receipt.pop('predicted_count')
            write(out/f'ORDERS_{i:03d}.json',dict(receipt,fixed_cluster_count=3,oracle_count_used=False,
                predicted_count_in_routing=False,method='count_free_constrained_three_slots'))
            row=dict(index=i,categories=case['categories'],nominal_levels_db=case['nominal_levels_db'],
                parent=parent,previous=before,candidate=after)
            write(out/f'CASE_{i:03d}.json',row);rows.append(row)
            del output,refs,mix,changed;gc.collect()
        summaries={key:{str(k):summary([v[key] for v in rows if v[key]['count']==k]) for k in (1,2,3)}
                   for key in ('parent','previous','candidate')}
        checks={}
        for k in (2,3):
            a,b=summaries['previous'][str(k)],summaries['candidate'][str(k)]
            checks[f'count{k}_nmse_lower']=b['mean_nmse']<a['mean_nmse']
            checks[f'count{k}_si_higher']=b['mean_si_sdr']>a['mean_si_sdr']
            checks[f'count{k}_weak_nmse_lower']=b['weakest_nmse']<a['weakest_nmse']
        checks['single_nmse_not_worse']=summaries['candidate']['1']['mean_nmse']<=summaries['previous']['1']['mean_nmse']
        groups[group]=dict(cases=rows,summaries=summaries,checks=checks)
        del data;gc.collect()
    verify();result=dict(status='VERIFIED',groups=groups,plan_sha256=sha(OUT/'PLAN.json'),
        optimizer_updates=0,reference_or_predicted_count_in_routing=False,heldout_access=False,
        scope='Already exposed B24/C24 development; fixed maximum-output tracking, not a new source-count estimator')
    write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),
        baseline_metrics_and_active_assignments_reproduced=True,cases=48,source_input_hashes=True))
    write(ROOT/'reports/2026-10-13/COUNT_FREE_ROUTING_RESULT.json',result)
    state('COMPLETE_AUDITED',cases=48)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        state('FAILED');raise
