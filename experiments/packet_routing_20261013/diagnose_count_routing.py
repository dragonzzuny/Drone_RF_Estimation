"""Oracle-count diagnostic of a fixed postprocessor; NEVER a deployed method."""
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha
from numpy_waveform_score import score
from spectral_routing_constrained import route
from native_full_context import components

HERE=Path(__file__).resolve().parent
C=ROOT/'local/spectral_routing_c24_20261013_v1'
PREVIOUS=ROOT/'local/constrained_routing_20261013_v1'
OUT=ROOT/'local/count_routing_diagnosis_20261013_v1'


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def verify_metrics(actual,expected):
    count=expected['count']
    assert actual['assignment'][:count]==expected['assignment'][:count]
    assert actual['weakest_index']==expected['weakest_index']
    for key in ('nmse','si_sdr','reference_power'):
        assert all(v is not None for v in actual[key]+expected[key])
        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-6,atol=1e-7)
    assert actual['sum_relative_error']<1e-9


def summary(rows):
    if not rows:return None
    vals=[v for row in rows for v in row['si_sdr']]
    return dict(cases=len(rows),mean_nmse=float(np.mean([v for row in rows for v in row['nmse']])),
        mean_si_sdr=float(np.mean(vals)) if all(v is not None for v in vals) else None,
        weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in rows])),
        original_count_accuracy=float(np.mean([r['count']==r['predicted_count'] for r in rows])))


def run():
    original=read(C/'RESULT.json');previous=read(PREVIOUS/'RESULT.json')
    assert all(read(d/'STATE.json')['status']=='COMPLETE_AUDITED' for d in (C,PREVIOUS))
    check=read(ROOT/'reports/2026-10-13/NUMPY_WAVEFORM_SCORE_CPU_CHECK.json')
    assert check['status']=='PASS' and check['source_sha256']==sha(HERE/'numpy_waveform_score.py')
    sources=dict(read(PREVIOUS/'PLAN.json')['source_sha256'])
    for p in (Path(__file__),HERE/'numpy_waveform_score.py',HERE/'test_numpy_waveform_score.py',
              ROOT/'reports/2026-10-13/COUNT_ROUTING_DIAGNOSIS_PLAN_KO.md'):
        sources[str(p.relative_to(ROOT))]=sha(p)
    files={str(p):sha(p) for p in (C/'RESULT.json',C/'AUDIT.json',PREVIOUS/'RESULT.json',PREVIOUS/'AUDIT.json')}
    assert files[str(C/'RESULT.json')]==read(C/'AUDIT.json')['result_sha256']
    assert files[str(PREVIOUS/'RESULT.json')]==read(PREVIOUS/'AUDIT.json')['result_sha256']
    assert not (OUT/'PLAN.json').exists()
    plan=dict(registered=time.time(),source_sha256=sources,input_sha256=files,cases=24,
        oracle_count_used=True,deployable=False,references_in_waveform_forward=False,
        optimizer_updates=0,heldout_access=False,metrics_rtol=1e-6,metrics_atol=1e-7)
    write(OUT/'PLAN.json',plan)
    for rel,digest in sources.items():
        p=OUT/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,p);assert sha(p)==digest
    def verify():
        assert all(sha(ROOT/p)==h and sha(OUT/'source_snapshot'/p)==h for p,h in sources.items())
        assert all(sha(p)==h for p,h in files.items())
    verify()
    sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
    from native_data import NativeMixtures
    data=NativeMixtures(Path(read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')['preparation']),
                        'validation_pack',1,use_features=True)
    old={r['index']:r for r in previous['groups']['C24']['cases']};rows=[]
    for position,case in enumerate(original['cases']):
        i=case['index'];state('CPU_DIAGNOSIS',case=position+1,total=24)
        path=C/f'OUTPUT_{i:03d}.npy';assert sha(path)==case['parent_output_sha256']
        output=np.load(path,mmap_mode='r',allow_pickle=False);refs=components(data,i);mix=refs.sum(0)
        count=case['parent']['count'];predicted=case['parent']['predicted_count']
        baseline=score(output,refs,mix,count,predicted);verify_metrics(baseline,case['parent'])
        deployed,receipt=route(output,predicted)
        before=score(deployed,refs,mix,count,predicted);verify_metrics(before,old[i]['candidate']);del deployed
        changed,receipt=route(output,count)
        after=score(changed,refs,mix,count,predicted)
        assert after['sum_relative_error']<1e-9
        if count==predicted:assert before==after
        write(OUT/f'ORACLE_COUNT_ORDERS_{i:03d}.json',dict(receipt,
            oracle_count_used=True,deployable=False,original_predicted_count=predicted))
        row=dict(index=i,categories=case['categories'],nominal_levels_db=case['nominal_levels_db'],
            original_count_correct=count==predicted,parent=baseline,
            predicted_count_routing=before,oracle_count_routing=after)
        write(OUT/f'CASE_{i:03d}.json',row);rows.append(row)
        del output,refs,mix,changed;gc.collect()
    summaries={}
    for key in ('parent','predicted_count_routing','oracle_count_routing'):
        summaries[key]={str(k):summary([r[key] for r in rows if r[key]['count']==k]) for k in (1,2,3)}
        summaries[key]['count_wrong']=summary([r[key] for r in rows if not r['original_count_correct']])
        summaries[key]['count_correct']=summary([r[key] for r in rows if r['original_count_correct']])
    verify();result=dict(status='VERIFIED',cases=rows,summaries=summaries,oracle_count_used=True,
        deployable=False,optimizer_updates=0,heldout_access=False,plan_sha256=sha(OUT/'PLAN.json'),
        baseline_metrics_reproduced=True,interpretation='Effect of true construction count on fixed routing of existing estimates ONLY')
    write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),
        all24_parent_and_previous_active_assignments_reproduced=True,source_input_hashes=True))
    write(ROOT/'reports/2026-10-13/COUNT_ROUTING_DIAGNOSIS_RESULT.json',result)
    state('COMPLETE_AUDITED',cases=24);print('COUNT_DIAGNOSIS_COMPLETE',summaries,flush=True)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        state('FAILED');raise
