"""Fixed quarter-block oracle diagnosis; no neural or deployable result."""
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha
from spectral_routing import PERMS,WINDOW
from sequence_data import Sequences,ROLES
from native_full_context import components

OUT=ROOT/'local/tracking_resolution_20261013_v1'
HERE=Path(__file__).resolve().parent
PREP=ROOT/'local/learned_tracking_preparation_20261013_v1'


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def minimum(output,refs,k,step):
    n=refs.shape[-1]
    energies=np.array([np.sum(abs(refs[j].astype(np.complex128))**2) for j in range(k)])
    total=np.zeros(k);orders=[]
    for a in range(0,n,step):
        b=min(a+step,n);r=refs[:k,a:b].astype(np.complex128);y=output[:3,a:b].astype(np.complex128)
        error=np.array([[np.sum(abs(y[j]-r[q])**2)/energies[q] for q in range(k)] for j in range(3)])
        values=np.array([sum(error[p[j],j] for j in range(k)) for p in PERMS])
        order=PERMS[int(values.argmin())];orders.append(list(order))
        total+=np.array([error[order[q],q] for q in range(k)])
    return dict(mean_nmse=float(total.mean()),component_nmse=total.tolist(),orders=orders,blocks=len(orders))


def run():
    OUT.mkdir(exist_ok=True)
    base=read(ROOT/'reports/2026-10-13/TRACKING_PERMUTATION_FLOOR.json')
    assert base['status']=='PASS' and base['references_used']
    sources=dict(read(PREP/'PREPARATION.json')['source_sha256'])
    for path in (Path(__file__),ROOT/'reports/2026-10-13/TRACKING_RESOLUTION_PLAN_KO.md',
                 HERE/'native_full_context.py',HERE/'sequence_data.py'):
        sources[str(path.relative_to(ROOT))]=sha(path)
    inputs={str(path):sha(path) for path in (PREP/'RESULT.json',PREP/'AUDIT.json',
        ROOT/'reports/2026-10-13/TRACKING_PERMUTATION_FLOOR.json')}
    assert not (OUT/'PLAN.json').exists()
    write(OUT/'PLAN.json',dict(registered=time.time(),source_sha256=sources,input_sha256=inputs,
        coarse_samples=WINDOW,fine_samples=WINDOW//4,references_used=True,deployable=False,
        cases=48,optimizer_updates=0,heldout_access=False))
    for name,h in sources.items():
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p);assert sha(p)==h
    sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
    from native_data import NativeMixtures
    groups={}
    for group,items in read(PREP/'RESULT.json')['groups'].items():
        previous={r['index']:r for r in base['groups'][group]['cases']};rows=[]
        data=Sequences(ROLES[1]) if group=='B24' else NativeMixtures(
            Path(read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')['preparation']),'validation_pack',1)
        for position,item in enumerate(items):
            i=item['index'];state('CPU_ORACLE_DIAGNOSIS',group=group,case=position+1,total=24)
            path=Path(item['metadata']['parent_output']);assert sha(path)==item['metadata']['parent_output_sha256']
            output=np.load(path,mmap_mode='r',allow_pickle=False)
            refs=data.components(i,True) if group=='B24' else components(data,i)
            old=minimum(output,refs,item['count'],WINDOW)
            np.testing.assert_allclose(old['mean_nmse'],previous[i]['reference_assisted_minimum_nmse'],rtol=1e-10,atol=1e-10)
            result=minimum(output,refs,item['count'],WINDOW//4)
            assert result['mean_nmse']<=old['mean_nmse']+1e-10
            record=dict(index=i,count=item['count'],categories=item['metadata']['categories'],
                nominal_levels_db=item['metadata']['nominal_levels_db'],coarse=old,quarter=result)
            write(OUT/f'CASE_{group}_{i:03d}.json',record);rows.append(record)
            del output,refs;gc.collect()
        summaries={str(k):{name:float(np.mean([r[name]['mean_nmse'] for r in rows if r['count']==k]))
                          for name in ('coarse','quarter')} for k in (1,2,3)}
        groups[group]=dict(cases=rows,by_count=summaries);del data;gc.collect()
    assert all(sha(ROOT/name)==h for name,h in sources.items()) and all(sha(name)==h for name,h in inputs.items())
    result=dict(status='PASS',groups=groups,plan_sha256=sha(OUT/'PLAN.json'),
        coarse_samples=WINDOW,fine_samples=WINDOW//4,fs_hz=100000000,
        references_used=True,deployable=False,optimizer_updates=0,heldout_access=False,
        all48_coarse_scores_reproduced=True,all48_finer_bounds_monotone=True,
        caveat='Oracle min-NMSE over existing source-slot permutations only; not clean-signal limit or SI-SDR bound')
    write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json')))
    write(ROOT/'reports/2026-10-13/TRACKING_RESOLUTION_RESULT.json',result)
    state('COMPLETE_AUDITED',cases=48)
    print({g:r['by_count'] for g,r in groups.items()},flush=True)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        state('FAILED');raise
