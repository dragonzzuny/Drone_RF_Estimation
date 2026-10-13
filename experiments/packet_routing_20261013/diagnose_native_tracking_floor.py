"""Reference-assisted minimum NMSE over existing whole-block permutations.

This is NOT inference, a learned result, or a general separation/noise bound.
The identity labels in the diagnostic are supplied by references themselves.
"""
import os
from pathlib import Path
import time
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha
from spectral_routing import PERMS


def run():
    prep=ROOT/'local/native_tracking_cache_20261013_v1'
    assert read(prep/'STATE.json')['status']=='COMPLETE_AUDITED'
    assert sha(prep/'RESULT.json')==read(prep/'AUDIT.json')['result_sha256']
    groups={}
    for group,items in read(prep/'RESULT.json')['groups'].items():
        rows=[]
        for item in items:
            assert sha(item['scoring'])==item['scoring_sha256']
            k=item['count']
            with np.load(item['scoring'],allow_pickle=False) as cache:
                total=cache['energy_r'].sum(0)
                cost=cache['sse'][:,:,:k]/total[None,None,:k]
                values=np.array([[sum(cost[b,p[j],j] for j in range(k))/k for p in PERMS]
                                 for b in range(len(cache['bounds']))])
                choice=values.argmin(1);orders=np.array([PERMS[int(q)] for q in choice])
                component=np.array([sum(cost[b,orders[b,j],j] for b in range(len(orders))) for j in range(k)])
                floor=float(values.min(1).sum())
                np.testing.assert_allclose(floor,component.mean(),rtol=1e-12,atol=1e-12)
            baseline=float(np.mean(item['baseline_metrics']['nmse']))
            constrained=float(np.mean(item['constrained_baseline']['nmse']))
            assert floor<=min(baseline,constrained)+1e-9
            rows.append(dict(index=item['index'],count=k,categories=item['metadata']['categories'],
                nominal_levels_db=item['metadata']['nominal_levels_db'],parent_nmse=baseline,
                constrained_nmse=constrained,reference_assisted_minimum_nmse=floor,
                diagnostic_component_nmse=component.tolist(),reference_order_per_block=orders.tolist()))
        summaries={}
        for k in (1,2,3):
            selected=[r for r in rows if r['count']==k]
            summaries[str(k)]={key:float(np.mean([r[key] for r in selected])) for key in
                ('parent_nmse','constrained_nmse','reference_assisted_minimum_nmse')}
            summaries[str(k)]['cases']=len(selected)
        groups[group]=dict(cases=rows,by_count=summaries)
    result=dict(status='PASS',created=time.time(),pid=os.getpid(),groups=groups,
        source_sha256=sha(__file__),preparation_result_sha256=sha(prep/'RESULT.json'),
        references_used=True,deployable=False,model_training_updates=0,
        scope='Minimum mean component NMSE over permutations of3 existing output blocks of63872samples; background unchanged',
        caveat='Cannot infer learnability, clean-signal noise floor, performance of shorter-block routing, or optimal SI-SDR')
    write(ROOT/'reports/2026-10-13/NATIVE_TRACKING_PERMUTATION_FLOOR.json',result)
    print({group:value['by_count'] for group,value in groups.items()},flush=True)


if __name__=='__main__':run()
