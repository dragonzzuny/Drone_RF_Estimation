"""Posthoc score decomposition; reference-aided optimum is diagnostic ONLY.

The oracle minimizes full-observation source-normalized NMSE over block-wise
hard permutations of existing estimates. It is NOT a waveform separation floor.
"""
import argparse
import gc
import itertools
from pathlib import Path
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha
from sequence_data import Sequences,ROLES,WINDOW,LENGTH

PERMS=list(itertools.permutations(range(3)))


def decompose(output,refs,case,orders):
    count=case['parent']['count'];power=np.asarray(case['parent']['reference_power'])
    assert np.all(power>0)
    parent_order=case['parent']['assignment'];method=case['methods']['slot_initialized_clusters']['metrics']
    routed_order=method['assignment'];rows=[]
    for receipt in orders:
        a,b=receipt['start'],receipt['stop']
        y=output[:3,a:b].astype(np.complex128);r=refs[:count,a:b].astype(np.complex128)
        yp=np.sum(abs(y)**2,axis=1);rp=np.sum(abs(r)**2,axis=1)
        cross=r.conj()@y.T;sse=rp[:,None]+yp[None,:]-2*cross.real
        assert np.min(sse)>-1e-10*max(yp.max(),rp.max(),1.)
        costs=np.maximum(sse,0)/(power[:,None]*LENGTH)
        actual_order=[receipt['permutation'][routed_order[j]] for j in range(count)]
        alternatives=np.array([[costs[j,p[j]] for j in range(count)] for p in PERMS])
        best=int(alternatives.mean(1).argmin())
        base=[float(costs[j,parent_order[j]]) for j in range(count)]
        routed=[float(costs[j,actual_order[j]]) for j in range(count)]
        oracle=alternatives[best].tolist()
        rows.append(dict(start=a,stop=b,parent_nmse_contribution=base,routed_nmse_contribution=routed,
            oracle_nmse_contribution=oracle,parent_active_order=parent_order[:count],
            actual_active_order=actual_order,oracle_active_order=list(PERMS[best][:count]),
            contribution_change=float(np.mean(routed)-np.mean(base)),reference_block_power=(rp/(b-a)).tolist(),
            global_power=power.tolist(),reference_energy_fraction=(rp/(power*LENGTH)).tolist()))
    sums={name:np.sum([r[name+'_nmse_contribution'] for r in rows],axis=0) for name in ('parent','routed','oracle')}
    np.testing.assert_allclose(sums['parent'],case['parent']['nmse'],rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(sums['routed'],method['nmse'],rtol=1e-9,atol=1e-10)
    assert sums['oracle'].mean()<=min(sums['parent'].mean(),sums['routed'].mean())+1e-10
    return dict(index=case['index'],categories=case['categories'],nominal_levels_db=case['nominal_levels_db'],
        count=count,predicted_count=case['parent']['predicted_count'],
        parent_nmse=sums['parent'].tolist(),routed_nmse=sums['routed'].tolist(),
        oracle_mean_nmse=float(sums['oracle'].mean()),oracle_per_source_nmse=sums['oracle'].tolist(),
        mean_remaining_routing_gap=float((sums['routed']-sums['oracle']).mean()),blocks=rows)


def run(directory,output):
    result=read(directory/'RESULT.json');assert result['status']=='VERIFIED'
    assert sha(directory/'RESULT.json')==read(directory/'AUDIT.json')['result_sha256']
    data=Sequences(ROLES[1]);rows=[]
    for case in result['cases']:
        index=case['index'];p=directory/f'OUTPUT_{index:03d}.npy'
        if case['reused_parent_output']:p=ROOT/'local/full_context_parent_20261013_v1'/p.name
        assert sha(p)==case['parent_output_sha256']
        order_path=directory/f'slot_initialized_clusters_{index:03d}_ORDERS.json'
        assert sha(order_path)==case['methods']['slot_initialized_clusters']['orders_sha256']
        y=np.load(p,mmap_mode='r',allow_pickle=False);refs=data.components(index,True)
        rows.append(decompose(y,refs,case,read(order_path)['orders']))
        del y,refs;gc.collect()
    public=dict(status='VERIFIED',cases=rows,input_sha256=sha(directory/'RESULT.json'),source_sha256=sha(Path(__file__)),
        parent_and_routed_full_nmse_reproduced=True,references_used_for_diagnosis_only=True,
        oracle='Per-block minimum full-observation normalized NMSE over six permutations of existing outputs; not deployable, not unrestricted separation floor',
        input_waveforms_changed=False,heldout_access=False,posthoc_analysis=True)
    assert not output.exists();write(output,public)
    print('DECOMPOSITION_COMPLETE',[(r['index'],np.mean(r['parent_nmse']),np.mean(r['routed_nmse']),r['oracle_mean_nmse']) for r in rows],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.directory,a.output)
