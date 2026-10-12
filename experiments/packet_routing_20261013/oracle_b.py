"""Reference-informed B DEV diagnosis, explicitly NOT deployable separation.

Reuse the previously checked mask algebra without importing its old trainer.
The masks optimize a restricted per-bin problem, not waveform NMSE or a
fundamental identifiability bound. No learned checkpoint is changed.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/bandwidth_timeblock_20261011'))
from data import Mixtures,ROLES,MANIFEST
from drone_rf.waveform import analyze,synthesize,complex_si_sdr
from drone_rf.data import sha256
from drone_rf.context_training_data import write_json

OUT=ROOT/'local/b_oracle_allocation_20261013_v1'
ALGEBRA=ROOT/'experiments/source_interaction_20261010/oracle_allocation.py'
METHODS=('oracle_power','oracle_phase_sensitive_simplex')


def helpers():
    nodes=[n for n in ast.parse(ALGEBRA.read_text()).body if isinstance(n,ast.FunctionDef)
           and n.name in ('simplex','allocations','checks')]
    assert len(nodes)==3
    scope={'torch':torch}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ALGEBRA),'exec'),scope)
    return scope['allocations'],scope['checks']()


def summary(rows):
    out=[]
    for method in METHODS:
        for k in (2,3):
            group=[r for r in rows if r['count']==k]
            si=[v for r in group for v in r['results'][method]['si_sdr']]
            out.append(dict(method=method,count=k,cases=len(group),
                mean_nmse=float(np.mean([v for r in group for v in r['results'][method]['nmse']])),
                mean_si_sdr=float(np.mean(si)) if all(v is not None for v in si) else None,
                undefined_si_sources=sum(v is None for v in si),
                weakest_nmse=float(np.mean([r['results'][method]['nmse'][r['weakest_index']] for r in group]))))
    return out


@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=True);assert not (OUT/'PLAN.json').exists()
    alloc,checks=helpers();data=Mixtures(ROLES[1]);parent=ROOT/'local/timeblock_extend100_20261012_v1'
    prior=json.loads((parent/'VALIDATION_097.json').read_text())
    assert prior['rows_sha256']==data.rows_hash
    sources={str(p):sha256(p) for p in [Path(__file__),ALGEBRA,MANIFEST]}
    plan=dict(time=time.time(),source_sha256=sources,rows_sha256=data.rows_hash,checks=checks,
        methods=list(METHODS),cases=420,reference_informed=True,deployable=False,optimizer_updates=0,
        target='All previously exposed two/three source B DEV cases, original reference contributions including receiver noise',
        precision='complex128 diagnostic arithmetic, same512/128/sqrtHann STFT',
        assignment='Known construction identity; oracle is not rematched by PIT',
        no_heldout=True,independent_recordings=False,
        limitation='No waveform optimum, unrestricted complex estimator bound, or noise identifiability claim')
    write_json(OUT/'PLAN.json',plan);rows=[]
    for i,raw in enumerate(data.rows):
        k=int(raw['count'])
        if k==1:continue
        item=data[i];ref=torch.from_numpy(item['references'][:k]).to(torch.complex128)
        mix=torch.from_numpy(item['mixture']).to(torch.complex128);power=ref.abs().square().mean(-1)
        np.testing.assert_allclose(power.numpy(),prior['rows'][i]['reference_power'],rtol=1e-12,atol=1e-12)
        spectra=analyze(ref);z=analyze(mix);masks=alloc(spectra,z)
        roundtrip=float(((synthesize(spectra,ref.shape[-1])-ref).abs().square().mean(-1)/power).max())
        assert roundtrip<1e-12
        result={};cost={}
        for method in METHODS:
            mask=masks[method];assert torch.isfinite(mask).all() and bool((mask>=0).all())
            torch.testing.assert_close(mask.sum(0),torch.ones_like(mask[0]),rtol=1e-10,atol=1e-10)
            estimate=synthesize(mask*z,mix.numel());nmse=((estimate-ref).abs().square().mean(-1)/power).tolist()
            si=complex_si_sdr(estimate,ref).tolist()
            residual=float((estimate.sum(0)-mix).abs().square().mean()/mix.abs().square().mean())
            assert residual<1e-12 and np.isfinite(nmse).all()
            cost[method]=float((mask*z-spectra).abs().square().sum())
            result[method]=dict(nmse=nmse,si_sdr=[v if np.isfinite(v) else None for v in si],sum_relative_error=residual)
        assert cost[METHODS[1]]<=cost[METHODS[0]]+1e-9*max(1.,cost[METHODS[0]])
        rows.append(dict(index=i,count=k,categories=prior['rows'][i]['categories'],
            reference_power=power.tolist(),weakest_index=int(power.argmin()),
            results=result,stft_error=cost,roundtrip_max_nmse=roundtrip))
        if len(rows)%20==0:write_json(OUT/'STATE.json',dict(status='CPU_REFERENCE_MASK_DIAGNOSIS',cases=len(rows),total=420,pid=os.getpid()))
    assert len(rows)==420 and all(sha256(p)==h for p,h in sources.items())
    result=dict(status='COMPLETE',summary=summary(rows),rows=rows,plan_sha256=sha256(OUT/'PLAN.json'),
        reference_informed=True,deployable=False,limitation=plan['limitation'])
    write_json(OUT/'RESULT.json',result)
    audit=dict(status='PASS',result_sha256=sha256(OUT/'RESULT.json'),plan_sha256=sha256(OUT/'PLAN.json'),
        rows=420,checks=['Source hashes unchanged','Same B reference powers','STFT roundtrip','Masks nonnegative and sum to one',
            'Waveform mixture sum','Simplex per-bin cost no greater than power mask'])
    write_json(OUT/'AUDIT.json',audit);write_json(OUT/'STATE.json',dict(status='COMPLETE_AUDITED',pid=os.getpid()))
    public=ROOT/'reports/2026-10-13';write_json(public/'B_REFERENCE_MASK_DIAGNOSIS.json',dict(summary=result['summary'],
        reference_informed=True,deployable=False,limitation=plan['limitation'],audit=audit))
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(exist_ok=True);write_json(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
