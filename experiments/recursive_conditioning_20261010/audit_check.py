"""Check saved full-model receipts against the earlier successive check.

This rechecks files, schemas, hashes and recorded numerical equivalence; it
does not rerun inference and must not be described as independent training.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(Path(p).read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(root):
    p=read(root/'PROTOCOL.json');r=read(root/'COMPLETE.json')
    assert r['status']=='PASS' and r['protocol_sha256']==digest(root/'PROTOCOL.json')
    assert r==read(ROOT/'reports/2026-10-10/ORIGINAL_CONDITIONING_FULL_CHECK.json')
    assert r['optimizer_steps']==0 and not r['gpu_use'] and not r['heldout_read']
    assert r['full_model'] and r['full_input_samples']==63872 and r['activation_checkpointing']
    assert r['parameters']==r['base_parameters']+r['extra_parameters']==32145163
    for rel,sha in p['sources'].items():assert digest(ROOT/rel)==digest(root/'source_snapshot'/rel)==sha
    algebra=ROOT/'reports/2026-10-10/ORIGINAL_CONDITIONING_ALGEBRA_CHECK.json'
    assert digest(algebra)==p['algebra_sha256'] and read(algebra)['status']=='PASS'
    old=read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    assert digest(old['parent_checkpoint'])==p['parent_sha256']==old['parent_checkpoint_sha256']
    assert digest(Path(old['preparation'])/'PREPARATION.json')==p['preparation_sha256']==old['preparation_sha256']
    prior_path=ROOT/'reports/2026-10-10/SUCCESSIVE_FULL_MODEL_CHECK.json'
    prior={a['index']:a for a in read(prior_path)['rows']}
    assert [a['index'] for a in r['rows']]==p['indices']==[0,4,2]
    assert [a['count'] for a in r['rows']]==[1,2,3]
    deltas=[]
    for a in r['rows']:
        b=prior[a['index']]
        assert all(a[k] for k in ('initial_waveforms_exact','initial_count_logits_exact',
                                  'all_parameter_gradients_finite','no_leaked_hooks','weights_unchanged'))
        assert math.isfinite(a['adapter_gradient_norm']) and a['adapter_gradient_norm']>0
        objective=abs(a['objective']-b['objective'])
        residual=abs(a['residual_gradient_norm']-b['residual_gradient_norm'])
        assert objective<2e-6 and residual<2e-7
        for key in ('nmse','si_sdr','reference_power'):
            assert len(a['metrics'][key])==a['count']
            assert max(abs(x-y) for x,y in zip(a['metrics'][key],b['successive_initial'][key]))<2e-5
        deltas.append(dict(index=a['index'],objective_absolute_difference=objective,
                           residual_gradient_norm_difference=residual))
    report=dict(status='PASS',protocol_sha256=digest(root/'PROTOCOL.json'),
        complete_sha256=digest(root/'COMPLETE.json'),prior_check_sha256=digest(prior_path),
        auditor_sha256=digest(Path(__file__)),deltas=deltas,all_source_hashes_checked=True,
        parent_and_preparation_hashes_checked=True,optimizer_steps=0,recorded_iq_reads=0,
        independently_reran_inference=False,heldout_read=False,trained_result=False)
    out=ROOT/'reports/2026-10-10/ORIGINAL_CONDITIONING_CHECK_AUDIT.json'
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args().run.resolve())
