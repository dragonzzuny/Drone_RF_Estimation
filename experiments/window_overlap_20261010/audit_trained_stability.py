"""Verify fixed-role/count coverage and all saved stability aggregations."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(root):
    p=read(root/'PROTOCOL.json');result=read(root/'COMPLETE.json');rows=result['rows']
    assert result['status']=='COMPLETE_CHECKED' and result['protocol_sha256']==digest(root/'PROTOCOL.json')
    assert len(rows)==36 and len({(r['model'],r['role'],r['index']) for r in rows})==36
    assert not result['heldout_read'] and result['updates']==0 and result['dev_reused']
    for model in p['checkpoint_sha256']:
        for role,indices in p['indices'].items():
            selected=[r for r in rows if r['model']==model and r['role']==role]
            assert {r['index'] for r in selected}==set(indices)
            for row in selected:
                assert row['common_samples']==47488 and abs(row['offset'])==16384
                assert len(row['per_source_relative_drift'])==row['count']
                assert row['consistency']>=0 and row['drift_over_mixture_energy']>=0
                for key in ('original_window','shifted_window'):
                    m=row[key];assert m['sum_relative_error']<1e-10 and m['count']==row['count']
                    for n,e,v in zip(m['nmse'],m['absolute_error_power'],m['reference_power']):assert abs(n-e/v)<1e-12
    for summary in result['summary']:
        selected=[r for r in rows if all(r[k]==summary[k] for k in ('model','role','count'))]
        assert len(selected)==summary['cases']==2
        for key in ('consistency','drift_over_mixture_energy'):
            assert abs(summary[key]-statistics.mean(r[key] for r in selected))<1e-12
        for key,mode in [('original_nmse','original_window'),('shifted_nmse','shifted_window')]:
            assert abs(summary[key]-statistics.mean(v for r in selected for v in r[mode]['nmse']))<1e-12
    for rel,sha in p['sources'].items():assert digest(ROOT/rel)==digest(root/'source_snapshot'/rel)==sha
    assert digest(root/'COMPLETE.json')==digest(ROOT/'reports/2026-10-10/TRAINED_WINDOW_STABILITY_RESULT.json')
    out=dict(status='PASS',protocol_sha256=digest(root/'PROTOCOL.json'),result_sha256=digest(root/'COMPLETE.json'),
        rows=36,all_18_conditions_reaggregated=True,all_reference_error_ratios_checked=True,
        source_snapshots_checked=True,independent_waveform_rerun=False,heldout_read=False,
        auditor_sha256=digest(Path(__file__)))
    (root/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    (ROOT/'reports/2026-10-10/TRAINED_WINDOW_STABILITY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run.resolve())
