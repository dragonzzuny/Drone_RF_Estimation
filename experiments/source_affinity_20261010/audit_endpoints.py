"""Reaggregate all saved small-sample affinity diagnoses; no new inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(Path(p).read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(root):
    p=read(root/'PROTOCOL.json');r=read(root/'COMPLETE.json')
    assert r['status']=='COMPLETE_CHECKED' and r['protocol_sha256']==digest(root/'PROTOCOL.json')
    assert r['optimizer_steps']==0 and r['inferences']==24 and not r['gpu_use'] and not r['heldout_read']
    assert len(r['rows'])==24 and len(r['summary'])==12
    assert r==read(ROOT/'reports/2026-10-10/SOURCE_AFFINITY_ENDPOINT_RESULT.json')
    study=Path(p['predecessor']);audit=ROOT/'reports/2026-10-10/SOURCE_AFFINITY_AUDIT.json'
    assert digest(audit)==r['audit_sha256'] and read(audit)['status']=='PASS'
    assert digest(study/'PROTOCOL.json')==p['predecessor_protocol_sha256']
    for rel,sha in p['sources'].items():assert digest(ROOT/rel)==digest(root/'source_snapshot'/rel)==sha
    for arm,sha in r['checkpoint_sha256'].items():assert digest(study/arm/'ACTUAL_001.pt')==sha
    reference={}
    for row in r['rows']:
        assert row['index'] in p['indices'][row['role']] and row['count'] in (2,3)
        key=(row['role'],row['index']);powers=row['waveform']['reference_power']
        if key in reference:assert reference[key]==powers
        else:reference[key]=powers
        assert len(powers)==len(row['waveform']['nmse'])==row['count']
        assert row['waveform']['sum_relative_error']<1e-10
        assert all(math.isfinite(row[k]) and row[k]>-1e-10 for k in ('hard_affinity','soft_affinity'))
        assert all(math.isfinite(v) and v>=0 for v in row['waveform']['nmse'])
    for summary in r['summary']:
        rows=[row for row in r['rows'] if all(row[k]==summary[k] for k in ('arm','role','count'))]
        assert len(rows)==summary['cases']==2 and len({row['index'] for row in rows})==2
        for key in ('hard_affinity','soft_affinity'):
            assert summary[key]==statistics.mean(row[key] for row in rows)
        assert summary['mean_nmse']==statistics.mean(v for row in rows for v in row['waveform']['nmse'])
    report=dict(status='PASS',protocol_sha256=digest(root/'PROTOCOL.json'),complete_sha256=digest(root/'COMPLETE.json'),
        auditor_sha256=digest(Path(__file__)),rows_checked=24,groups_checked=12,
        source_and_checkpoint_hashes_checked=True,same_reference_powers_checked=True,
        recorded_iq_reads=0,independently_reran_inference=False,heldout_read=False)
    (ROOT/'reports/2026-10-10/SOURCE_AFFINITY_ENDPOINT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args().run.resolve())
