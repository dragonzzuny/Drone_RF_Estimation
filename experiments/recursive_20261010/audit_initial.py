"""Independent all-row audit of the initial TRAIN48 architecture diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(root):
    p=read(root/'PROTOCOL.json');result=read(root/'COMPLETE.json');rows=result['rows']
    assert result['status']=='COMPLETE_CHECKED' and result['protocol_sha256']==digest(root/'PROTOCOL.json')
    assert len(rows)==48 and {r['index'] for r in rows}==set(p['indices'])
    assert result['optimizer_steps']==0 and not result['heldout_read'] and not result['gpu_use']
    prior={r['index']:r for r in read(ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json')['rows'] if r['model']=='parent/e0'}
    for row in rows:
        before=prior[row['index']];assert row['count']==before['count'] and row['categories']==before['categories']
        for key in ('nmse','si_sdr','reference_power'):
            for a,b in zip(row['parent'][key],before[key]):assert abs(a-b)<2e-5
        assert row['parent']['reference_power']==row['successive']['reference_power']
        for mode in ('parent','successive'):
            metric=row[mode];assert metric['sum_relative_error']<1e-10
            assert sorted(metric['assignment'])==[0,1,2]
            for nmse,error,power in zip(metric['nmse'],metric['absolute_error_power'],metric['reference_power']):
                assert abs(nmse-error/power)<1e-12
    for summary in result['summary']:
        selected=[r[summary['mode']] for r in rows if r['count']==summary['count']]
        assert len(selected)==summary['cases']
        for key in ('nmse','si_sdr'):
            assert abs(summary[key]-statistics.mean(v for r in selected for v in r[key]))<1e-12
        assert abs(summary['weakest_nmse']-statistics.mean(r['nmse'][r['weakest_index']] for r in selected))<1e-12
        assert abs(summary['inactive_leak']-statistics.mean(r['inactive_leak'] for r in selected))<1e-12
    for rel,sha in p['sources'].items():assert digest(ROOT/rel)==digest(root/'source_snapshot'/rel)==sha
    assert digest(root/'COMPLETE.json')==digest(ROOT/'reports/2026-10-10/SUCCESSIVE_INITIAL48_RESULT.json')
    out=dict(status='PASS',protocol_sha256=digest(root/'PROTOCOL.json'),result_sha256=digest(root/'COMPLETE.json'),
        cases=48,all_conditions_reaggregated=True,parent_reproduction_checked=True,source_snapshots_checked=True,
        waveforms_reinferred=False,heldout_read=False,auditor_sha256=digest(Path(__file__)))
    (root/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    (ROOT/'reports/2026-10-10/SUCCESSIVE_INITIAL48_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run.resolve())
