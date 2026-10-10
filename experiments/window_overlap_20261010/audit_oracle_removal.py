"""Independent saved-row aggregation audit; no waveform inference."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def audit(run):
    result=read(run/'COMPLETE.json');protocol=read(run/'PROTOCOL.json')
    assert result['status']=='COMPLETE_CHECKED'
    assert result['protocol_sha256']==digest(run/'PROTOCOL.json')
    assert result['oracle'] and result['updates']==0 and not result['heldout_read']
    original={r['index']:r for r in result['original']}
    assert len(original)==14 and set(original)==set(protocol['indices'])
    rows=result['rows'];assert len(rows)==42
    assert len({(r['index'],r['removed_slot']) for r in rows})==42
    for row in rows:
        before=original[row['index']];removed=row['removed_slot'];kept=row['kept_slots']
        assert kept==[i for i in range(3) if i!=removed]
        rank=sorted(range(3),key=lambda i:-before['reference_power'][i]).index(removed)+1
        assert rank==row['removed_power_rank']
        assert row['count']==2 and len(row['nmse'])==len(row['si_sdr'])==2
        for field in ('nmse','si_sdr'):
            assert row['parent_kept_'+field]==[before[field][j] for j in kept]
        for i,j in enumerate(kept):
            assert abs(row['reference_power'][i]-before['reference_power'][j])<1e-12
            assert abs(row['nmse'][i]-row['absolute_error_power'][i]/row['reference_power'][i])<1e-12
        assert row['subtraction_roundoff_relative']<1e-12
        assert row['sum_relative_error']<1e-10
    for saved in result['summary']:
        selected=[r for r in rows if r['removed_power_rank']==saved['removed_power_rank']]
        assert len(selected)==saved['cases']==14 and saved['source_contributions']==28
        for field,key in [('parent_kept_nmse','parent_kept_nmse'),('oracle_remaining_nmse','nmse'),
                          ('parent_kept_si_sdr','parent_kept_si_sdr'),('oracle_remaining_si_sdr','si_sdr')]:
            assert abs(saved[field]-statistics.mean(v for r in selected for v in r[key]))<1e-12
        assert saved['both_improved_sources']==sum(a<b and c>d for r in selected for a,b,c,d in zip(r['nmse'],r['parent_kept_nmse'],r['si_sdr'],r['parent_kept_si_sdr']))
    for rel,sha in protocol['source_sha256'].items():
        assert digest(ROOT/rel)==digest(run/'source_snapshot'/rel)==sha
    assert digest(ROOT/'reports/2026-10-10/ORACLE_REMOVAL_RESULT.json')==digest(run/'COMPLETE.json')
    out=dict(status='PASS',result_sha256=digest(run/'COMPLETE.json'),protocol_sha256=digest(run/'PROTOCOL.json'),
             original_cases=14,removal_rows=42,all_ranks_reaggregated=True,all_remaining_powers_preserved=True,
             source_snapshots_unchanged=True,waveforms_reinferred=False,heldout_read=False,
             auditor_sha256=digest(Path(__file__)))
    (run/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    (ROOT/'reports/2026-10-10/ORACLE_REMOVAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run.resolve())
