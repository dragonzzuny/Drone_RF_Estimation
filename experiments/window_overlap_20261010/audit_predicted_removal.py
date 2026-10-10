"""Reaggregate every saved first-removal diagnostic condition independently."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(run):
    p=read(run/'PROTOCOL.json');result=read(run/'COMPLETE.json');rows=result['rows']
    assert result['status']=='COMPLETE_CHECKED' and result['protocol_sha256']==digest(run/'PROTOCOL.json')
    assert len(rows)==14 and {r['index'] for r in rows}==set(p['indices'])
    oracle=read(ROOT/'local/oracle_removal_20261010_v1/COMPLETE.json')
    for row in rows:
        index=row['index'];removed=row['matched_reference_slot'];kept=row['kept_slots']
        original=next(r for r in oracle['original'] if r['index']==index)
        reference=next(r for r in oracle['rows'] if r['index']==index and r['removed_slot']==removed)
        assert row['selected_prediction_slot']==original['assignment'][removed]
        assert kept==[i for i in range(3) if i!=removed]
        assert row['selected_reference_power_rank']==reference['removed_power_rank']
        for key in ('nmse','si_sdr'):
            for a,b in zip(row['original_remaining_'+key],[original[key][i] for i in kept]):assert abs(a-b)<2e-5
            assert row['oracle_recomputed_context_'+key]==reference[key]
            assert abs(row['first_estimate_'+key]-original[key][removed])<2e-5
        for measured in row['conditions'].values():
            assert measured['count']==2 and len(measured['nmse'])==2 and measured['sum_relative_error']<1e-10
            for i,j in enumerate(kept):
                assert abs(measured['reference_power'][i]-original['reference_power'][j])<1e-12
                assert abs(measured['nmse'][i]-measured['absolute_error_power'][i]/measured['reference_power'][i])<1e-12
        assert row['residual_error_identity_roundoff']<1e-12
        expected=original['absolute_error_power'][removed]/sum(original['reference_power'][j] for j in kept)
        assert abs(row['first_error_over_remaining_energy']-expected)<2e-5
    for summary in result['summary']:
        mode=summary['mode'];values=[];nmse=[];si=[];both=0
        for row in rows:
            n=row['conditions'][mode]['nmse'] if mode in row['conditions'] else row[mode+'_nmse']
            s=row['conditions'][mode]['si_sdr'] if mode in row['conditions'] else row[mode+'_si_sdr']
            nmse.extend(n);si.extend(s)
            both+=sum(a<b and c>d for a,b,c,d in zip(n,row['original_remaining_nmse'],s,row['original_remaining_si_sdr']))
        assert len(nmse)==summary['source_contributions']==28
        assert abs(statistics.mean(nmse)-summary['nmse'])<1e-12
        assert abs(statistics.mean(si)-summary['si_sdr'])<1e-12
        assert both==summary['both_improved_sources_vs_original']
    for rel,sha in p['sources'].items():assert digest(ROOT/rel)==digest(run/'source_snapshot'/rel)==sha
    assert digest(run/'COMPLETE.json')==digest(ROOT/'reports/2026-10-10/PREDICTED_REMOVAL_RESULT.json')
    out=dict(status='PASS',protocol_sha256=digest(run/'PROTOCOL.json'),result_sha256=digest(run/'COMPLETE.json'),
        cases=14,all_conditions_reaggregated=True,oracle_reference_identity_checked=True,all_sources_unchanged=True,
        waveforms_reinferred=False,highest_predicted_power_recomputed=False,heldout_read=False,auditor_sha256=digest(Path(__file__)))
    (run/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    (ROOT/'reports/2026-10-10/PREDICTED_REMOVAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run.resolve())
