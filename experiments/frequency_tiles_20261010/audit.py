"""Audit all saved tiled-inference rows and prediction-only permutations."""
import argparse
import hashlib
import itertools
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
    assert r==read(ROOT/'reports/2026-10-10/FREQUENCY_TILES_RESULT.json')
    assert r['optimizer_steps']==0 and not r['gpu_use'] and not r['heldout_read'] and not r['dev_read']
    assert len(r['rows'])==6 and len(r['summary'])==9 and read(root/'ROWS.json')==r['rows']
    for rel,sha in p['sources'].items():assert digest(ROOT/rel)==digest(root/'source_snapshot'/rel)==sha
    assert digest(ROOT/'reports/2026-10-10/FREQUENCY_TILES_ALGEBRA_CHECK.json')==p['check_sha256']
    old=read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    assert digest(old['parent_checkpoint'])==p['parent_sha256']
    assert digest(Path(old['preparation'])/'PREPARATION.json')==p['preparation_sha256']
    prior={x['index']:x for x in read(ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json')['rows'] if x['model']=='parent/e0'}
    assert [row['index'] for row in r['rows']]==p['indices']
    permutations=list(itertools.permutations(range(3)))
    for row in r['rows']:
        assert set(row['modes'])==set(p['modes']) and len(row['tile_alignment'])==8
        for number,tile in enumerate(row['tile_alignment']):
            assert tile['indices']==[(number*64+i)%512 for i in range(128)]
            costs,=tile['costs'];order,=tile['order'];assert len(costs)==6
            assert all(math.isfinite(c) and c>=0 for c in costs)
            chosen=min(range(6),key=lambda i:costs[i]);assert order==list(permutations[chosen])+[3]
        for mode,m in row['modes'].items():
            assert len(m['nmse'])==len(m['si_sdr'])==row['count']
            assert m['sum_relative_error']<1e-10
            assert all(math.isfinite(x) for x in m['nmse']+m['si_sdr'])
            assert m['reference_power']==row['modes']['parent']['reference_power']
        for key in ('nmse','si_sdr','reference_power'):
            assert max(abs(x-y) for x,y in zip(row['modes']['parent'][key],prior[row['index']][key]))<2e-5
    for s in r['summary']:
        rows=[row['modes'][s['mode']] for row in r['rows'] if row['count']==s['count']]
        assert len(rows)==s['cases']==2
        assert s['nmse']==statistics.mean(v for row in rows for v in row['nmse'])
        assert s['si_sdr']==statistics.mean(v for row in rows for v in row['si_sdr'])
        assert s['weakest_nmse']==statistics.mean(row['nmse'][row['weakest_index']] for row in rows)
    report=dict(status='PASS',protocol_sha256=digest(root/'PROTOCOL.json'),complete_sha256=digest(root/'COMPLETE.json'),
        auditor_sha256=digest(Path(__file__)),rows_checked=6,predicted_alignment_choices_checked=48,
        all_sources_and_parent_preparation_hashes_checked=True,all_groups_reaggregated=True,
        heldout_read=False,recorded_iq_reads=0,independently_reran_inference=False,
        prior_run_failed_before_result='v1 path type error; v2 repeats same six cases after fix')
    (ROOT/'reports/2026-10-10/FREQUENCY_TILES_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args().run.resolve())
