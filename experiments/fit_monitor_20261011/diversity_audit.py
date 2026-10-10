"""CPU receipt/weight audit; does not repeat full-length waveform inference."""
from pathlib import Path
import os
import sys
import time
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tfgridnet_rf_20261011'))
import fit
from model import build

w = fit.w
RUN = ROOT / 'local/tfgridnet_diversity_20261011_v1'
OUT = ROOT / 'local/tfgridnet_diversity_audit_20261011_v1'


def state(status, **fields):
    w.write(OUT / 'STATE.json', dict(status=status, pid=os.getpid(), time=time.time(), **fields))


def main():
    OUT.mkdir(exist_ok=False)
    began = time.time()
    while not (RUN / 'COMPLETE.json').exists():
        if (RUN / 'FAILURE.json').exists():
            raise RuntimeError('Training failed; preserve its failure record')
        if time.time() - began > 3600:
            raise TimeoutError('No completed diversity run within audit wait budget')
        state('WAITING_COMPLETION')
        time.sleep(15)
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    state('CPU_AUDIT')
    p = w.read(RUN / 'PROTOCOL.json')
    fit.verify(RUN, p)
    ph = w.digest(RUN / 'PROTOCOL.json')
    result = w.read(RUN / 'COMPLETE.json')
    assert result['protocol_sha256'] == ph
    assert result['total_updates'] == 64 + result['added_updates']
    assert 0 < result['added_updates'] <= 32
    assert len(set(p['train_indices'])) == 128
    assert len(set(p['probe_indices'])) == 30
    assert not set(p['train_indices']) & set(p['probe_indices'])
    assert [i for batch in p['batches'] for i in batch] == p['train_indices']
    ck = torch.load(RUN / 'LAST.pt', map_location='cpu', weights_only=False)
    assert ck['updates'] == result['total_updates'] and ck['protocol_sha256'] == ph
    assert result['final']['checkpoint_sha256'] == w.digest(RUN / 'LAST.pt')
    net = build()
    net.load_state_dict(ck['model'])
    assert sum(t.numel() for t in net.parameters()) == p['parameters']
    assert all(torch.isfinite(t).all() for t in net.parameters())
    assert {int(s['step']) for s in ck['optimizer']['state'].values()} == {result['total_updates']}
    for values in ck['optimizer']['state'].values():
        for value in values.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
    initial = torch.load(p['parent_checkpoint'], map_location='cpu', weights_only=False)['model']
    changes = {}
    for block in range(6):
        keys = [key for key in initial if key.startswith(f'blocks.{block}.')]
        changes[block] = sum(float((ck['model'][key].double() - initial[key].double()).square().sum())
                             for key in keys) ** .5
        assert changes[block] > 0
    history = result['history']
    assert history == w.read(RUN / 'HISTORY.json')['history']
    assert history[0]['added_updates'] == 0 and history[-1]['added_updates'] == result['added_updates']
    if result['added_updates'] == 32:
        assert [h['added_updates'] for h in history] == [0, 16, 32]
    maxsum = 0.
    for point in history:
        assert point == w.read(RUN / f"ADDED_{point['added_updates']:03d}.json")
        assert point['total_updates'] == 64 + point['added_updates']
        assert [r['index'] for r in point['rows']] == p['probe_indices']
        for row in point['rows']:
            assert row['count'] in (2, 3)
            assert len(row['nmse']) == len(row['si_sdr']) == len(row['reference_power']) == row['count']
            assert np.isfinite(row['nmse'] + row['si_sdr'] + row['reference_power']).all()
            assert all(v > 0 for v in row['reference_power'])
            assert row['weakest_index'] == int(np.argmin(row['reference_power']))
            assert sorted(row['assignment']) == [0, 1, 2]
            assert 1 <= row['predicted_count'] <= 3
            maxsum = max(maxsum, row['sum_relative_error'])
            assert row['sum_relative_error'] < 1e-9
        assert [g['count'] for g in point['by_count']] == [2, 3]
        for group in point['by_count']:
            rows = [r for r in point['rows'] if r['count'] == group['count']]
            assert len(rows) == group['cases']
            expected = dict(mean_nmse=np.mean([v for r in rows for v in r['nmse']]),
                            mean_si_sdr=np.mean([v for r in rows for v in r['si_sdr']]),
                            weakest_nmse=np.mean([r['nmse'][r['weakest_index']] for r in rows]),
                            max_source_nmse=max(v for r in rows for v in r['nmse']))
            for key, value in expected.items():
                assert abs(group[key] - value) < 1e-12
    checks = [dict(count=a['count'], nmse=b['mean_nmse'] < a['mean_nmse'],
                   si=b['mean_si_sdr'] > a['mean_si_sdr'], weak=b['weakest_nmse'] <= a['weakest_nmse'])
              for a, b in zip(history[0]['by_count'], history[-1]['by_count'])]
    candidate = result['added_updates'] == 32 and all(c['nmse'] and c['si'] and c['weak'] for c in checks)
    assert checks == result['checks'] and candidate == result['probe_improvement_candidate']
    assert not result['incumbent_replaced'] and not result['heldout_read'] and not result['validation_read']
    fit.verify(RUN, p)
    receipt = dict(status='PASS', added_updates=result['added_updates'], total_updates=result['total_updates'],
                   parameters=p['parameters'], per_block_parameter_change_since_update64_l2=changes,
                   summaries_recomputed=True, decision_recomputed=True, indices_disjoint=True,
                   probe_improvement_candidate=candidate, maximum_sum_relative_error=maxsum,
                   checkpoint_sha256=w.digest(RUN / 'LAST.pt'), protocol_sha256=ph,
                   audit_source_sha256=w.digest(Path(__file__)), independent_full_length_cpu_inference=False,
                   optimizer_updates=0, validation_read=False, heldout_read=False, time=time.time())
    w.write(OUT / 'COMPLETE.json', receipt)
    w.write(ROOT / 'reports/2026-10-11/TFGRIDNET_DIVERSITY_AUDIT.json', receipt)
    state('COMPLETED', probe_improvement_candidate=candidate)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        w.write(OUT / 'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        state('FAILED')
        raise
