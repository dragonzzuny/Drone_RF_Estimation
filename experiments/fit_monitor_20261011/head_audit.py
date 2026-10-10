"""Independent CPU check of frozen tensors, optimizer state and decisions."""
from pathlib import Path
import os
import time
import traceback
import numpy as np
import torch
import head_adapt as head

ROOT, RUN, w, fit = head.ROOT, head.OUT, head.w, head.fit
OUT = ROOT / 'local/tfgridnet_head_audit_20261011_v1'


def main():
    OUT.mkdir(exist_ok=False)
    began = time.time()
    while not (RUN / 'COMPLETE.json').exists():
        if (RUN / 'FAILURE.json').exists():
            raise RuntimeError('Head adaptation failed')
        if time.time()-began > 2100:
            raise TimeoutError('Head audit wait limit')
        w.write(OUT/'STATE.json', dict(status='WAITING_COMPLETION',pid=os.getpid(),time=time.time()))
        time.sleep(15)
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    w.write(OUT/'STATE.json', dict(status='CPU_AUDIT',pid=os.getpid(),time=time.time()))
    p = w.read(RUN/'PROTOCOL.json')
    fit.verify(RUN,p)
    result = w.read(RUN/'COMPLETE.json')
    ph = w.digest(RUN/'PROTOCOL.json')
    assert result['protocol_sha256'] == ph
    original = torch.load(p['parent_checkpoint'],map_location='cpu',weights_only=False)
    ck = torch.load(RUN/'LAST.pt',map_location='cpu',weights_only=False)
    assert ck['protocol_sha256'] == ph and ck['updates'] == result['total_updates']
    assert w.digest(RUN/'LAST.pt') == result['final']['checkpoint_sha256']
    net = head.base.build()
    net.load_state_dict(ck['model'])
    assert sum(t.numel() for t in net.parameters()) == p['parameters']
    names = [name for name,_ in net.named_parameters()]
    ids = [i for g in ck['optimizer']['param_groups'] for i in g['params']]
    assert len(names) == len(ids)
    changed = []
    for key,tensor in ck['model'].items():
        assert torch.isfinite(tensor).all()
        if key in p['trainable_names']:
            assert not torch.equal(tensor,original['model'][key])
            changed.append(key)
        else:
            assert torch.equal(tensor,original['model'][key]),key
    for name,index in zip(names,ids):
        values = ck['optimizer']['state'][index]
        assert int(values['step']) == (result['total_updates'] if name in p['trainable_names'] else 64)
        for key,value in values.items():
            if isinstance(value,torch.Tensor):
                assert torch.isfinite(value).all()
                if name not in p['trainable_names']:
                    assert torch.equal(value,original['optimizer']['state'][index][key])
    history = result['history']
    assert history == w.read(RUN/'HISTORY.json')['history']
    maximum_sum_error = 0.
    for point in history:
        assert point == w.read(RUN/f"ADDED_{point['added_updates']:03d}.json")
        assert [r['index'] for r in point['rows']] == p['probe_indices']
        for row in point['rows']:
            assert len(row['nmse']) == len(row['si_sdr']) == row['count']
            assert np.isfinite(row['nmse']+row['si_sdr']).all()
            assert sorted(row['assignment']) == [0,1,2]
            assert row['weakest_index'] == int(np.argmin(row['reference_power']))
            maximum_sum_error = max(maximum_sum_error,row['sum_relative_error'])
            assert row['sum_relative_error'] < 1e-9
        for group in point['by_count']:
            rows = [r for r in point['rows'] if r['count'] == group['count']]
            expected = dict(mean_nmse=np.mean([v for r in rows for v in r['nmse']]),
                median_nmse=np.median([v for r in rows for v in r['nmse']]),
                mean_si_sdr=np.mean([v for r in rows for v in r['si_sdr']]),
                weakest_nmse=np.mean([r['nmse'][r['weakest_index']] for r in rows]),
                max_source_nmse=max(v for r in rows for v in r['nmse']))
            for key,value in expected.items():
                assert abs(value-group[key]) < 1e-12
    selected = history[0]
    for point in history[1:]:
        checks = [dict(count=a['count'],nmse=b['mean_nmse']<a['mean_nmse'],
                       si=b['mean_si_sdr']>a['mean_si_sdr'],weak=b['weakest_nmse']<=a['weakest_nmse'],
                       median=b['median_nmse']<=a['median_nmse'])
                  for a,b in zip(selected['by_count'],point['by_count'])]
        passed = all(c['nmse'] and c['si'] and c['weak'] and c['median'] for c in checks)
        assert checks == point['checks'] and passed == point['continuation_pass']
        if passed:
            selected = point
        else:
            assert point == history[-1] and result['reason'] == 'NO_JOINT_IMPROVEMENT_REVIEW'
    assert result['selected'] == selected
    selected_sha = w.digest(RUN/'SELECTED.pt')
    expected_sha = selected.get('checkpoint_sha256',p['parent_checkpoint_sha256'])
    assert selected_sha == expected_sha == result['selected_checkpoint_sha256']
    fit.verify(RUN,p)
    receipt = dict(status='PASS',added_updates=result['added_updates'],total_updates=result['total_updates'],
        changed_tensors=changed,all_other_tensors_bitwise_unchanged=True,
        frozen_optimizer_state_bitwise_unchanged=True,summaries_recomputed=True,decisions_recomputed=True,
        selected_added_updates=selected['added_updates'],selected_checkpoint_sha256=selected_sha,
        maximum_sum_relative_error=maximum_sum_error,protocol_sha256=ph,
        audit_source_sha256=w.digest(Path(__file__)),independent_full_length_cpu_inference=False,
        optimizer_updates=0,validation_read=False,heldout_read=False,time=time.time())
    w.write(OUT/'COMPLETE.json',receipt)
    w.write(ROOT/'reports/2026-10-11/TFGRIDNET_HEAD_ADAPT_AUDIT.json',receipt)
    w.write(OUT/'STATE.json',dict(status='COMPLETED',pid=None,time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
