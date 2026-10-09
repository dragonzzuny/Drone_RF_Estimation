"""CPU-only, fixed TRAIN6 overlapping-window diagnosis; no model updates."""
import argparse
import importlib.util
import itertools
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/count_pcgrad_20261010'))
spec = importlib.util.spec_from_file_location('overlap_parent', ROOT / 'experiments/count_pcgrad_20261010/train.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from drone_rf.waveform import waveform_metrics, complex_si_sdr
w = base.w
OFFSETS = (0, -16384, 16384)
FIELDS = ('mixture', 'references', 'active', 'context_features', 'crop_start', 'construction_count')


def batch(raw):
    return {k: torch.as_tensor(np.asarray(raw[k])[None]) for k in FIELDS}


def align(pred, anchor):
    permutations = list(itertools.permutations(range(3)))
    costs = [float((pred[:, list(p)] - anchor[:, :3]).abs().square().sum()) for p in permutations]
    order = list(permutations[int(np.argmin(costs))]) + [3]
    return pred[:, order], order, costs


def metrics(pred, item, assignment=None):
    m = waveform_metrics(pred, item['references'], item['active'], item['mixture'])
    if assignment is None:
        assignment = m['assignment']
    estimates = pred[:, :3].gather(1, assignment[..., None].expand(-1, -1, pred.shape[-1])).to(torch.complex128)
    refs = item['references'].to(torch.complex128)
    n = int(item['construction_count'])
    power = refs.abs().square().mean(-1)
    err = (estimates - refs).abs().square().mean(-1)
    nmse = err / torch.where(power > 0, power, 1.)
    si = complex_si_sdr(estimates, refs)
    row = dict(count=n, nmse=nmse[0, :n].tolist(), si_sdr=si[0, :n].tolist(),
        reference_power=power[0, :n].tolist(), absolute_error_power=err[0, :n].tolist(),
        weakest_index=int(power[0, :n].argmin()), assignment=assignment[0].tolist(),
        independent_pit_assignment=m['assignment'][0].tolist(),
        sum_relative_error=float(m['sum_relative_error'][0]))
    assert np.isfinite(row['nmse'] + row['si_sdr']).all()
    assert row['sum_relative_error'] < 1e-10
    return row, assignment


def check():
    torch.manual_seed(0)
    reference = torch.randn(1, 4, 257, dtype=torch.complex64)
    permuted = reference[:, [2, 0, 1, 3]]
    aligned, order, _ = align(permuted, reference)
    assert torch.equal(aligned, reference) and order == [1, 2, 0, 3]
    length = 63872
    positions = torch.arange(16384, length - 16384)
    weights = torch.stack([torch.sin(torch.pi * (positions - off + .5) / length).square() for off in OFFSETS])
    weights /= weights.sum(0)
    assert float((weights.sum(0) - 1).abs().max()) < 2e-7
    assert bool((weights > 0).all())
    return dict(status='PASS', prediction_only_permutation_checked=True,
        common_samples=len(positions), positive_partition_of_unity_checked=True)


def run(root, public):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Already registered')
    check_result = check()
    prior_path = ROOT / 'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json'
    prior = w.read(prior_path)
    reference = [r for r in prior['rows'] if r['model'] == 'parent/e0']
    indices = [r['index'] for n in (1, 2, 3) for r in [q for q in reference if q['count'] == n][:2]]
    old = w.read(ROOT / 'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    sources = dict(old['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))] = w.digest(Path(__file__))
    plan_path = ROOT / 'reports/2026-10-10/WINDOW_OVERLAP_PLAN_KO.md'
    p = dict(status='REGISTERED_CPU_WINDOW_OVERLAP', indices=indices, offsets=list(OFFSETS),
        source_sha256=sources, parent=old['parent_checkpoint'], parent_sha256=old['parent_checkpoint_sha256'],
        preparation=old['preparation'], preparation_sha256=old['preparation_sha256'],
        plan_sha256=w.digest(plan_path), prior_sha256=w.digest(prior_path),
        assignment='Align predicted source slots to central-window predictions on common samples; background fixed. References used only for scoring.',
        selection='Same fixed first two TRAIN48 examples per count; offsets and all modes fixed before inference.',
        modes=['central', 'earlier_window', 'later_window', 'uniform_average', 'center_weighted_average'],
        updates=0, inferences=18, heldout_read=False, gpu_use=False, registered_at=time.time(), check=check_result)
    for rel, sha in sources.items():
        assert w.digest(ROOT / rel) == sha
        target = root / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    w.write(root / 'PROTOCOL.json', p)
    def verify():
        for rel, sha in sources.items():
            assert w.digest(ROOT / rel) == w.digest(root / 'source_snapshot' / rel) == sha
        assert w.digest(Path(p['parent'])) == p['parent_sha256']
        assert w.digest(Path(p['preparation']) / 'PREPARATION.json') == p['preparation_sha256']
        assert w.digest(plan_path) == p['plan_sha256']
    verify()
    torch.set_num_threads(2)
    started = time.time()
    data = base.worker.NativeMixtures(p['preparation'], 'train_pack', 1)
    net = base.worker.make_model('retained_unet', Path(p['parent'])).eval()
    assert sum(v.numel() for v in net.parameters()) == 32142859
    rows, diagnostics = [], []
    with torch.inference_mode():
        for case, index in enumerate(indices):
            original_start = int(data.rows[index]['crop_start'])
            length = data.length
            lo, hi = 16384, length - 16384
            original = data[index]
            common = batch(original)
            for key in ('mixture', 'references'):
                common[key] = common[key][..., lo:hi]
            predictions, count_logits = [], []
            diag = dict(index=index, crop_start=original_start, offsets=list(OFFSETS),
                common_samples=hi-lo, reference_and_mixture_overlap_bitwise_equal=True, windows=[])
            for position, offset in enumerate(OFFSETS):
                new_start = original_start + offset
                assert new_start >= 0
                assert all(new_start + length <= data.library.clips[int(i)]['samples'] for i in data.rows[index]['indices'][:int(data.rows[index]['count'])])
                data.rows[index]['crop_start'] = new_start
                try:
                    raw = data[index]
                finally:
                    data.rows[index]['crop_start'] = original_start
                assert np.array_equal(raw['context_features'], original['context_features'])
                item = batch(raw)
                left, right = lo-offset, hi-offset
                for key in ('mixture', 'references'):
                    assert torch.equal(item[key][..., left:right], common[key])
                full, logits = base.worker.predict(net, item)
                if offset == 0:
                    full_metric, _ = metrics(full, item)
                    expected = next(r for r in reference if r['index'] == index)
                    assert max(abs(a-b) for a, b in zip(full_metric['nmse'], expected['nmse'])) < 2e-5
                    anchor = full[..., lo:hi].clone()
                    _, fixed_assignment = metrics(anchor, common)
                prediction, order, costs = align(full[..., left:right], anchor)
                predictions.append(prediction)
                count_logits.append(logits)
                diag['windows'].append(dict(offset=offset, prediction_only_order=order,
                    matching_costs=costs, drift_relative_energy=float((prediction-anchor).abs().square().sum()/common['mixture'].abs().square().sum()),
                    predicted_count=int(logits.argmax(-1))+1))
                w.write(root / 'STATE.json', dict(status='CPU_OVERLAP_DIAGNOSIS', inferences=case*3+position+1,
                    total=18, pid=os.getpid(), time=time.time()))
            positions = torch.arange(lo, hi)
            weights = torch.stack([torch.sin(torch.pi * (positions-off+.5)/length).square() for off in OFFSETS])
            weights /= weights.sum(0)
            stack = torch.stack(predictions)
            estimates = predictions + [stack.mean(0), (stack * weights[:, None, None]).sum(0)]
            for mode, estimate in zip(p['modes'], estimates):
                row, _ = metrics(estimate, common, fixed_assignment)
                row.update(index=index, mode=mode)
                rows.append(row)
            diag['count_logits_bitwise_equal'] = all(torch.equal(count_logits[0], x) for x in count_logits[1:])
            diagnostics.append(diag)
            w.write(root / 'PARTIAL.json', dict(rows=rows, diagnostics=diagnostics))
    verify()
    summaries=[]
    for mode in p['modes']:
        for n in (1,2,3):
            selected=[r for r in rows if r['mode']==mode and r['count']==n]
            summaries.append(dict(mode=mode,count=n,cases=len(selected),
                nmse=float(np.mean([np.mean(r['nmse']) for r in selected])),
                si_sdr=float(np.mean([np.mean(r['si_sdr']) for r in selected])),
                weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in selected]))))
    result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),
        rows=rows,diagnostics=diagnostics,by_count=summaries,updates=0,inferences=18,
        seconds=time.time()-started,heldout_read=False,gpu_use=False,
        limitation='Six reused TRAIN cases, common middle 311.04 microseconds only; window shift also changes local normalization and position-conditioned context. No isolated padding cause or DEV benefit claimed.')
    w.write(root/'COMPLETE.json',result)
    w.write(public,result)
    w.write(root/'STATE.json',dict(status='COMPLETE',inferences=18,pid=os.getpid(),time=time.time()))
    print(dict(status='COMPLETE',seconds=result['seconds'],by_count=summaries),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True)
    a=parser.parse_args()
    try:
        run(a.run.resolve(),a.public.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
