"""Reference-assisted diagnostics on the already fitted four TRAIN mixtures.

No model update, validation/held-out I/Q, or deployable oracle correction.
Keep cross terms: recordings are not assumed orthogonal.
"""
import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_mix_fit import (HERE, LOCK, PREPARATION, admitted_dataset, batch,
                           make_model, predict, sha256, waveform_metrics, write_json)


def energy(x):
    return float(np.vdot(x, x).real)


def decompose(estimate, references, target):
    """Projection on full-window reference span; not a causal error attribution."""
    matrix = references.T
    coefficient, _, rank, singular = np.linalg.lstsq(matrix, estimate, rcond=None)
    source = references[target]
    denominator = energy(source)
    remainder = estimate - matrix @ coefficient
    gain_error = (coefficient[target] - 1) * source
    leakage = matrix @ coefficient - coefficient[target] * source
    parts = [energy(gain_error), energy(leakage), energy(remainder)]
    cross = 2 * float(np.vdot(gain_error, leakage).real)
    total = energy(estimate - source) / denominator
    closure = (sum(parts) + cross) / denominator
    if abs(total - closure) > 1e-9:
        raise RuntimeError('Error decomposition failed closure')
    estimate_energy = energy(estimate)
    scalar = np.vdot(estimate, source) / estimate_energy if estimate_energy else 0j
    positive_gain = max(0., float(np.real(scalar)))
    return dict(nmse=total, gain_error_nmse=parts[0]/denominator,
        other_reference_span_nmse=parts[1]/denominator,
        outside_reference_span_nmse=parts[2]/denominator,
        cross_term_nmse=cross/denominator, closure_error=total-closure,
        coefficients_real=coefficient.real.tolist(), coefficients_imag=coefficient.imag.tolist(),
        target_coefficient_abs=float(abs(coefficient[target])),
        target_coefficient_phase_deg=float(np.angle(coefficient[target], deg=True)),
        oracle_positive_gain=positive_gain,
        oracle_positive_gain_nmse=energy(positive_gain*estimate-source)/denominator,
        oracle_complex_gain_abs=float(abs(scalar)),
        oracle_complex_gain_nmse=energy(scalar*estimate-source)/denominator,
        reference_matrix_rank=int(rank), reference_condition_number=float(singular[0]/singular[-1]))


def chunk_assignment_diagnostic(aligned, refs, chunk_size):
    """Active outputs only; normalization fixed to full-window reference energy."""
    count, length = refs.shape
    permutations = list(itertools.permutations(range(count)))
    denominator = np.sum(np.abs(refs)**2, axis=-1)
    fixed, oracle, changed = 0., 0., 0
    for start in range(0, length, chunk_size):
        stop = min(length, start + chunk_size)
        costs = [float(np.mean(np.sum(np.abs(aligned[list(p), start:stop]-refs[:, start:stop])**2,
                                     axis=-1)/denominator)) for p in permutations]
        fixed += costs[0]
        oracle += min(costs)
        changed += int(np.argmin(costs) != 0)
    return dict(chunk_samples=chunk_size, chunks=(length+chunk_size-1)//chunk_size,
        changed_chunks=changed, fixed_nmse=fixed, oracle_chunk_assignment_nmse=oracle,
        oracle_relative_reduction=(fixed-oracle)/fixed if fixed else 0.)


@torch.no_grad()
def run(fit, output):
    if output.exists():
        raise RuntimeError('Refuse to overwrite audit')
    protocol = json.loads((fit/'PROTOCOL.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        if sha256(Path(name)) != digest:
            raise RuntimeError('Original fit source changed')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    data = admitted_dataset(PREPARATION, 'train_pack', 1)
    items = [batch(data[record['schedule_index']], 'cuda') for record in protocol['examples']]
    config = json.loads((PREPARATION/'PREPARATION.json').read_text())
    result = dict(scope='four already fitted training mixtures; no generalization claim',
        no_validation_or_heldout_iq=True, no_optimizer_or_weight_updates=True,
        oracle_warning='Reference regressions and gain/chunk corrections are diagnostics only.',
        decomposition_warning='Outside-span error includes time-varying distortions/leakage; not a noise-floor estimate.',
        audit_source_sha256=sha256(Path(__file__)),
        fit_checkpoint_sha256=sha256(fit/'BEST.pt'), rows=[])
    net = make_model('unet_mean', config).cuda().eval()
    for name in ('initial_e22', 'fit_300'):
        if name == 'fit_300':
            saved = torch.load(fit/'BEST.pt', map_location='cpu', weights_only=False)
            assert saved['step'] == 300
            net.load_state_dict(saved['model'], strict=True)
        for i, (item, record) in enumerate(zip(items, protocol['examples'])):
            count = record['count']
            estimates, _ = predict(net, item)
            metrics = waveform_metrics(estimates, item['references'], item['active'], item['mixture'])
            assignment = metrics['assignment'][0].cpu().numpy()
            pred = estimates[0].cpu().numpy().astype(np.complex128)
            refs = item['references'][0, :count].cpu().numpy().astype(np.complex128)
            aligned = pred[assignment[:count]]
            components = [decompose(aligned[j], refs, j) for j in range(count)]
            expected = json.loads((fit/f'STEP_{0 if name == "initial_e22" else 300:03d}.json').read_text())
            np.testing.assert_allclose([c['nmse'] for c in components], expected['rows'][i]['nmse'],
                                       rtol=1e-5, atol=1e-7)
            result['rows'].append(dict(checkpoint=name, example=i, count=count,
                categories=record['categories'], assignment=assignment.tolist(), components=components,
                background_relative_power=float(metrics['background_nmse'][0]),
                inactive_relative_power=float(metrics['inactive_leak'][0].sum()),
                chunk_assignment=[chunk_assignment_diagnostic(aligned, refs, n) for n in (2048,8192,16384)]))
    result['summary'] = []
    for name in ('initial_e22', 'fit_300'):
        for count in (2,3):
            rows = [r for r in result['rows'] if r['checkpoint']==name and r['count']==count]
            components = [c for r in rows for c in r['components']]
            fields = ('nmse','gain_error_nmse','other_reference_span_nmse','outside_reference_span_nmse',
                      'cross_term_nmse','oracle_positive_gain_nmse','oracle_complex_gain_nmse')
            result['summary'].append(dict(checkpoint=name,count=count,
                **{key:float(np.mean([c[key] for c in components])) for key in fields},
                max_chunk_oracle_relative_reduction=max(v['oracle_relative_reduction'] for r in rows for v in r['chunk_assignment'])))
    write_json(output, result)
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:]))
    os.nice(10)
    with LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.fit, args.output)
