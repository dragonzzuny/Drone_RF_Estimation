"""CPU input-dependence check at fixed RMS, power context, and fine I/Q.

Counterfactual sign flips more than 1 ms outside the decoded interval leave
all magnitudes unchanged. This isolates access to distant complex samples;
it is NOT a quality or physically realistic augmentation experiment.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import phase_packing as pp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/rfuav_native_frequency_20261009'))
from native_data import NativeMixtures, sha256, write_json
from drone_rf.context_data import component_gains


def run(preparation, output):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    data = NativeMixtures(preparation, 'train_pack', 1)
    item, row = data[4], data.rows[4]
    k = int(row['count'])
    indices = row['indices'][:k]
    gains = component_gains([data.library.clips[int(i)]['mean_power'] for i in indices],
                            row['levels'][:k], row['phases'][:k])
    mix = np.zeros(pp.SAMPLES, np.complex64)
    for index, gain in zip(indices, gains):
        mix += (data.library._array(int(index)) * gain).astype(np.complex64)
    start = int(item['crop_start'])
    np.testing.assert_array_equal(mix[start:start + pp.FINE], item['mixture'])
    x = torch.from_numpy(mix)[None]
    positions = torch.arange(pp.SAMPLES)
    distant = (positions < start - 100_000) | (positions >= start + pp.FINE + 100_000)
    changed = x.clone()
    changed[:, distant] = -changed[:, distant]
    assert torch.count_nonzero(x[:, distant]) > 0
    assert torch.equal(x.abs().square(), changed.abs().square())
    assert torch.equal(x.abs().square().mean(-1), changed.abs().square().mean(-1))
    assert torch.equal(x[:, start:start + pp.FINE], changed[:, start:start + pp.FINE])
    features = torch.from_numpy(item['context_features'])[None]
    crop = torch.tensor([start])
    local, long = pp.PhasePackedWaveNet('local'), pp.PhasePackedWaveNet('long')
    for key, value in local.state_dict().items():
        assert torch.equal(value, long.state_dict()[key])
    local.eval(); long.eval()
    began = time.time()
    with torch.no_grad():
        a = local(x, features, crop)
        b = local(changed, features, crop)
        c = long(x, features, crop)
        d = long(changed, features, crop)
    assert torch.equal(a['estimates'], b['estimates'])
    for out in (a, b, c, d):
        assert torch.isfinite(out['estimates']).all()
        assert torch.equal(out['count_logits'], a['count_logits'])
    absolute_change = (c['estimates'] - d['estimates']).abs().square().mean().item()
    assert absolute_change > 0
    result = dict(status='PASS', device='cpu', train_index=4, trained=False,
        quality_evaluated=False, validation_iq_read=False, heldout_read=False,
        full_capacity_parameters=pp.PARAMETERS, input_samples=pp.SAMPLES,
        crop_start=start, fine_samples=pp.FINE, unchanged_guard_samples=100_000,
        changed_sample_count=int(distant.sum()), unchanged_fine_iq=True,
        unchanged_sample_magnitudes=True, unchanged_rms_exact=True,
        unchanged_context_features=True, unchanged_count_logits_exact=True,
        local_output_unchanged_exact=True,
        long_output_change_mean_squared=absolute_change,
        long_output_change_relative_to_mixture=absolute_change / x.abs().square().mean().item(),
        interpretation='Distant I/Q can affect the decoded interval at fixed RMS and auxiliary context; no trained quality result',
        perturbation='Synthetic sign flips >1ms outside decoded interval; numerical dependency check only',
        seconds=time.time()-began, source_sha256={p.name: sha256(p) for p in
            (Path(__file__), Path(pp.__file__))},
        preparation_sha256=sha256(preparation/'PREPARATION.json'))
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.preparation.resolve(), args.output)
