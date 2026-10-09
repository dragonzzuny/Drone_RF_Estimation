"""TRAIN-only representation audit, without fitting or changing the live study.

STFT bin ordering changes which frequencies are adjacent for a zero-padded 2D
convolution. A reversible FFT shift changes no signal information by itself.
The descriptive energy checks below do not establish an improvement in a model.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/rfuav_native_frequency_20261009'))
from native_data import NativeMixtures, sha256, write_json
from drone_rf.waveform import analyze, synthesize


def run(preparation, output):
    torch.set_num_threads(2)
    data = NativeMixtures(preparation, 'train_pack', 1)
    indices = list(range(48))
    rows = []
    checks = []
    with torch.no_grad():
        for index in indices:
            item = data[index]
            count = item['construction_count']
            wave = torch.from_numpy(item['references'][:count])
            z = analyze(wave)
            shifted = torch.fft.fftshift(z, dim=-2)
            restored = torch.fft.ifftshift(shifted, dim=-2)
            if not torch.equal(restored, z):
                raise ValueError('Frequency reordering changed complex values')
            reconstruction = synthesize(restored, wave.shape[-1])
            nmse = ((reconstruction - wave).abs().square().mean(-1) /
                    wave.abs().square().mean(-1)).tolist()
            checks += nmse
            power = z.abs().square().mean(-1).numpy().astype(np.float64)
            source_indices = data.rows[index]['indices'][:count]
            for source, library_index in enumerate(source_indices):
                meta = data.library.clips[int(library_index)]
                p = power[source]
                total = float(p.sum())
                # Each seam has 16 bins on either side, the same 6.25 MHz span.
                dc_left, dc_right = float(p[-16:].sum()/total), float(p[:16].sum()/total)
                nyq_left, nyq_right = float(p[240:256].sum()/total), float(p[256:272].sum()/total)
                rows.append(dict(index=index, source=source, category=meta['category'],
                    clip_id=meta['clip_id'], pack_id=meta['pack_id'],
                    old_dc_seam_energy_fraction=dc_left+dc_right,
                    new_nyquist_seam_energy_fraction=nyq_left+nyq_right,
                    dc_left_energy_fraction=dc_left, dc_right_energy_fraction=dc_right,
                    dc_both_sides_above_1pct=bool(min(dc_left, dc_right)>.01)))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['category']].append(row)
    summaries = []
    for category, group in sorted(grouped.items()):
        summaries.append(dict(category=category, appearances=len(group),
            unique_contexts=len({r['clip_id'] for r in group}),
            recording_groups=len({r['pack_id'] for r in group}),
            mean_dc_seam_energy_fraction=float(np.mean([r['old_dc_seam_energy_fraction'] for r in group])),
            mean_nyquist_seam_energy_fraction=float(np.mean([r['new_nyquist_seam_energy_fraction'] for r in group])),
            dc_both_sides_above_1pct=sum(r['dc_both_sides_above_1pct'] for r in group)))
    result = dict(status='COMPLETE', source_sha256=sha256(Path(__file__)),
        preparation_sha256=sha256(preparation/'PREPARATION.json'),
        selection='First48 TRAIN epoch1 mixtures, fixed before reading IQ; repeated appearances are not independent recordings',
        train_indices=indices, validation_iq_read=False, heldout_read=False,
        sample_rate_hz=100_000_000, fft_size=512, fine_hop=128, zero_frequency_at_bin=0,
        fftshift_zero_frequency_at_bin=256, seam_total_bandwidth_hz=6_250_000,
        shift_inverse_exact=True, maximum_roundtrip_nmse=max(checks),
        train_source_appearances=len(rows), by_category=summaries, rows=rows,
        current_training_changed=False, trained_shifted_model=False,
        interpretation='An input adjacency hypothesis only; neither a training gain nor evidence that FFT ordering caused the current reconstruction errors',
        time=time.time())
    write_json(output, result)
    print(__import__('json').dumps({k:result[k] for k in ('status','train_source_appearances','maximum_roundtrip_nmse','by_category')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.preparation, args.output)
