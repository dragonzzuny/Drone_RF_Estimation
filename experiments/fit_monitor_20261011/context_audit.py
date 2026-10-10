"""CPU audit of the CURRENT mean-context branch, not a candidate comparison."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tfgridnet_rf_20261011'))
import fit
from model import build


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    p = fit.w.read(ROOT / 'local/tfgridnet_diversity_20261011_v1/PROTOCOL.json')
    index = p['train_indices'][0]
    data = fit.worker.NativeMixtures(p['preparation'], 'train_pack', 1)
    features = torch.from_numpy(data.features[index].copy())[None]
    net = build().eval()
    ck = torch.load(p['parent_checkpoint'], map_location='cpu', weights_only=False)
    net.load_state_dict(ck['model'])
    del ck

    def branch(x):
        x = x.mean(-1, keepdim=True).expand_as(x)
        encoded = net.context_encoder(x)
        injection = .1 * net.context_projection(encoded).mean(-1)
        return injection, net.count_head(encoded.mean(-1))

    with torch.no_grad():
        original = branch(features)
        reversed_time = branch(features.flip(-1))
        # These tests intentionally cover the long branch only. Full waveform
        # inference is unnecessary for its mathematical permutation invariance.
        max_injection = float((original[0] - reversed_time[0]).abs().max())
        max_count = float((original[1] - reversed_time[1]).abs().max())
        relative_injection = float((original[0] - reversed_time[0]).square().sum()
                                   / original[0].square().sum().clamp_min(1e-30))
    x = features.clone().requires_grad_(True)
    branch(x)[0].square().mean().backward()
    max_time_gradient_variation = float((x.grad - x.grad[..., :1]).abs().max())
    assert max_time_gradient_variation == 0.
    fs = 100_000_000
    result = dict(status='COMPLETE', pid=os.getpid(), time=time.time(), train_index=index,
                  checkpoint_sha256=fit.w.digest(Path(p['parent_checkpoint'])),
                  model_source_sha256=fit.w.digest(ROOT / 'experiments/tfgridnet_rf_20261011/model.py'),
                  audit_source_sha256=fit.w.digest(Path(__file__)),
                  fine_complex_samples=63872, fine_duration_ms=63872/fs*1000,
                  fine_stft_bins=512, fine_stft_frames=500, fine_hop_us=128/fs*1e6,
                  long_context_tokens=255, long_token_us=8192/fs*1e6,
                  long_context_ms=255*8192/fs*1000,
                  tcn_theoretical_token_receptive_field=1+4*sum([1,2,4,8,16,32]),
                  mean_branch_time_permutation_invariant_in_exact_arithmetic=True,
                  current_crop_start_used_for_localization=False,
                  reversed_input_absolute_difference=float((features-features.flip(-1)).abs().max()),
                  reverse_max_injection_difference=max_injection,
                  reverse_relative_injection_squared_difference=relative_injection,
                  reverse_max_count_logit_difference=max_count,
                  input_gradient_max_variation_across_time=max_time_gradient_variation,
                  full_waveform_inference=False, optimizer_updates=0,
                  validation_read=False, heldout_read=False,
                  interpretation='Long-order information is removed before the encoder; local fine-grid time information remains. Not evidence that restoring order improves separation.')
    fit.w.write(ROOT / 'reports/2026-10-11/TFGRIDNET_CONTEXT_AUDIT.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
