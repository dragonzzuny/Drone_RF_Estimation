"""Full-size synthetic CPU interface/gradient check; no recorded I/Q reads."""
import argparse
import gc
from pathlib import Path
import time

import torch
from torch.nn import functional as F

import output_parameterization as candidate
import watch_epochs as watch
from drone_rf.context_model import ContextualSeparator
from drone_rf.waveform import analyze, synthesize, waveform_metrics
from drone_rf.losses import pit_waveform_loss


def run(public):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    references = torch.randn(1, 3, 63872, dtype=torch.complex64)
    references[:, 1] *= .316227766
    references[:, 2] = 0
    mixture = references.sum(1)
    active = torch.tensor([[True, True, False]])
    context = torch.randn(1, 65, 255)
    start = torch.tensor([4097])
    z = analyze(mixture)
    mapping, masking = (candidate.build(mode).eval() for mode in candidate.MODES)
    assert all(torch.equal(v, masking.state_dict()[k]) for k, v in mapping.state_dict().items())
    parent = ContextualSeparator('tcn', 'mean').eval()
    # Verify the mapping clone against the frozen parent at a NONZERO head.
    torch.nn.init.normal_(mapping.output.weight, std=1e-3)
    parent.load_state_dict(mapping.state_dict(), strict=True)
    with torch.no_grad():
        expected = parent(z, context, start)
        actual = mapping(z, context, start)
        for key in ('estimates', 'count_logits'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    del parent, expected, actual
    gc.collect()
    torch.nn.init.zeros_(mapping.output.weight)
    assert all(torch.equal(v, masking.state_dict()[k]) for k, v in mapping.state_dict().items())
    with torch.no_grad():
        a, b = mapping(z, context, start), masking(z, context, start)
        torch.testing.assert_close(a['estimates'], b['estimates'], rtol=0, atol=0)
        torch.testing.assert_close(a['estimates'], z[:, None].expand(-1, 4, -1, -1)/4, rtol=0, atol=0)
    results = []
    for net in (mapping, masking):
        before = time.time()
        optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4, foreach=False)
        norms = []
        for step in (1, 2):
            net.train(); optimizer.zero_grad(set_to_none=True)
            result = net(z, context, start)
            estimate = synthesize(result['estimates'], mixture.shape[-1])
            loss = pit_waveform_loss(estimate, references, active, mixture)['loss']
            loss = loss + .1*F.cross_entropy(result['count_logits'], torch.tensor([1]))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
            if norm <= 0 or net.output.weight.grad.abs().sum() <= 0:
                raise ValueError('Missing head gradient')
            # Zero head delays waveform-backbone gradients on the first update.
            if step == 2 and not any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.down.parameters()):
                raise ValueError('Waveform encoder lacks gradient after head update')
            optimizer.step(); norms.append(norm)
        net.eval()
        with torch.no_grad():
            result = net(z, context, start)
            estimate = synthesize(result['estimates'], mixture.shape[-1])
            score = waveform_metrics(estimate, references, active, mixture)
            if not torch.isfinite(score['nmse']).all() or score['sum_relative_error'].max() > 1e-9:
                raise ValueError('Metric or mixture consistency failure')
            if net.mode == 'masking':
                zero_bins = z.clone(); zero_bins[:, 7:11, 6:10] = 0
                zero_result = net(zero_bins, context, start)['estimates']
                if not torch.equal(zero_result[:, :, 7:11, 6:10], torch.zeros_like(zero_result[:, :, 7:11, 6:10])):
                    raise ValueError('Masking failed zero-mixture-bin property')
        results.append(dict(mode=net.mode, full_parameters=candidate.PARAMETERS,
            synthetic_steps=2, finite_gradient_norms=norms,
            sum_relative_error=float(score['sum_relative_error'].max()),
            seconds=time.time()-before, weights_discarded=True))
        del optimizer, result, estimate, loss, score
        gc.collect()
    sources = {str(p.relative_to(watch.ROOT)): watch.digest(p)
               for p in (Path(__file__), Path(candidate.__file__))}
    evidence = dict(status='PASS', source_sha256=sources,
        full_samples=63872, complex_stft_shape=list(z.shape), context_shape=list(context.shape),
        parameter_tensors_identical=True, initial_predictions_identical=True,
        mapping_nonzero_head_matches_parent_exactly=True, head_and_encoder_gradients_checked=True,
        zero_mixture_bins_force_zero_mask_output=True, results=results,
        recorded_iq_reads=0, validation_read=False, heldout_read=False,
        performance_evaluation=False, gpu_fit_pending=True)
    watch.write(public, evidence)
    print('PASS full-capacity/full-input synthetic CPU check; no RF performance conclusion')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    run(a.output.resolve())
