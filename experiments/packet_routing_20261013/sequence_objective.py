"""Matched independent versus shared PIT assignments across adjacent windows.

The source-normalized loss and power floors stay exactly those of the parent.
Only the admissible permutation changes. This is not a larger receptive field.
"""
import itertools
import torch
from drone_rf.losses import pit_waveform_loss

PERMUTATIONS = tuple(itertools.permutations(range(3)))


def fixed_loss(estimate, item, permutation):
    return pit_waveform_loss(estimate[:, [*permutation, 3]], item['references'],
        item['active'], item['mixture'], assignment_mode='fixed')['loss']


@torch.no_grad()
def assignments(estimates, items, mode):
    if mode not in ('independent', 'shared') or len(estimates) != len(items) or not estimates:
        raise ValueError('Matching nonempty window lists and a registered assignment mode required')
    costs = torch.stack([torch.stack([fixed_loss(y, item, p) for p in PERMUTATIONS])
                         for y, item in zip(estimates, items)])
    if not torch.isfinite(costs).all():
        raise ValueError('Nonfinite candidate assignment cost')
    local = costs.argmin(-1)
    global_index = costs.mean(0).argmin()
    chosen = local if mode == 'independent' else global_index.expand_as(local)
    return [PERMUTATIONS[int(i)] for i in chosen], dict(
        local_indices=local.cpu().tolist(), shared_index=int(global_index),
        local_min_loss=float(costs.min(-1).values.mean()),
        shared_min_loss=float(costs.mean(0).min()),
        local_disagreement=len(set(local.cpu().tolist())) > 1,
        costs=costs.cpu().tolist())


def assert_replay_safe(net):
    for layer in net.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            raise ValueError('Running-stat layers require an explicit replay-state implementation')
        if isinstance(layer, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d,
                              torch.nn.Dropout3d, torch.nn.AlphaDropout)) and layer.p:
            raise ValueError('Stochastic layers require an explicit replay RNG implementation')


def replay_bundle(net, predict, items, mode, divisor):
    """Exact hard-assignment gradient with one window activation graph at a time.

    Both arms run the same no-gradient selection pass and gradient replay pass.
    No optimizer update occurs between them. Inputs and weights stay fixed.
    """
    assert divisor > 0
    with torch.no_grad():
        predictions = [predict(net, item)[0] for item in items]
    order, diagnostic = assignments(predictions, items, mode)
    losses = []
    max_replay_error = 0.
    for item, target, permutation in zip(items, predictions, order):
        estimate, logits = predict(net, item)
        error = float((estimate.detach()-target).abs().max())
        max_replay_error = max(max_replay_error, error)
        if not torch.allclose(estimate.detach(), target, rtol=1e-5, atol=2e-6):
            raise ValueError(f'Forward replay mismatch: {error}')
        wave = fixed_loss(estimate, item, permutation)
        count = torch.nn.functional.cross_entropy(logits, item['construction_count']-1)
        loss = wave + .1*count
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        (loss/(divisor*len(items))).backward()
        losses.append([float(loss.detach()), float(wave.detach()), float(count.detach())])
    return losses, dict(diagnostic, max_replay_error=max_replay_error)
