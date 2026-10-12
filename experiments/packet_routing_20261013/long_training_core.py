"""Conditional full-length training support; no registered or queued run.

Preserve the original per-window source normalization and mean loss. A single
forward processes the concatenation, and one common permutation supervises its
three contiguous parts. This is different from normalizing one 1.916ms loss.
"""
import torch
from checkpoint_blocks import combine_items, enable_checkpointing
from sequence_objective import assignments, fixed_loss, assert_replay_safe


def predict_contiguous(net, predict, items):
    combined = combine_items(items)
    estimate, logits = predict(net, combined)
    lengths = [item['mixture'].shape[-1] for item in items]
    parts = list(estimate.split(lengths, dim=-1))
    assert len(parts) == len(items) == 3
    return parts, logits


def contiguous_loss(estimates, logits, items):
    """Exactly the mean original window loss under a single hard PIT mapping."""
    order, diagnostic = assignments(estimates, items, 'shared')
    assert order[0] == order[1] == order[2]
    waveform = torch.stack([fixed_loss(y, item, p)
                            for y, item, p in zip(estimates, items, order)])
    count = torch.nn.functional.cross_entropy(logits, items[0]['construction_count'] - 1)
    total = waveform + .1 * count
    if not torch.isfinite(total).all():
        raise ValueError('Nonfinite long-input objective')
    values = torch.stack([total.detach(), waveform.detach(), count.detach().expand(3)], dim=1)
    return total.mean(), values.cpu().tolist(), diagnostic


def train_bundle(net, predict, items, divisor):
    assert divisor > 0
    estimate, logits = predict_contiguous(net, predict, items)
    loss, values, diagnostic = contiguous_loss(estimate, logits, items)
    (loss / divisor).backward()
    return values, diagnostic


def load_model(path, architecture, device):
    from long_temporal_candidate import LongTemporalUNet
    from models import build
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    state = saved['model']
    if architecture == 'long_plain':
        net = build('unet_mean')
        net.load_state_dict(state)
        expected = 32142859
    elif architecture == 'long_gated':
        torch.manual_seed(0)
        net = LongTemporalUNet()
        if any(key.startswith('gates.') for key in state):
            net.load_state_dict(state)
        else:
            net.initialize_parent(state)
        expected = 35820299
    else:
        raise ValueError('Explicit long_plain or long_gated architecture required')
    assert sum(p.numel() for p in net.parameters()) == expected
    assert_replay_safe(net)
    enable_checkpointing(net)
    return net.to(device), saved
