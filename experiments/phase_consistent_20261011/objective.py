"""Train the actual eight-view prediction and its within-orbit consistency.

Two-pass output-gradient replay keeps one full U-Net graph in memory at a
time. It is an exact chain rule for deterministic forwards and the selected
piecewise-constant slot permutations, not distillation or a frozen teacher.
"""
import math
import torch
from torch.nn import functional as F
from common import worker, phase

DEGREES = (0, 90, 180, 270, 45, 135, 225, 315)
FACTORS = (1+0j, 1j, -1+0j, -1j) + tuple(
    complex(math.cos(math.radians(x)), math.sin(math.radians(x))) for x in DEGREES[4:])

def rotate(item, factor):
    # References, class labels and true counts cannot enter the model.
    result = {k: item[k] for k in ('mixture', 'context_features', 'crop_start',
                                  'long_mixture', 'long_start') if k in item}
    result['mixture'] = result['mixture'] * factor
    if 'long_mixture' in result:
        result['long_mixture'] = result['long_mixture'] * factor
    return result

def restore(predicted, factor, order):
    value = predicted / factor
    return torch.cat((value[:, order], value[:, 3:]), 1)

def average(views):
    # Same nesting and summation order as the selected inference implementation.
    return .5 * (views[:4].mean(0) + views[4:].mean(0))

def collect(net, predict, item):
    views, orders, logits = [], [], None
    for i, factor in enumerate(FACTORS):
        output, count = predict(net, rotate(item, factor))
        if i == 0:
            views.append(output); orders.append([0, 1, 2]); logits = count
        else:
            if not torch.equal(count, logits):
                raise RuntimeError('Power-context count changed under common phase')
            value, order = phase.align_predictions(output / factor, views[0])
            views.append(value); orders.append(order)
    return torch.stack(views), logits, orders

def losses(views, logits, item, consistency_weight=.1):
    mean = average(views)
    result = worker.pit_waveform_loss(mean, item['references'], item['active'], item['mixture'])
    # PIT is for TRAIN supervision only. Inference view alignment above is
    # prediction-only, and never uses this reference-dependent assignment.
    source_variance = (views[:, :, :3] - mean[None, :, :3]).abs().square().mean((0, 3))
    assigned = source_variance.gather(1, result['assignment'])
    mix_power = item['mixture'].abs().square().mean(-1).clamp_min(1e-8)
    ref_power = item['references'].abs().square().mean(-1)
    scale = torch.where(item['active'], torch.maximum(ref_power, 1e-6 * mix_power[:, None]),
                        mix_power[:, None])
    consistency = (assigned / scale).mean()
    consistency = consistency + ((views[:, :, -1] - mean[None, :, -1]).abs().square()
                                  .mean((0, 2)) / mix_power).mean()
    count = F.cross_entropy(logits, item['construction_count'] - 1)
    loss = result['loss'] + consistency_weight * consistency + .1 * count
    return loss, dict(waveform=float(result['loss'].detach()),
                     consistency=float(consistency.detach()), count=float(count.detach()),
                     total=float(loss.detach()))

def backward_replay(net, predict, item, divisor=1., consistency_weight=.1, check_replay=False):
    # train/eval must have deterministic behavior; callers reject Dropout/BN.
    with torch.no_grad():
        stored, stored_logits, orders = collect(net, predict, item)
    leaf = stored.detach().requires_grad_(True)
    count_leaf = stored_logits.detach().requires_grad_(True)
    loss, receipt = losses(leaf, count_leaf, item, consistency_weight)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite C8 training loss')
    view_gradient, count_gradient = torch.autograd.grad(loss / divisor, (leaf, count_leaf))
    for i, factor in enumerate(FACTORS):
        raw, logits = predict(net, rotate(item, factor))
        output = restore(raw, factor, orders[i])
        if check_replay:
            torch.testing.assert_close(output.detach(), stored[i], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(logits.detach(), stored_logits, rtol=0, atol=0)
        if i == 0:
            torch.autograd.backward((output, logits), (view_gradient[i], count_gradient))
        else:
            output.backward(view_gradient[i])
        del raw, output, logits
    receipt['prediction_only_orders'] = orders
    return receipt

