"""PCGrad on sample-weighted source-count task gradients (Yu et al., 2020).

Each row is sum(example gradients in that count)/effective_batch, NOT a
separately normalized group mean. Summing rows recovers the original objective.
Projection always uses the ORIGINAL reference row, with seeded random order.
"""
import random
import torch


def surgery(groups, rng):
    if groups.ndim != 2 or not torch.isfinite(groups).all():
        raise ValueError('Expected finite task-by-parameter gradients')
    norms2 = torch.stack([torch.dot(g, g) for g in groups])
    gram = torch.stack([torch.stack([torch.dot(a, b) for b in groups]) for a in groups])
    projected = groups.clone()
    events, orders = [], []
    for i in range(len(groups)):
        order = [j for j in range(len(groups)) if j != i]
        rng.shuffle(order)
        orders.append(order)
        for j in order:
            dot = torch.dot(projected[i], groups[j])
            if dot.item() < 0 and norms2[j].item() > 0:
                coefficient = dot / norms2[j]
                projected[i].sub_(coefficient * groups[j])
                events.append(dict(task=i+1, reference=j+1, coefficient=float(coefficient)))
    result = projected.sum(0)
    ordinary = groups.sum(0)
    stats = dict(gram=gram.tolist(), projection_orders=orders, projections=events,
                 ordinary_norm=float(ordinary.norm()), projected_norm=float(result.norm()),
                 change_norm=float((result-ordinary).norm()),
                 projected_direction_dot_original_tasks=[float(torch.dot(result, g)) for g in groups])
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite projected gradient')
    return result, stats


class CountAccumulator:
    """One backward per example; retain three gradient sums, not three graphs."""
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.sizes = [p.numel() for p in self.parameters]
        p = self.parameters[0]
        self.groups = torch.zeros(3, sum(self.sizes), device=p.device, dtype=p.dtype)
        self.views = [list(row.split(self.sizes)) for row in self.groups]
        self.counts = [0, 0, 0]

    @torch.no_grad()
    def collect(self, count):
        if count not in (1, 2, 3):
            raise ValueError(count)
        for p, view in zip(self.parameters, self.views[count-1]):
            # All retained U-Net parameters participate in every original loss.
            # Fail rather than silently change AdamW's absent-gradient semantics.
            if p.grad is None:
                raise ValueError('Disconnected parameter')
            view.add_(p.grad.reshape(-1))
            p.grad = None
        self.counts[count-1] += 1

    @torch.no_grad()
    def assign(self, flat):
        for p, part in zip(self.parameters, flat.split(self.sizes)):
            p.grad = part.view_as(p)

    @torch.no_grad()
    def reset(self):
        self.groups.zero_()
        self.counts = [0, 0, 0]


def seeded_rng(seed=0):
    # Separate from torch's model/data RNG; surgery must not reshuffle mixtures.
    return random.Random(seed)
