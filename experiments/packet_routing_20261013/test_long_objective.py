"""Numerical objective/gradient equivalence on synthetic complex windows."""
import itertools
import json
from pathlib import Path
import torch
from sequence_data import ROOT
from sequence_objective import fixed_loss
from long_training_core import contiguous_loss


def main():
    torch.manual_seed(0)
    results = []
    for count in (1, 2, 3):
        items, predictions = [], []
        for _ in range(3):
            reference = torch.randn(1, 3, 32, dtype=torch.complex64)
            reference[:, count:] = 0
            mixture = reference.sum(1)
            items.append(dict(references=reference, mixture=mixture,
                active=torch.tensor([[j < count for j in range(3)]]),
                construction_count=torch.tensor([count])))
            predictions.append(torch.randn(1, 4, 32, dtype=torch.complex64, requires_grad=True))
        logits = torch.randn(1, 3, requires_grad=True)
        loss, _, diagnostic = contiguous_loss(predictions, logits, items)
        costs = torch.stack([torch.stack([fixed_loss(y, item, p)
            for y, item in zip(predictions, items)]).mean()
            for p in itertools.permutations(range(3))])
        direct = costs.min() + .1 * torch.nn.functional.cross_entropy(logits, torch.tensor([count - 1]))
        torch.testing.assert_close(loss, direct, rtol=1e-6, atol=1e-7)
        ga = torch.autograd.grad(loss, [*predictions, logits], retain_graph=True)
        gb = torch.autograd.grad(direct, [*predictions, logits])
        maximum = max(float((a - b).abs().max()) for a, b in zip(ga, gb))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        assert diagnostic['shared_min_loss'] + 1e-6 >= diagnostic['local_min_loss']
        results.append(dict(count=count, gradient_max_difference=maximum,
                            objective_difference=float((loss - direct).detach().abs())))
    result = dict(status='PASS', checks=results, rf_training_updates=0,
                  note='Synthetic objective algebra only; full-length GPU preflight pending')
    (ROOT/'reports/2026-10-13/LONG_OBJECTIVE_CPU_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
