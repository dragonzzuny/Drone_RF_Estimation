"""Numerical contract checks for weighted accumulation and original PCGrad."""
import hashlib
import json
from pathlib import Path
import torch
from gradient import CountAccumulator, surgery, seeded_rng


def run():
    torch.set_num_threads(2)
    g = torch.tensor([[1., 0.], [-1., 1.]], dtype=torch.float64)
    actual, receipt = surgery(g, seeded_rng())
    assert torch.allclose(actual, torch.tensor([.5, 1.5], dtype=torch.float64))
    assert torch.equal(g, torch.tensor([[1., 0.], [-1., 1.]], dtype=torch.float64))
    for g in (torch.eye(3, dtype=torch.float64), torch.ones(3, 5, dtype=torch.float64),
              torch.zeros(3, 4, dtype=torch.float64)):
        actual, receipt = surgery(g, seeded_rng())
        assert torch.equal(actual, g.sum(0)) and not receipt['projections']
    g = torch.tensor([[1., 0.], [-1., 0.], [0., 0.]], dtype=torch.float64)
    actual, _ = surgery(g, seeded_rng())
    assert torch.equal(actual, torch.zeros(2, dtype=torch.float64))
    # Independently implement Algorithm 1 with Python scalar arithmetic.
    source = [[1., 2., -3.], [-2., 1., 1.], [1., -2., 1.]]
    reference = [row[:] for row in source]
    rng = seeded_rng()
    for i in range(3):
        order = [j for j in range(3) if i != j]; rng.shuffle(order)
        for j in order:
            dot = sum(a*b for a, b in zip(reference[i], source[j]))
            denominator = sum(a*a for a in source[j])
            if dot < 0:
                reference[i] = [a-dot/denominator*b for a, b in zip(reference[i], source[j])]
    expected = torch.tensor([sum(row[k] for row in reference) for k in range(3)], dtype=torch.float64)
    actual, _ = surgery(torch.tensor(source, dtype=torch.float64), seeded_rng())
    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    # Unequal source-count frequencies must preserve the ordinary sample mean.
    torch.manual_seed(0)
    net = torch.nn.Linear(3, 2).double()
    x, y = torch.randn(7, 3, dtype=torch.float64), torch.randn(7, 2, dtype=torch.float64)
    labels = [1, 1, 1, 1, 2, 3, 3]
    ((net(x)-y).square().mean()).backward()
    expected = torch.cat([p.grad.flatten() for p in net.parameters()]).clone()
    net.zero_grad(set_to_none=True)
    accumulator = CountAccumulator(net.parameters())
    for i, count in enumerate(labels):
        ((net(x[i:i+1])-y[i:i+1]).square().mean()/len(labels)).backward()
        accumulator.collect(count)
    error = float((accumulator.groups.sum(0)-expected).abs().max())
    assert error < 1e-14 and accumulator.counts == [4, 1, 2]
    accumulator.assign(accumulator.groups.sum(0))
    assert torch.allclose(torch.cat([p.grad.flatten() for p in net.parameters()]), expected)
    accumulator.reset()
    assert not accumulator.groups.any() and accumulator.counts == [0, 0, 0]
    here = Path(__file__).resolve().parent; root = here.parents[1]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (here/'gradient.py', Path(__file__))}
    return dict(status='PASS',unequal_group_sample_mean_max_error=error,
                independent_scalar_algorithm=True,original_reference_vectors_unchanged=True,
                zero_and_nonconflicting_cases=True,opposing_collinear_case=True,
                source_sha256=hashes,gpu_use=False,recorded_iq_reads=0)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run();a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
