"""Controller/optimizer check; reuse the unchanged full-input model/loss check."""
import ast
from pathlib import Path

import torch

import watch_epochs as watch
from low_lr_unet_comparison import build


def main():
    torch.set_num_threads(2)
    worker = Path(__file__).with_name('low_lr_unet_comparison.py')
    tree = ast.parse(worker.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    train = functions['train_epoch']
    optimizers = [n for n in ast.walk(train) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'AdamW']
    assert len(optimizers) == 1
    keywords = {n.arg: n.value for n in optimizers[0].keywords}
    assert ast.unparse(keywords['lr']) == "protocol['learning_rate']"
    text = ast.unparse(functions['register'])
    for required in ('learning_rate=0.0001', 'reference_learning_rate=0.0005',
                     'epochs_per_arm=1', 'new_updates_per_arm=75', 'train_schedule_epochs=[1]'):
        assert required in text, required
    assert 'range(1, 2)' in ast.unparse(functions['run'])
    assert "final['updates'] != 75" in ast.unparse(functions['run'])
    assert "final['epoch'] != 1" in ast.unparse(functions['run'])
    # Architecture and forward/loss functions are identical to the prior worker.
    prior = ast.parse(worker.with_name('robust_unet_comparison.py').read_text())
    before = next(n for n in prior.body if isinstance(n, ast.FunctionDef) and n.name == 'train_epoch')
    calls = lambda node: [ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Call)
                          and ast.unparse(n.func) in ('build', 'predict', 'loss_module.objective',
                                                     'fit.base.batch', 'fit.base.validate')]
    assert calls(before) == calls(train)
    model = build('unet_mean')
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 32_142_859
    del model
    # An optimizer unit check, not a miniature separator or an RF fitting trial.
    deltas = []
    for lr in (5e-4, 1e-4):
        parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=lr, weight_decay=1e-4, foreach=False)
        assert not optimizer.state
        parameter.grad = torch.tensor([.5], dtype=torch.float64)
        optimizer.step()
        deltas.append(float(1-parameter.detach()[0]))
        assert float(optimizer.state[parameter]['step']) == 1
    assert abs(deltas[1]/deltas[0]-.2) < 1e-10
    prior_check = watch.ROOT/'reports/2026-10-09/ROBUST_NMSE_CPU_CHECK.json'
    previous = watch.read(prior_check)
    assert previous['status'] == 'PASS' and not previous['trained'] and not previous['recorded_iq_reads']
    for rel, digest in previous['source_sha256'].items():
        assert watch.digest(watch.ROOT/rel) == digest
    result = dict(status='PASS', worker_sha256=watch.digest(worker), parameters=parameters,
                  source_sha256={str(Path(__file__).relative_to(watch.ROOT)): watch.digest(Path(__file__))},
                  reused_full_input_check_sha256=watch.digest(prior_check),
                  same_model_forward_loss_calls=True, configured_epochs_per_arm=1,
                  configured_updates_per_arm=75, learning_rate=1e-4, reference_learning_rate=5e-4,
                  synthetic_optimizer_step_ratio=deltas[1]/deltas[0], recorded_iq_reads=0,
                  rf_training_updates=0, synthetic_optimizer_unit_steps=2,
                  note='No new RF/model fitting. Full-capacity construction and unchanged input/loss checks plus optimizer unit check.')
    watch.write(watch.ROOT/'reports/2026-10-09/LOW_LR_UNET_CPU_CHECK.json', result)
    print(result)


if __name__ == '__main__':
    main()
