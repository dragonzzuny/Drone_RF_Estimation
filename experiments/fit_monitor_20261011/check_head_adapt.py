"""CPU optimizer-contract check, not a reduced-input separation experiment."""
from pathlib import Path
import json
import sys
import torch
import head_adapt as head


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    p = head.w.read(head.COMPARATOR / 'PROTOCOL.json')
    saved = torch.load(p['parent_checkpoint'], map_location='cpu', weights_only=False)
    net = head.base.build()
    net.load_state_dict(saved['model'])
    names = {'deconv.weight','deconv.bias'}
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(name in names)
    count = sum(t.numel() for t in net.parameters())
    learn = sum(t.numel() for t in net.parameters() if t.requires_grad)
    assert count == 9004297 and learn == 9224
    opt = torch.optim.AdamW(net.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'], foreach=False)
    opt.load_state_dict(saved['optimizer'])
    # Exercise the real output module and original optimizer mapping only.
    # This artificial latent tensor is not RF data or a performance benchmark.
    hidden = torch.randn(1,128,7,9)
    loss = net.deconv(hidden).square().mean()
    loss.backward()
    assert {n for n,t in net.named_parameters() if t.grad is not None} == names
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    changes = []
    for name, parameter in net.named_parameters():
        expected = 65 if name in names else 64
        assert int(opt.state[parameter]['step']) == expected
        if not torch.equal(parameter, saved['model'][name]):
            changes.append(name)
    assert set(changes) == names
    for name, tensor in net.state_dict().items():
        if name not in names:
            assert torch.equal(tensor, saved['model'][name])
    receipt = dict(status='PASS', parameters=count, training_parameters=learn, changed_names=changes,
        frozen_optimizer_steps=64, head_optimizer_steps=65, synthetic_latent_unit_check_only=True,
        model_capacity_reduced=False, waveform_inference=False, training_run_started=False,
        checkpoint_written=False, validation_read=False, heldout_read=False,
        parent_checkpoint_sha256=p['parent_checkpoint_sha256'],
        source_sha256={str(path.relative_to(head.ROOT)):head.w.digest(path)
                       for path in (Path(__file__),Path(head.__file__))})
    head.w.write(head.ROOT / 'reports/2026-10-11/TFGRIDNET_HEAD_ADAPT_CPU_CHECK.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
