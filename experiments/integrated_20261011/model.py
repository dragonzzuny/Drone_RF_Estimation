"""Full parent + ordered context + dual-axis adapter + balanced source head."""
import importlib.util
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/source_interaction_20261010'))
import train_comparison as worker
from balanced_source_head import augment as add_head


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def augment(net):
    module('integrated_order', 'experiments/ordered_context_20261010/model.py').augment(net)
    module('integrated_axis', 'experiments/tf_axis_20261010/adapter.py').augment(net)
    add_head(net)
    assert sum(p.numel() for p in net.parameters()) == 37_444_428
    return net


def is_added(name):
    return (name == 'context_encoder.order_gate' or name.startswith('tf_axes.')
            or (name.startswith('output.') and not name.startswith('output.base.')))


def configure(net):
    groups = {'order': [], 'axes': [], 'head': []}
    for name, p in net.named_parameters():
        p.requires_grad_(is_added(name))
        if p.requires_grad:
            groups['order' if name == 'context_encoder.order_gate' else
                   'axes' if name.startswith('tf_axes.') else 'head'].append(p)
    rates = dict(order=1e-2, axes=1e-4, head=1e-3)
    opt = torch.optim.AdamW([dict(params=ps, lr=rates[name], name=name,
        weight_decay=0. if name == 'order' else 1e-4) for name, ps in groups.items()], foreach=False)
    named = [(n,p) for n,p in net.named_parameters() if p.requires_grad]
    assert sum(p.numel() for _,p in named) == 5_301_569
    assert len({id(p) for g in opt.param_groups for p in g['params']}) == len(named)
    return opt, named


def build(parent, device='cuda'):
    torch.manual_seed(0)
    net = worker.make_model('retained_unet')
    saved = torch.load(parent, map_location='cpu', weights_only=False, mmap=True)
    net.load_state_dict(saved['model'])
    del saved
    augment(net)
    net.to(device)
    opt, named = configure(net)
    return net, opt, named


def check_frozen(net, parent):
    saved = torch.load(parent, map_location='cpu', weights_only=False, mmap=True)['model']
    state = net.state_dict()
    for name, value in saved.items():
        mapped = ('context_encoder.base.'+name[len('context_encoder.'):] if name.startswith('context_encoder.')
                  else 'output.base.'+name[len('output.'):] if name.startswith('output.') else name)
        assert torch.equal(state[mapped].detach().cpu(), value), mapped
    return True


def guard_components():
    sys.path.insert(0, str(ROOT/'experiments/count_pcgrad_20261010'))
    from wave_update_guard import WaveAccumulator, correct
    return WaveAccumulator, correct
