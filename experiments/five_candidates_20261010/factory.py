"""Reuse full, previously checked architectures and optimizer partitions."""
import importlib.util
import sys
import torch
from workflow import ROOT, SPECS

sys.path.insert(0, str(ROOT/'experiments/source_interaction_20261010'))
import train_comparison as worker


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def build(name, parent=None, device='cpu'):
    net = worker.make_model('retained_unet')
    if parent is not None:
        saved = torch.load(parent, map_location='cpu', weights_only=False, mmap=True)
        net.load_state_dict(saved['model']); del saved
    if name == 'frozen_tf':
        module('five_axis_adapter', 'experiments/tf_axis_20261010/adapter.py').augment(net)
    elif name == 'ordered_context':
        module('five_ordered_context', 'experiments/ordered_context_20261010/model.py').augment(net)
    elif name == 'balanced_head':
        from balanced_source_head import augment
        augment(net)
    elif name != 'wave_guard':
        raise ValueError(name)
    net.to(device)
    if name == 'frozen_tf':
        opt, params = module('five_axis_adaptation', 'experiments/tf_axis_20261010/adaptation.py').configure(net, 'frozen_backbone')
    elif name == 'ordered_context':
        gate = [p for k,p in net.named_parameters() if k == 'context_encoder.order_gate']
        base = [p for k,p in net.named_parameters() if k != 'context_encoder.order_gate']
        opt = torch.optim.AdamW([dict(params=base, lr=1e-5, weight_decay=1e-4),
                                dict(params=gate, lr=1e-2, weight_decay=0.)], foreach=False)
    elif name == 'balanced_head':
        from head_learning_rate_probe import optimizer
        opt = optimizer(net, 1e-3)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=1e-4, foreach=False)
    params = [p for p in net.parameters() if p.requires_grad]
    assert sum(p.numel() for p in net.parameters()) == SPECS[name]['parameters']
    assert sum(p.numel() for p in params) == SPECS[name]['trainable']
    assert len({id(p) for g in opt.param_groups for p in g['params']}) == len(params)
    return net, opt, params


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a) == type(b) and a == b


def restore(name, path, epoch, device):
    value = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert value['epoch'] == epoch and value['updates'] == 75*epoch
    net, opt, params = build(name, device=device)
    net.load_state_dict(value['model']); opt.load_state_dict(value['optimizer'])
    assert equal(net.state_dict(), value['model']) and equal(opt.state_dict(), value['optimizer'])
    assert len(opt.state) == len(params)
    assert {int(s['step']) for s in opt.state.values()} == {75*epoch}
    torch.set_rng_state(value['torch_rng'])
    assert torch.equal(torch.get_rng_state(), value['torch_rng'])
    if device == 'cuda':
        torch.cuda.set_rng_state_all(value['cuda_rng'])
        assert equal(torch.cuda.get_rng_state_all(), value['cuda_rng'])
    meta = {k:value[k] for k in ('best','protocol_sha256','epoch','updates')}
    meta['lr_monitor'] = value.get('lr_monitor', dict(best=None,bad=0,reductions=0))
    # The guard has no randomized projection. Preserve the legacy RNG anyway.
    meta['projection_rng'] = value.get('projection_rng')
    return net,opt,params,meta


def predict_train(name, net, item):
    if name != 'ordered_context':
        return worker.predict(net,item)
    from torch.utils.checkpoint import checkpoint
    return checkpoint(lambda mix,context,position: worker.predict(net,dict(
        mixture=mix,context_features=context,crop_start=position)),
        item['mixture'],item['context_features'],item['crop_start'],use_reentrant=False)


def guard_components():
    sys.path.insert(0,str(ROOT/'experiments/count_pcgrad_20261010'))
    from wave_update_guard import WaveAccumulator, correct
    return WaveAccumulator, correct
