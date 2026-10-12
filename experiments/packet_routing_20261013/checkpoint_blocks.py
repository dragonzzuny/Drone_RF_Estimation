"""Activation-memory support for a possible longer-input experiment.

This changes neither parameters nor forward outputs. No existing trainer imports
it. Enabling it is NOT an architecture or performance experiment by itself.
"""
from types import MethodType
import torch
from torch.utils.checkpoint import checkpoint


def enable_checkpointing(net):
    """Keep state_dict names and full widths; recompute deterministic U-Net blocks."""
    blocks=list(net.down)+list(net.up)
    assert len(blocks)==9
    for block in blocks:
        if getattr(block,'_activation_checkpoint_enabled',False):
            raise ValueError('Block checkpointing already enabled')
        if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in block.modules()):
            raise ValueError('Stateful block is not supported')
    for block in blocks:
        original=block.forward
        def wrap(method):
            def forward(self,x):
                if self.training and torch.is_grad_enabled():
                    return checkpoint(method,x,use_reentrant=False,preserve_rng_state=True)
                return method(x)
            return forward
        block.forward=MethodType(wrap(original),block)
        block._activation_checkpoint_enabled=True
    return net


def combine_items(items):
    """Concatenate contiguous observed windows, preserving one long context."""
    if len(items)!=3:raise ValueError('Exactly three windows required')
    for left,right in zip(items,items[1:]):
        assert torch.equal(left['crop_start']+left['mixture'].shape[-1],right['crop_start'])
        assert torch.equal(left['context_features'],right['context_features'])
        assert torch.equal(left['construction_count'],right['construction_count'])
    return dict(mixture=torch.cat([v['mixture'] for v in items],-1),
        references=torch.cat([v['references'] for v in items],-1),
        active=torch.stack([v['active'] for v in items]).any(0),
        context_features=items[0]['context_features'],crop_start=items[0]['crop_start'],
        construction_count=items[0]['construction_count'])
