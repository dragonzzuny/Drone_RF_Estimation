"""Prepared, untrained extension retaining the entire parent U-Net.

Six dilated gated convolutions per frequency cell extend the bottleneck time
connections. Zero projection preserves the parent output exactly initially.
This is an RF Challenge-inspired adapter, not a novel WaveNet reproduction.
"""
from pathlib import Path
import sys
import math
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/rfuav_dense_gated_20261008'))
from models import GatedUNet,GatedResidual


class LongGates(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce=nn.Conv1d(1024,256,1)
        self.blocks=nn.ModuleList(GatedResidual(256,d) for d in (1,2,4,8,16,32))
        self.expand=nn.Conv1d(256,1024,1)
        nn.init.zeros_(self.expand.weight);nn.init.zeros_(self.expand.bias)

    def forward(self,x):
        b,c,f,t=x.shape
        h=self.reduce(x.permute(0,2,1,3).reshape(b*f,c,t));skips=0
        for block in self.blocks:h,s=block(h);skips=skips+s
        delta=self.expand(skips/math.sqrt(len(self.blocks)))
        return x+delta.reshape(b,f,c,t).permute(0,2,1,3)


class LongTemporalUNet(GatedUNet):
    def __init__(self):
        super().__init__()
        self.gates=LongGates()

    def initialize_parent(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected and missing and all(k.startswith('gates.') for k in missing)
        for key,tensor in state.items():assert torch.equal(self.state_dict()[key],tensor),key
        assert not bool(self.gates.expand.weight.count_nonzero())
        assert not bool(self.gates.expand.bias.count_nonzero())
        return self
