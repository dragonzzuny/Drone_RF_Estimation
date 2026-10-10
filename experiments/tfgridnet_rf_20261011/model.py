"""Full six-block TF-GridNetV2 RF adaptation, not a speech reproduction.

Reference capacity: D128, six blocks, bidirectional LSTM hidden192, four heads.
RF changes: full two-sided512 STFT, three unordered slots plus background,
original mixture-only mean long context/count head, original sum projection.
No reference waveform, source identity or actual count enters forward.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/rfuav_dense_gated_20261008/vendor'))
from drone_rf.context_model import TemporalEncoder
from vendor.gridnet_block import GridNetV2Block


class ChunkedLSTM(nn.Module):
    """Serialize independent sequences, keeping every weight and time sample."""
    def __init__(self,module,chunk=32):
        super().__init__();self.lstm=module;self.chunk=chunk

    def forward(self,x):
        def apply(t):return self.lstm(t)[0]
        parts=[]
        for t in x.split(self.chunk,dim=0):
            parts.append(checkpoint(apply,t,use_reentrant=False) if self.training and torch.is_grad_enabled() else apply(t))
        return torch.cat(parts,dim=0),None


def memory_chunks(block):
    block.intra_rnn=ChunkedLSTM(block.intra_rnn)
    block.inter_rnn=ChunkedLSTM(block.inter_rnn)
    return block


class RFGridNet(nn.Module):
    def __init__(self):
        super().__init__();self.max_sources=3;self.width=128
        self.conv=nn.Sequential(nn.Conv2d(2,128,3,padding=1),nn.GroupNorm(1,128,eps=1e-5))
        self.blocks=nn.ModuleList(memory_chunks(GridNetV2Block(128,1,1,512,192,n_head=4,
            approx_qk_dim=512,activation='prelu',eps=1e-5)) for _ in range(6))
        self.deconv=nn.ConvTranspose2d(128,8,3,padding=1)
        self.context_encoder=TemporalEncoder('tcn',65,128)
        self.context_projection=nn.Conv1d(128,128,1,bias=False)
        self.count_head=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3))
        # Reference recipe uses Xavier initialization. The RF output head is
        # kept nonzero so every block receives waveform gradients immediately.
        for name,param in self.named_parameters():
            if param.ndim>1 and not name.endswith(('.gamma','.beta')):nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):nn.init.zeros_(param)

    def forward(self,z,context_features,crop_start):
        if z.ndim!=3 or not z.is_complex() or z.shape[1]!=512:raise ValueError('Two-sided512 complex STFT required')
        if context_features.ndim!=3 or context_features.shape[1:]!=(65,255):raise ValueError('Original long-context shape required')
        if crop_start.shape!=(z.shape[0],):raise ValueError('Crop positions required')
        scale=z.abs().square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-8)
        # Recurrent frequency ordering runs from negative to positive frequency.
        normalized=torch.fft.fftshift(z/scale,dim=1)
        x=torch.stack((normalized.real,normalized.imag),dim=1).transpose(2,3)
        context=context_features.mean(-1,keepdim=True).expand_as(context_features)
        encoded=self.context_encoder(context)
        x=self.conv(x)+.1*self.context_projection(encoded).mean(-1)[:,:,None,None]
        for block in self.blocks:
            x=checkpoint(block,x,use_reentrant=False) if self.training and torch.is_grad_enabled() else block(x)
        raw=self.deconv(x).float().transpose(2,3)
        streams=torch.complex(raw[:,0::2],raw[:,1::2])
        streams=torch.fft.ifftshift(streams,dim=2)*scale[:,None]
        estimates=streams+(z-streams.sum(1))[:,None]/4
        return dict(estimates=estimates,count_logits=self.count_head(encoded.mean(-1)))


def build():
    torch.manual_seed(0)
    return RFGridNet()
