"""Convolutional dependency intervals, excluding known global/statistical paths."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/rfuav_dense_gated_20261008'))
from models import build
from drone_rf.model import ComplexSeparator


def block(x):
    for _ in range(2):
        x=np.stack(([x[max(0,j-1):min(len(x),j+2),0].min() for j in range(len(x))],
                    [x[max(0,j-1):min(len(x),j+2),1].max() for j in range(len(x))]),-1)
    return x


def intervals(n):
    x=np.tile(np.arange(n)[:,None],(1,2));skips=[]
    for i in range(5):
        if i:x=np.stack((x[:len(x)//2*2:2,0],x[1:len(x)//2*2:2,1]),-1)
        x=block(x);skips.append(x.copy())
    for skip in reversed(skips[:-1]):
        at=(np.arange(len(skip))+.5)*len(x)/len(skip)-.5
        lo=np.floor(at).astype(int).clip(0,len(x)-1);hi=np.ceil(at).astype(int).clip(0,len(x)-1)
        x=np.stack((np.minimum(x[lo,0],skip[:,0]),np.maximum(x[hi,1],skip[:,1])),-1);x=block(x)
    return x


def autograd_support(n,indices):
    # Positive scalar-channel operators verify only the time-axis geometry.
    # They are not a trained/downsized separator experiment.
    source=torch.zeros(1,1,n,dtype=torch.float64,requires_grad=True);x=source;skips=[]
    kernel=torch.ones(1,1,3,dtype=torch.float64)
    for i in range(5):
        if i:x=F.avg_pool1d(x,2)
        for _ in range(2):x=F.conv1d(x,kernel,padding=1)
        skips.append(x)
    for skip in reversed(skips[:-1]):
        x=F.interpolate(x,size=skip.shape[-1],mode='linear',align_corners=False)+skip
        for _ in range(2):x=F.conv1d(x,kernel,padding=1)
    rows=[]
    for index in indices:
        gradient=torch.autograd.grad(x[0,0,index],source,retain_graph=True)[0][0,0]
        nonzero=torch.where(gradient!=0)[0]
        rows.append(dict(index=index,first=int(nonzero.min()),last=int(nonzero.max())))
    return rows


def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    model=build('unet_mean');assert len(model.down)==5 and len(model.up)==4
    for layer in list(model.down)+list(model.up):
        conv=[m for m in layer.modules() if isinstance(m,torch.nn.Conv2d)]
        assert len(conv)==2 and all(m.kernel_size==(3,3) and m.stride==(1,1) and m.padding==(1,1) and m.dilation==(1,1) for m in conv)
    assert model.context_mode=='mean'
    rows=[]
    for n in (500,1498):
        x=intervals(n);span=x[:,1]-x[:,0]+1
        indices=sorted(set([0,n//4,n//2,3*n//4,n-1,int(span.argmax())]))
        checks=autograd_support(n,indices)
        assert all([r['first'],r['last']]==x[r['index']].tolist() for r in checks)
        rows.append(dict(frames=n,maximum_span_frames=int(span.max()),central_span_frames=int(span[n//2]),
            max_waveform_support_us=float(((span.max()-1)*128+512)/100),
            selected_position_autograd_checks=checks))
    code=Path(inspect.getsourcefile(ComplexSeparator))
    result=dict(status='PASS',results=rows,full_model_parameters=sum(p.numel() for p in model.parameters()),
        backbone_source=str(code.relative_to(ROOT)),backbone_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),
        excludes=['GroupNorm moments across frequency/time','Input RMS normalization','Long-context branch'],
        disclaimer='Not the complete model receptive field or evidence of the cause of a particular error; STFT edge reflection ignored in sample-span conversion')
    out=ROOT/'reports/2026-10-13/LOCAL_CONVOLUTION_SUPPORT.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
