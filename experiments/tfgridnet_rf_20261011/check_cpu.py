"""Full reference capacity with short synthetic tensors; no recorded I/Q."""
import copy
import json
import os
from pathlib import Path
import time
import traceback
import torch
from model import build,memory_chunks,GridNetV2Block,ROOT
from drone_rf.waveform import analyze,synthesize

OUT=ROOT/'local/tfgridnet_cpu_20261011_v2'


def write(path,d):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,indent=2)+'\n')


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    write(OUT/'STATE.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    torch.manual_seed(1729)
    direct=GridNetV2Block(128,1,1,512,192,n_head=4,approx_qk_dim=512).eval()
    chunked=memory_chunks(copy.deepcopy(direct)).eval()
    x=torch.randn(1,128,17,512)
    with torch.no_grad():a=direct(x);b=chunked(x)
    # v1 retained a near-zero elementwise FP32 tolerance failure (4/1,114,112
    # values, max2.74e-6). Independently verify in float64 before classifying it
    # as backend batching roundoff; report the FP32 difference explicitly.
    block_diff=float((a-b).abs().max())
    block_relative=float((a-b).norm()/a.norm())
    assert block_relative<1e-6
    direct.double();chunked.double()
    with torch.no_grad():da=direct(x.double());db=chunked(x.double())
    torch.testing.assert_close(da,db,rtol=1e-10,atol=1e-11)
    double_diff=float((da-db).abs().max());del direct,chunked,x,a,b,da,db
    net=build().train();parameters=sum(p.numel() for p in net.parameters())
    mix=torch.randn(1,2048,dtype=torch.complex64)
    z=analyze(mix);roundtrip=synthesize(z,2048)
    torch.testing.assert_close(roundtrip,mix,rtol=1e-5,atol=2e-6)
    result=net(z,torch.randn(1,65,255),torch.tensor([0]))
    out=synthesize(result['estimates'],2048)
    torch.testing.assert_close(out.sum(1),mix,rtol=1e-5,atol=5e-6)
    target=torch.randn_like(out);loss=(out-target).abs().square().mean()+result['count_logits'].square().mean()
    loss.backward()
    grads=[]
    for block in net.blocks:
        values=[p.grad for p in block.parameters()]
        assert all(v is not None and torch.isfinite(v).all() for v in values)
        norm=float(torch.stack([v.double().square().sum() for v in values]).sum().sqrt())
        assert norm>0;grads.append(norm)
    receipt=dict(status='PASS',parameters=parameters,blocks=6,width=128,lstm_hidden=192,heads=4,
        two_sided_bins=512,synthetic_samples=2048,training_crop_samples_not_yet_checked=63872,
        upstream_chunk_forward_max_abs=block_diff,upstream_chunk_forward_relative_l2=block_relative,
        upstream_chunk_float64_max_abs=double_diff,block_gradient_norms=grads,
        sum_relative_error=float((out.sum(1)-mix).abs().square().sum()/mix.abs().square().sum()),
        optimizer_updates=0,recorded_iq_reads=0,gpu_initialized=False,time=time.time())
    write(OUT/'COMPLETE.json',receipt);write(ROOT/'reports/2026-10-11/TFGRIDNET_CPU_CHECK.json',receipt)
    write(OUT/'STATE.json',dict(status='COMPLETED',pid=None,time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
