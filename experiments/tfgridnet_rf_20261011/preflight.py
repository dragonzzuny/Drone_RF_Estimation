"""One full-length GPU optimizer step; temporary weights are discarded."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from model import build,ROOT
sys.path.insert(0,str(ROOT/'experiments/source_interaction_20261010'))
import train_comparison as worker
from drone_rf.waveform import predict
from drone_rf.losses import pit_waveform_loss
w=worker.watch


def run(root):
    root.mkdir(exist_ok=False)
    check=ROOT/'reports/2026-10-11/TFGRIDNET_CPU_CHECK.json'
    assert w.read(check)['status']=='PASS'
    p=w.read(ROOT/'local/convergence_20261011_v1/PROTOCOL.json')
    sources=dict(p['source_sha256'])
    sources.update({str(path.relative_to(ROOT)):w.digest(path) for path in Path(__file__).parent.rglob('*') if path.is_file() and '__pycache__' not in str(path)})
    protocol=dict(source_sha256=sources,cpu_check_sha256=w.digest(check),
        preparation=p['preparation'],preparation_sha256=w.digest(Path(p['preparation'])/'PREPARATION.json'),
        train_index=2,optimizer_steps=1,temporary_weights_discarded=True,
        original_capacity='6 blocks/D128/BiLSTM192/4heads, RF512 bins',
        crop_samples=63872,heldout_read=False,validation_read=False,registered_at=time.time())
    w.write(root/'PROTOCOL.json',protocol)
    w.write(root/'STATE.json',dict(status='WAITING_GPU_LOCK',pid=os.getpid(),time=time.time()))
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert torch.cuda.is_available()
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
        data=worker.NativeMixtures(p['preparation'],'train_pack',1)
        item=worker.fit.base.batch([data[2]])
        net=build().cuda().train()
        parameters=sum(t.numel() for t in net.parameters())
        assert parameters==w.read(check)['parameters']
        opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-4,foreach=False)
        start=time.time();w.write(root/'STATE.json',dict(status='GPU_FORWARD',pid=os.getpid(),time=time.time()))
        outputs,logits=predict(net,item)
        sumerr=float(((outputs.sum(1)-item['mixture']).abs().square().mean()/item['mixture'].abs().square().mean()).detach())
        assert torch.isfinite(outputs).all() and sumerr<1e-9
        loss=pit_waveform_loss(outputs,item['references'],item['active'],item['mixture'])['loss']
        loss=loss+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
        assert torch.isfinite(loss)
        w.write(root/'STATE.json',dict(status='GPU_BACKWARD',pid=os.getpid(),loss=float(loss.detach()),time=time.time()))
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
        blocknorms=[sum(float(x.grad.double().square().sum()) for x in block.parameters() if x.grad is not None)**.5 for block in net.blocks]
        assert all(v>0 for v in blocknorms)
        opt.step();torch.cuda.synchronize()
        assert all(torch.isfinite(t).all() for t in net.parameters())
        assert {int(s['step']) for s in opt.state.values()}=={1}
        for path,digest in sources.items():assert w.digest(ROOT/path)==digest
        result=dict(status='PASS',parameters=parameters,optimizer_steps=1,train_index=2,crop_samples=63872,
            loss_before_step=float(loss.detach()),gradient_norm_before_clip=norm,block_gradient_norms_after_clip=blocknorms,
            sum_relative_error=sumerr,peak_memory_bytes=torch.cuda.max_memory_allocated(),
            seconds=time.time()-start,temporary_weights_discarded=True,validation_read=False,heldout_read=False,
            protocol_sha256=w.digest(root/'PROTOCOL.json'),time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-11/TFGRIDNET_GPU_PREFLIGHT.json',result)
        w.write(root/'STATE.json',dict(status='COMPLETED',pid=None,time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    try:run(args.run.resolve())
    except Exception:
        w.write(args.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        w.write(args.run/'STATE.json',dict(status='FAILED',pid=None,time=time.time()));raise
