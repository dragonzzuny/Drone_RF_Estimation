"""Full-resolution GPU contract check; discard all diagnostic updates."""
import fcntl
import os
from pathlib import Path
import signal
import time
import traceback
import torch
import architecture as model

ROOT,w,worker=model.ROOT,model.w,model.worker
OUT=ROOT/'local/ordered_branch_preflight_20261011_v1'
PLAN=ROOT/'reports/2026-10-11/ORDERED_BRANCH_PREFLIGHT_PLAN_KO.md'


def norm(net,prefix):
    ps=[p for n,p in net.named_parameters() if n==prefix or n.startswith(prefix+'.')]
    assert ps and all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps)
    return sum(float(p.grad.double().square().sum()) for p in ps)**.5


def main():
    OUT.mkdir(exist_ok=False)
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    check=w.read(ROOT/'reports/2026-10-11/ORDERED_BRANCH_CPU_CHECK.json')
    assert check['status']=='PASS'
    sources=dict(old['source_sha256'])
    for p in (Path(__file__),Path(model.__file__),Path(__file__).with_name('check_cpu.py')):
        sources[str(p.relative_to(ROOT))]=w.digest(p)
    for rel,h in sources.items(): assert w.digest(ROOT/rel)==h,rel
    p=dict(parent_checkpoint=old['parent_checkpoint'],parent_checkpoint_sha256=old['parent_checkpoint_sha256'],
        preparation=old['preparation'],source_sha256=sources,plan_sha256=w.digest(PLAN),
        cpu_check_sha256=w.digest(ROOT/'reports/2026-10-11/ORDERED_BRANCH_CPU_CHECK.json'),registered_at=time.time())
    w.write(OUT/'PROTOCOL.json',p)
    assert w.digest(Path(p['parent_checkpoint']))==p['parent_checkpoint_sha256']
    data=worker.NativeMixtures(p['preparation'],'train_pack',3)
    indices=[next(i for i,r in enumerate(data.rows) if int(r['count'])==k) for k in (1,2,3)]
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        signal.alarm(1200)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        net=model.build(Path(p['parent_checkpoint'])).cuda().eval()
        began=time.time();torch.cuda.reset_peak_memory_stats();rows=[]
        for index in indices:
            w.write(OUT/'STATE.json',dict(status='GPU_CHECK',index=index,pid=os.getpid(),time=time.time()))
            item=worker.fit.base.batch([data[index]])
            with torch.no_grad():
                a,ac=worker.predict(net.parent,item);b,bc=worker.predict(net,item)
                assert torch.equal(a,b) and torch.equal(ac,bc)
                error=float((b.sum(1)-item['mixture']).abs().square().sum()/item['mixture'].abs().square().sum())
                assert error<1e-9
            rows.append(dict(index=index,count=int(item['construction_count'][0]),
                initial_prediction_exact=True,count_exact=True,sum_relative_error=error))
        item=worker.fit.base.batch([data[indices[0]]])
        net.train();net.parent.eval()
        def backward():
            out,logits=worker.predict(net,item)
            loss=worker.pit_waveform_loss(out,item['references'],item['active'],item['mixture'])['loss']
            assert torch.isfinite(loss);loss.backward()
            return float(loss.detach())
        zero_loss=backward()
        zero={k:norm(net,k) for k in ('gate','ordered_encoder','ordered_projection')}
        assert zero['gate']>0 and zero['ordered_encoder']==0 and zero['ordered_projection']==0
        net.zero_grad(set_to_none=True)
        with torch.no_grad():net.gate.fill_(.01)
        loss=backward()
        nonzero={k:norm(net,k) for k in ('gate','ordered_encoder','ordered_projection')}
        assert all(v>0 for v in nonzero.values())
        assert all(p.grad is None for p in net.parent.parameters())
        opt=model.optimizer(net)
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        opt.step()
        frozen=model.verify_parent(net,Path(p['parent_checkpoint']))
        for rel,h in sources.items():assert w.digest(ROOT/rel)==h
        result=dict(status='PASS',rows=rows,zero_gate_loss=zero_loss,nonzero_gate_loss=loss,
            zero_gate_gradients=zero,nonzero_gate_gradients=nonzero,frozen_tensors=frozen,
            diagnostic_optimizer_updates=1,discarded_test_weights=True,validation_read=False,heldout_read=False,
            seconds=time.time()-began,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            protocol_sha256=w.digest(OUT/'PROTOCOL.json'),source_sha256=sources)
        w.write(OUT/'COMPLETE.json',result)
        w.write(ROOT/'reports/2026-10-11/ORDERED_BRANCH_GPU_CHECK.json',result)
        w.write(OUT/'STATE.json',dict(status='PASS',pid=os.getpid(),time=time.time()))
        print({k:v for k,v in result.items() if k!='source_sha256'},flush=True)


if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('20minute preflight cap')))
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
