"""Bounded accuracy/profile/throughput check after the full RF fit finishes."""
import contextlib
import fcntl
import gc
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tfgridnet_rf_20261011'))
import fit
from model import build,ChunkedLSTM
w=fit.w;study=fit.convergence;worker=fit.worker
RUN=ROOT/'local/tfgridnet_fit_20261011_v1'
OUT=ROOT/'local/tfgridnet_throughput_20261011_v1'


def state(status,**values):w.write(OUT/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**values))


def amp(bf16):return torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16)


@torch.no_grad()
def accuracy(net,items,bf16):
    net.eval();outputs=[];rows=[]
    for item in items:
        with amp(bf16):out,_=worker.predict(net,item)
        out=out.cpu();batch={k:v.cpu() for k,v in item.items()};outputs.append(out)
        m=study.waveform_metrics(out,batch['references'],batch['active'],batch['mixture'])
        assert float(m['sum_relative_error'][0])<1e-9 and torch.isfinite(out).all()
        rows.append(dict(nmse=m['nmse'][batch['active']].tolist(),si_sdr=m['si_sdr'][batch['active']].tolist(),
                         assignment=m['assignment'][0].tolist()))
    return outputs,rows


def train_step(net,opt,item,bf16,capture=False):
    net.train();opt.zero_grad(set_to_none=True)
    with amp(bf16):
        out,logits=worker.predict(net,item)
        loss=study.original.prior.pit_waveform_loss(out,item['references'],item['active'],item['mixture'])['loss']
        loss=loss+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
    assert torch.isfinite(loss);loss.backward()
    gradient=None
    if capture:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        gradient=torch.cat([p.grad.detach().flatten().cpu() for p in net.parameters()])
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
    return gradient


def main():
    OUT.mkdir(exist_ok=False)
    plan=ROOT/'reports/2026-10-11/TFGRIDNET_THROUGHPUT_PLAN_KO.md'
    source_sha=w.digest(Path(__file__));plan_sha=w.digest(plan)
    w.write(OUT/'REGISTRATION.json',dict(source_sha256=source_sha,plan_sha256=plan_sha,time=time.time(),
        configurations=['fp32_chunk32','fp32_chunk64','bf16_chunk32'],warmups=1,measured_repeats=3,
        validation_read=False,heldout_read=False))
    began=time.time()
    while not (RUN/'COMPLETE.json').exists():
        if (RUN/'FAILURE.json').exists():raise RuntimeError('Fit failed')
        if time.time()-began>3600:raise TimeoutError('Fit did not finish')
        state('WAITING_FIT');time.sleep(15)
    with worker.fit.base.LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
        assert torch.cuda.is_available();torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        p=w.read(RUN/'PROTOCOL.json');fit.verify(RUN,p)
        checkpoint_sha=w.digest(RUN/'LAST.pt');saved=torch.load(RUN/'LAST.pt',map_location='cpu',weights_only=False)
        weights=saved['model'];del saved
        torch.cuda.synchronize();t=time.perf_counter()
        data=worker.NativeMixtures(p['preparation'],'train_pack',1)
        items=[worker.fit.base.batch([data[i]]) for i in p['train_indices']]
        torch.cuda.synchronize();preparation_seconds=time.perf_counter()-t
        settings=[('fp32_chunk32',32,False),('fp32_chunk64',64,False),('bf16_chunk32',32,True)]
        results=[];baseline_outputs=baseline_rows=baseline_gradient=None
        for name,chunk,bf16 in settings:
            state('NUMERICAL_CHECK',configuration=name)
            net=opt=None
            try:
                if bf16 and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unsupported')
                net=build().cuda();net.load_state_dict(weights)
                for module in net.modules():
                    if isinstance(module,ChunkedLSTM):module.chunk=chunk
                assert sum(t.numel() for t in net.parameters())==p['parameters']
                outputs,rows=accuracy(net,items,bf16)
                opt=torch.optim.AdamW(net.parameters(),lr=0.,weight_decay=p['weight_decay'],foreach=False)
                gradient=train_step(net,opt,items[2],bf16,capture=True);calls=1
                if baseline_outputs is None:
                    baseline_outputs,baseline_rows,baseline_gradient=outputs,rows,gradient
                    published=w.read(RUN/'COMPLETE.json')['final']['rows']
                    for a,b in zip(rows,published):
                        np.testing.assert_allclose(a['nmse'],b['nmse'],rtol=1e-5,atol=1e-6)
                        np.testing.assert_allclose(a['si_sdr'],b['si_sdr'],rtol=1e-5,atol=1e-5)
                    errors=dict(prediction_relative_mse=0.,nmse_max_abs=0.,si_max_abs_db=0.,gradient_relative_l2=0.,gradient_cosine=1.)
                else:
                    a=baseline_gradient.double();b=gradient.double()
                    errors=dict(prediction_relative_mse=max(float((x-y).abs().square().sum()/x.abs().square().sum()) for x,y in zip(baseline_outputs,outputs)),
                        nmse_max_abs=max(abs(x-y) for a,b in zip(baseline_rows,rows) for x,y in zip(a['nmse'],b['nmse'])),
                        si_max_abs_db=max(abs(x-y) for a,b in zip(baseline_rows,rows) for x,y in zip(a['si_sdr'],b['si_sdr'])),
                        gradient_relative_l2=float((a-b).norm()/a.norm()),gradient_cosine=float(torch.dot(a,b)/(a.norm()*b.norm())))
                    del a,b
                limits=(1e-4,5e-4,.05,.02,.999) if bf16 else (1e-10,1e-5,1e-3,1e-4,.999999)
                numerical_pass=all(errors[k]<=v for k,v in zip(('prediction_relative_mse','nmse_max_abs','si_max_abs_db','gradient_relative_l2'),limits[:4])) and errors['gradient_cosine']>=limits[4]
                # First warmup above includes allocation and first optimizer state.
                if name=='fp32_chunk32':
                    state('PROFILING_BASELINE')
                    activities=[torch.profiler.ProfilerActivity.CPU]
                    if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False) as profiler:
                        train_step(net,opt,items[2],False);torch.cuda.synchronize();calls+=1
                    key='self_cuda_time_total' if len(activities)>1 else 'self_cpu_time_total'
                    (OUT/'PROFILE.txt').write_text(profiler.key_averages().table(sort_by=key,row_limit=25))
                state('BENCHMARKING',configuration=name,numerical_pass=numerical_pass)
                torch.cuda.reset_peak_memory_stats();times=[]
                for repetition in range(3):
                    torch.cuda.synchronize();t=time.perf_counter()
                    train_step(net,opt,items[2],bf16);torch.cuda.synchronize()
                    times.append(time.perf_counter()-t);calls+=1
                unchanged=all(torch.equal(value.cpu(),weights[key]) for key,value in net.state_dict().items());assert unchanged
                results.append(dict(configuration=name,numerical_pass=numerical_pass,errors=errors,accuracy_rows=rows,
                    synchronized_seconds=times,median_seconds=statistics.median(times),peak_memory_bytes=torch.cuda.max_memory_allocated(),
                    zero_lr_optimizer_calls=calls,model_parameters_unchanged=unchanged))
                del outputs,rows,gradient
            except RuntimeError as e:
                if name=='fp32_chunk32':raise
                results.append(dict(configuration=name,status='FAILED',error=str(e),traceback=traceback.format_exc()))
            finally:
                del net,opt;gc.collect();torch.cuda.empty_cache()
            w.write(OUT/'PARTIAL.json',results)
        for row in results:
            if 'median_seconds' in row:
                row['speedup']=results[0]['median_seconds']/row['median_seconds']
                row['qualifies_for_followup']=row['numerical_pass'] and row['speedup']>=1.1 and row['configuration']!='fp32_chunk32'
        assert w.digest(Path(__file__))==source_sha and w.digest(plan)==plan_sha and w.digest(RUN/'LAST.pt')==checkpoint_sha
        fit.verify(RUN,p)
        result=dict(status='COMPLETE',results=results,preparation_and_transfer_seconds=preparation_seconds,
            torch_version=torch.__version__,cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            parameters=p['parameters'],input_shape=list(items[2]['mixture'].shape),checkpoint_sha256=checkpoint_sha,
            registration_sha256=w.digest(OUT/'REGISTRATION.json'),retained_parameter_updates=0,
            validation_read=False,heldout_read=False,automatic_training_switch=False,time=time.time())
        w.write(OUT/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-11/TFGRIDNET_THROUGHPUT_RESULT.json',result)
        state('COMPLETED')


if __name__=='__main__':
    try:main()
    except Exception:
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
