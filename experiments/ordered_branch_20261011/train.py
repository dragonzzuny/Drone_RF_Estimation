"""One full schedule3 epoch; frozen incumbent plus learnable temporal branch."""
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import architecture as model
import evaluation
from preflight import norm

ROOT,w,worker=model.ROOT,model.w,model.worker
OUT=ROOT/'local/ordered_branch_20261011_v1'
PUBLIC=ROOT/'reports/2026-10-11'
PLAN=PUBLIC/'ORDERED_BRANCH_PLAN_KO.md'


def state(status,**kw):
    payload=dict(status=status,pid=os.getpid(),time=time.time(),**kw)
    w.write(OUT/'STATE.json',payload)
    text='# 별도 시간 변화 경로: 실제 저장 상태\n\n'
    text+=f"상태: {status}. epoch {kw.get('epoch',0)}, 업데이트 {kw.get('updates',0)}/75.\n\n"
    if 'case' in kw:text+=f"평가 {kw['case']}/630 (단일·4위상 함께).\n\n"
    if 'seconds' in kw:text+=f"학습 경과 {kw['seconds']:.1f}초.\n\n"
    text+='이 상태 파일은 자동 채팅 보고가 아니다. 실제 완료 지표는 RESULT와 CPU AUDIT에서 확인한다.\n'
    w.write(PUBLIC/'ORDERED_BRANCH_LIVE_KO.md',text)


def verify(p):
    for rel,h in p['source_sha256'].items():
        assert w.digest(ROOT/rel)==h and w.digest(OUT/'source_snapshot'/rel)==h,rel
    for f,h in p['pinned_files'].items():assert w.digest(Path(f))==h,f


def register():
    OUT.mkdir(exist_ok=False)
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    checked=w.read(PUBLIC/'ORDERED_BRANCH_GPU_CHECK.json');assert checked['status']=='PASS'
    sources=dict(checked['source_sha256'])
    for source in list(Path(__file__).parent.glob('*.py'))+[evaluation.PHASE_SOURCE]:
        sources[str(source.relative_to(ROOT))]=w.digest(source)
    for rel,h in checked['source_sha256'].items():assert w.digest(ROOT/rel)==h
    for rel,h in sources.items():
        assert w.digest(ROOT/rel)==h
        dest=OUT/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    baseline=Path(old['baseline']);four=baseline.with_name('FOUR_PHASE.json')
    train=worker.NativeMixtures(old['preparation'],'train_pack',3)
    dev=worker.NativeMixtures(old['preparation'],'validation_pack',1)
    pinned=[PLAN,Path(old['parent_checkpoint']),baseline,four,PUBLIC/'ORDERED_BRANCH_CPU_CHECK.json',
        PUBLIC/'ORDERED_BRANCH_GPU_CHECK.json',Path(old['preparation'])/'features/train_pack_003.json',
        ROOT/'local/fresh_schedule_20261010_v1/VALIDATION_001.json']
    p=dict(parent_checkpoint=old['parent_checkpoint'],parent_checkpoint_sha256=old['parent_checkpoint_sha256'],
        preparation=old['preparation'],baseline=str(baseline),four_phase_baseline=str(four),
        source_sha256=sources,pinned_files={str(f):w.digest(f) for f in pinned},
        train_rows_sha256=train.rows_hash,dev_rows_sha256=dev.rows_hash,train_schedule=3,epochs=1,
        seed=0,examples=2400,updates=75,effective_batch=32,microbatch=1,extra_shuffle=False,
        parameters=32878091,trainable_parameters=735232,frozen_parent_parameters=32142859,
        encoder_projection_lr=1e-4,gate_lr=1e-2,weight_decay=1e-4,clip_norm=1.,
        precision='FP32 TF32off',maximum_seconds=2700,
        acceptance='Both single/four-phase: NMSE2/3 lower, complexSI2/3 higher, weakestNMSE2/3 nonworse',
        heldout_read=False,independent_test=False,registered_at=time.time())
    assert w.digest(Path(p['parent_checkpoint']))==p['parent_checkpoint_sha256']
    w.write(OUT/'PROTOCOL.json',p);verify(p)
    return p,train,dev


def main():
    p,train,dev=register();ph=w.digest(OUT/'PROTOCOL.json')
    with worker.fit.base.LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);signal.alarm(2700)
        assert torch.cuda.is_available();torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        net=model.build(Path(p['parent_checkpoint'])).cuda()
        assert sum(q.numel() for q in net.parameters() if q.requires_grad)==p['trainable_parameters']
        assert sum(q.numel() for q in net.parameters())==p['parameters']
        optimizer=model.optimizer(net);torch.manual_seed(0)
        net.train();net.parent.eval();optimizer.zero_grad(set_to_none=True)
        for name,src in [('SINGLE_000.json',p['baseline']),('FOUR_PHASE_000.json',p['four_phase_baseline'])]:
            shutil.copyfile(src,OUT/name)
        worker.atomic_torch(OUT/'INITIAL.pt',dict(model=net.state_dict(),epoch=0,protocol_sha256=ph))
        losses=[];norms=[];gradients=[];counts=[0,0,0];began=time.time();torch.cuda.reset_peak_memory_stats()
        params=[q for q in net.parameters() if q.requires_grad]
        for i in range(len(train)):
            item=worker.fit.base.batch([train[i]])
            predicted,logits=worker.predict(net,item)
            loss=worker.pit_waveform_loss(predicted,item['references'],item['active'],item['mixture'])['loss']
            loss=loss+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
            assert torch.isfinite(loss);(loss/32).backward()
            losses.append(float(loss.detach()));counts[int(item['construction_count'][0])-1]+=1
            if (i+1)%32==0:
                gradients.append({k:norm(net,k) for k in ('gate','ordered_encoder','ordered_projection')})
                assert all(q.grad is None for q in net.parent.parameters())
                norms.append(float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)))
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                state('TRAINING',epoch=1,updates=len(norms),target_updates=75,examples=i+1,
                    seconds=time.time()-began,recent_training_loss=float(np.mean(losses[-32:])))
                if len(norms)%25==0:
                    worker.atomic_torch(OUT/'RECOVERY.pt',dict(model=net.state_dict(),optimizer=optimizer.state_dict(),
                        epoch=1,updates=len(norms),protocol_sha256=ph,torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all()))
        elapsed=time.time()-began
        assert len(norms)==75 and counts==[800,800,800]
        assert {int(v['step']) for v in optimizer.state.values()}=={75}
        assert model.verify_parent(net,Path(p['parent_checkpoint']))==131
        worker.atomic_torch(OUT/'ACTUAL_001.pt',dict(model=net.state_dict(),optimizer=optimizer.state_dict(),
            epoch=1,updates=75,protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        del predicted,logits,loss,item
        train_receipt=dict(epoch=1,updates=75,examples=len(train),counts=counts,mean_loss=float(np.mean(losses)),
            preclip_norms=norms,gradient_history=gradients,train_seconds=elapsed,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),protocol_sha256=ph)
        w.write(OUT/'TRAIN_001.json',train_receipt)
        start_eval=time.time()
        a,b=evaluation.evaluate(net,dev,w.read(Path(p['baseline'])),state,OUT)
        checks=dict(single=evaluation.compare(a,w.read(Path(p['baseline']))),
            four_phase=evaluation.compare(b,w.read(Path(p['four_phase_baseline']))))
        candidate=all(all(v for k,v in c.items() if k!='count') for rows in checks.values() for c in rows)
        chosen='ACTUAL_001.pt' if candidate else 'INITIAL.pt'
        shutil.copyfile(OUT/chosen,OUT/'SELECTED.pt')
        verify(p)
        result=dict(status='COMPLETE',epoch=1,updates=75,selected_epoch=1 if candidate else 0,
            candidate=candidate,checks=checks,single=a,four_phase=b,training=train_receipt,
            evaluation_seconds=time.time()-start_eval,protocol_sha256=ph,
            actual_checkpoint_sha256=w.digest(OUT/'ACTUAL_001.pt'),selected_checkpoint_sha256=w.digest(OUT/'SELECTED.pt'),
            heldout_read=False,independent_test=False,incumbent_replaced=False)
        w.write(OUT/'COMPLETE.json',result);w.write(PUBLIC/'ORDERED_BRANCH_RESULT.json',result)
        state('GPU_COMPLETE_CPU_AUDIT_PENDING',epoch=1,updates=75,selected_epoch=result['selected_epoch'])
        signal.alarm(0)
    del net,optimizer;torch.cuda.empty_cache()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    subprocess.run(['taskset','-c','12,13',sys.executable,str(Path(__file__).with_name('audit.py'))],env=env,check=True)
    state('COMPLETE_AUDITED',epoch=1,updates=75,selected_epoch=result['selected_epoch'])


if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('45minute training/evaluation cap')))
    try:main()
    except Exception:
        if OUT.exists():
            w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED')
        raise
