"""Full TF-GridNet continuation with broader approved TRAIN mixture coverage."""
import copy
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tfgridnet_rf_20261011'))
import fit
from model import build,ChunkedLSTM
w,study,worker=fit.w,fit.convergence,fit.worker
PARENT=ROOT/'local/tfgridnet_continuation_20261011_v1'
OUT=ROOT/'local/tfgridnet_diversity_20261011_v1'


def state(status,**fields):w.write(OUT/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**fields))


def coverage(data,indices):
    ratios=[];categories=set();packs=set();bands=set()
    for index in indices:
        raw=data[index];powers=np.mean(np.abs(raw['references'].astype(np.complex128))**2,axis=-1)
        count=int(raw['construction_count']);powers=powers[:count]
        ratios.extend((10*np.log10(powers/(powers.sum()-powers))).tolist())
        for i in data.rows[index]['indices'][:count]:
            clip=data.library.clips[int(i)];categories.add(clip['category']);packs.add(clip['pack_id']);bands.add(clip['common_center_hz'])
    return dict(categories=sorted(categories),record_groups=sorted(packs),common_centers_hz=sorted(bands),
        sources=len(ratios),minimum_power_ratio_db=min(ratios),maximum_power_ratio_db=max(ratios),
        below_minus25db=sum(x<-25 for x in ratios),below_minus15db=sum(x<-15 for x in ratios))


@torch.no_grad()
def evaluate(net,data,indices):
    net.eval();rows=[]
    for case,index in enumerate(indices):
        state('EVALUATING_TRAIN_PROBE',case=case+1,total=len(indices))
        item=worker.fit.base.batch([data[index]]);out,logits=worker.predict(net,item)
        m=study.waveform_metrics(out,item['references'],item['active'],item['mixture']);active=item['active'][0]
        powers=m['reference_power'][0][active].tolist();nm=m['nmse'][0][active].tolist();si=m['si_sdr'][0][active].tolist()
        assert np.isfinite(nm+si).all() and float(m['sum_relative_error'][0])<1e-9
        count=int(item['construction_count']);clips=[data.library.clips[int(j)] for j in data.rows[index]['indices'][:count]]
        rows.append(dict(index=index,count=count,categories=[c['category'] for c in clips],pack_ids=[c['pack_id'] for c in clips],
            reference_power=powers,nmse=nm,si_sdr=si,weakest_index=int(np.argmin(powers)),
            assignment=m['assignment'][0].tolist(),sum_relative_error=float(m['sum_relative_error'][0]),
            predicted_count=int(logits.argmax(-1)[0])+1))
    grouped=[]
    for count in (2,3):
        group=[r for r in rows if r['count']==count];assert group
        grouped.append(dict(count=count,cases=len(group),mean_nmse=float(np.mean([v for r in group for v in r['nmse']])),
            mean_si_sdr=float(np.mean([v for r in group for v in r['si_sdr']])),
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in group])),
            max_source_nmse=max(v for r in group for v in r['nmse'])))
    return dict(rows=rows,by_count=grouped)


def main():
    OUT.mkdir(exist_ok=False)
    assert w.read(ROOT/'reports/2026-10-11/TFGRIDNET_CONTINUATION_AUDIT.json')['status']=='PASS'
    assert w.read(ROOT/'local/tfgridnet_outside_fit_20261011_v1/COMPLETE.json')['status']=='COMPLETE'
    p=copy.deepcopy(w.read(PARENT/'PROTOCOL.json'));fit.verify(PARENT,p)
    chosen=w.read(PARENT/'COMPLETE.json')['selected'];assert chosen['step']==64
    checkpoint=Path(chosen['checkpoint']);assert w.digest(checkpoint)==chosen['checkpoint_sha256']
    data=worker.NativeMixtures(p['preparation'],'train_pack',1)
    two=[i for i in range(1200) if int(data.rows[i]['count'])==2][:64]
    three=[i for i in range(1200) if int(data.rows[i]['count'])==3][:64]
    assert len(two)==len(three)==64
    rng=np.random.default_rng(0);rng.shuffle(two);rng.shuffle(three)
    batches=[two[2*i:2*i+2]+three[2*i:2*i+2] for i in range(32)]
    train=[i for b in batches for i in b]
    probe=[i for i in range(len(data)-48,len(data)) if int(data.rows[i]['count']) in (2,3)]
    assert len(set(train))==128 and not (set(train)&set(probe))
    p['source_sha256'][str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    for rel in p['source_sha256']:
        target=OUT/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    for path in [checkpoint,PARENT/'PROTOCOL.json',ROOT/'reports/2026-10-11/TFGRIDNET_DIVERSITY_PLAN_KO.md']:
        p['pinned_files'][str(path)]=w.digest(path)
    p.update(start_update=64,max_updates=96,additional_updates=32,train_indices=train,batches=batches,probe_indices=probe,
        parent_checkpoint=str(checkpoint),parent_checkpoint_sha256=w.digest(checkpoint),maximum_training_seconds=2700,
        observations=[0,16,32],selection='only final vs initial probe; NMSE2/3 down, SI2/3 up, weakest2/3 no worse',
        stop_rule='single128-mixture pass then review; no automatic extension; immediate stop on numerical error',
        success='joint improvement on disjoint TRAIN mixture indices only; not DEV or independent-recording evidence',
        control='reuse same checkpoint initial probe; no separate control training; no pure causal diversity claim',
        source_rows_sha256=data.rows_hash,registered_at=time.time())
    w.write(OUT/'PROTOCOL.json',p);ph=w.digest(OUT/'PROTOCOL.json');fit.verify(OUT,p)
    state('TRAIN_METADATA_AND_POWER_COVERAGE')
    w.write(OUT/'COVERAGE.json',dict(training=coverage(data,train),probe=coverage(data,probe),
        training_mixtures=len(train),probe_mixtures=len(probe),indices_disjoint=True,not_independent_recordings=True))
    with worker.fit.base.LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);assert torch.cuda.is_available()
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        net=build().cuda();net.load_state_dict(saved['model'])
        for module in net.modules():
            if isinstance(module,ChunkedLSTM):module.chunk=p['lstm_chunk']
        opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],foreach=False)
        opt.load_state_dict(saved['optimizer']);assert {int(v['step']) for v in opt.state.values()}=={64}
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
        began=time.time();history=[];reason='ONE128_MIXTURE_PASS_REVIEW'
        for added in range(33):
            if added:
                net.train();opt.zero_grad(set_to_none=True);total=0.
                for case,index in enumerate(batches[added-1]):
                    state('TRAINING',completed_added_updates=added-1,working_added_update=added,total_added_updates=32,
                        total_optimizer_updates=64+added-1,case=case+1,index=index,seconds=time.time()-began)
                    item=worker.fit.base.batch([data[index]]);out,logits=worker.predict(net,item)
                    loss=study.original.prior.pit_waveform_loss(out,item['references'],item['active'],item['mixture'])['loss']
                    loss=loss+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
                    assert torch.isfinite(loss);(loss/4).backward();total+=float(loss.detach())/4
                norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True));opt.step()
            timed=time.time()-began>=2700
            if added not in (0,16,32) and not timed:continue
            point=dict(added_updates=added,total_updates=64+added,**evaluate(net,data,probe),seconds=time.time()-began)
            if added:
                assert {int(v['step']) for v in opt.state.values()}=={64+added}
                worker.atomic_torch(OUT/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),updates=64+added,
                    protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
                point.update(checkpoint_sha256=w.digest(OUT/'LAST.pt'),training_loss=total,gradient_norm_before_clip=norm)
            history.append(point);w.write(OUT/f'ADDED_{added:03d}.json',point)
            progress=dict(status='TRAIN_PROBE_DIAGNOSIS',history=history,added_updates=added,
                protocol_sha256=ph,validation_read=False,heldout_read=False,incumbent_replaced=False)
            w.write(OUT/'HISTORY.json',progress);w.write(ROOT/'reports/2026-10-11/TFGRIDNET_DIVERSITY_PROGRESS.json',progress)
            print(dict(added_updates=added,by_count=point['by_count']),flush=True)
            if timed:reason='TIME_BUDGET_REVIEW';break
        fit.verify(OUT,p)
        checks=[dict(count=a['count'],nmse=b['mean_nmse']<a['mean_nmse'],si=b['mean_si_sdr']>a['mean_si_sdr'],
            weak=b['weakest_nmse']<=a['weakest_nmse']) for a,b in zip(history[0]['by_count'],history[-1]['by_count'])]
        candidate=added==32 and all(c['nmse'] and c['si'] and c['weak'] for c in checks)
        result=dict(status='COMPLETE',reason=reason,history=history,final=point,added_updates=added,total_updates=64+added,
            checks=checks,probe_improvement_candidate=candidate,incumbent_replaced=False,protocol_sha256=ph,
            validation_read=False,heldout_read=False,seconds=time.time()-began)
        w.write(OUT/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-11/TFGRIDNET_DIVERSITY_RESULT.json',result)
        state('COMPLETED',added_updates=added,total_updates=64+added,probe_improvement_candidate=candidate)


if __name__=='__main__':
    try:main()
    except Exception:
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
