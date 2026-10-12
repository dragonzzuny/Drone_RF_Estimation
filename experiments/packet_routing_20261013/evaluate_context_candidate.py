"""Conditional final-checkpoint follow-up; registration is separate from prep.

Evaluate all five pre-existing full contexts, the known packet, and C630.
The candidate always receives mixture-only neighbours. The packet receiver and
all scored target waveforms remain unchanged. No performance-based case choice.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
LOCK=ROOT.parent/'uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock'
OLD=ROOT.parent/'uav_analysis/rf_detection/separator_detector_20261012'


def read(p):return json.loads(p.read_text())
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)


def attach(item,mixture,start,device):
    import numpy as np
    import torch
    from chunk_evaluation import neighbour_start
    from sequence_data import WINDOW,LENGTH
    assert mixture.shape==(LENGTH,)
    first=neighbour_start(start);observed=mixture[first:first+3*WINDOW].copy()
    assert np.array_equal(observed[start-first:start-first+WINDOW],item['mixture'][0].cpu().numpy())
    item['long_mixture']=torch.from_numpy(observed)[None].to(device)
    item['long_start']=torch.tensor([first],device=device,dtype=torch.long)
    return item


def add_native_context(data,index,raw,device):
    import numpy as np
    import torch
    from sequence_data import WINDOW,component_gains
    from chunk_evaluation import neighbour_start
    from study import batch
    row=data.rows[index];k=int(row['count']);start=int(raw['crop_start']);first=neighbour_start(start)
    ids=row['indices'][:k];clips=[data.library.clips[int(j)] for j in ids]
    assert all(c['role']=='validation_pack' for c in clips)
    gains=component_gains([c['mean_power'] for c in clips],row['levels'][:k],row['phases'][:k])
    parts=np.zeros((3,3*WINDOW),np.complex64)
    for j,(source,gain) in enumerate(zip(ids,gains)):
        parts[j]=(data.library._array(int(source))[first:first+3*WINDOW]*gain).astype(np.complex64)
    observation=parts.sum(0);offset=start-first
    assert np.array_equal(observation[offset:offset+WINDOW],raw['mixture'])
    item=batch(raw,device);item['long_mixture']=torch.from_numpy(observation)[None].to(device)
    item['long_start']=torch.tensor([first],device=device,dtype=torch.long)
    return item


class Evaluation:
    def __init__(self,out):self.out=out
    def state(self,status,**kw):write(self.out/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))

    def register(self,training,plan):
        training=training.resolve();p=read(training/'PROTOCOL.json')
        assert p['epochs']==3 and p['architecture'] in ('chunk_temporal','dual_context')
        assert p['optimization_scope']=='added','This evaluation verifies unchanged parent count predictions'
        assert plan.is_file() and not (self.out/'PLAN.json').exists()
        self.out.mkdir(parents=True,exist_ok=True)
        sources=dict(p['source_sha256'])
        for path in (Path(__file__),HERE/'full_context_eval.py',HERE/'check_packets.py',HERE/'test_followup_context.py',plan):
            sources[str(path.resolve().relative_to(ROOT))]=sha(path)
        parent_full=ROOT/'local/full_context_parent_20261013_v1'
        parent_c=ROOT/'local/b99_on_c_20261013_v1'
        assert all(read(d/'STATE.json')['status']=='COMPLETE_AUDITED' for d in (parent_full,parent_c))
        for existing in (read(parent_c/'PROTOCOL.json'),read(parent_full/'PLAN.json')):
            for name,digest in existing['source_sha256'].items():
                if name in sources:assert sources[name]==digest,name
                sources[name]=digest
        files={str(q):sha(q) for q in (training/'PROTOCOL.json',parent_full/'PLAN.json',parent_full/'RESULT.json',
            parent_c/'PROTOCOL.json',parent_c/'RESULT.json',OLD/'raw_diagnosis/SEPARATION_PLAN.json',
            OLD/'receiver_pilot_v1/PLAN.json')}
        registration=dict(registered=time.time(),training=str(training),architecture=p['architecture'],epoch=3,
            source_sha256=sources,pinned_files=files,full_indices=[0,7,14,15,20],
            cases='All five pre-existing equal-power full contexts; one exposed packet; all original C630',
            evaluation_context='3 mixture-only neighbouring windows; unchanged primary reference targets',
            no_training=True,no_crc_or_threshold_tuning=True,no_heldout_access=True,
            final_epoch_fixed_before_scores=True,independent_final_test=False)
        write(self.out/'PLAN.json',registration)
        for name,digest in sources.items():
            assert sha(ROOT/name)==digest,name
            dest=self.out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        self.state('REGISTERED_NOT_RUNNING')

    def verify(self,p):
        for name,h in p['source_sha256'].items():
            assert sha(ROOT/name)==h and sha(self.out/'source_snapshot'/name)==h,name
        for name,h in p['pinned_files'].items():assert sha(name)==h,name

    def full(self,net,predict):
        import numpy as np
        import torch
        from sequence_data import Sequences,ROLES
        from full_context_eval import evaluate
        data=Sequences(ROLES[1]);baseline=read(ROOT/'local/full_context_parent_20261013_v1/RESULT.json')
        assert [v['index'] for v in baseline['cases']]==[0,7,14,15,20]
        results=[]
        for i,base in enumerate(baseline['cases']):
            index=base['index'];raw=data.rows[index];k=int(raw['count']);refs=data.components(index,True)
            features=np.array(data.features[index])
            def progress(done,total):self.state('FULL_CONTEXT',case=i+1,total_cases=5,windows=done,total_windows=total)
            output,result=evaluate(net,predict,refs,features,k,'cuda',progress,attach_neighbours=attach)
            assert result['metrics']['reference_power']==base['metrics']['reference_power']
            assert result['metrics']['predicted_count']==base['metrics']['predicted_count']
            path=self.out/f'FULL_OUTPUT_{index:03d}.npy';np.save(path,output,allow_pickle=False)
            result.update(index=index,categories=base['categories'],nominal_levels_db=base['nominal_levels_db'],
                output_sha256=sha(path),parent_metrics=base['metrics'])
            write(self.out/f'FULL_CASE_{index:03d}.json',result);results.append(result)
            del output,refs;gc.collect();torch.cuda.empty_cache()
        return results

    def packet(self,net,predict):
        import numpy as np
        import torch
        from sequence_eval import predict_eight,merge,score
        from sequence_data import WINDOW
        from study import batch
        from drone_rf.context_data import component_gains,mixture_context_features
        from check_packets import run as receiver
        plan=read(OLD/'raw_diagnosis/SEPARATION_PLAN.json')
        first_path=OLD/'raw_diagnosis/FIRST_EVENT.npz';second_path=Path(plan['second_clip']['cache_path'])
        assert sha(first_path)==plan['first_event_hash'] and sha(second_path)==plan['second_clip']['cache_sha256']
        first=np.load(first_path)['filtered'];second=np.load(second_path)
        gains=component_gains([float(np.mean(abs(x.astype(np.complex128))**2)) for x in (first,second)],[0,0],[0,.7])
        assert np.array_equal([[g.real,g.imag] for g in gains],plan['gains_real_imag'])
        refs=np.stack([(x*g).astype(np.complex64) for x,g in zip((first,second),gains)])
        mixture=refs.sum(0);features=mixture_context_features(mixture)['context_features'];lo,hi=plan['roi']
        original=np.load(OLD/'retest_best_e079/SEPARATED_EVENT.npz')
        assert np.array_equal(refs[:,lo:hi],original['references'][:2])
        assert np.array_equal(mixture[lo:hi],original['mixture'])
        target=np.zeros((3,hi-lo),np.complex64);target[:2]=refs[:,lo:hi];ys=[]
        for j,start in enumerate(plan['starts']):
            item=batch(dict(mixture=mixture[start:start+WINDOW].copy(),context_features=features,crop_start=start,
                references=np.zeros((3,WINDOW),np.complex64),active=np.zeros(3,bool),construction_count=2),'cuda')
            item=attach(item,mixture,start,'cuda')
            _,y,logits,_=predict_eight(net,predict,item);ys.append(y[0].cpu().numpy())
            self.state('PACKET_INFERENCE',windows=j+1,total=8)
        begin=plan['starts'][0];length=plan['starts'][-1]+WINDOW-begin
        output,receipt=merge(ys,[s-begin for s in plan['starts']],length);output=output[:,lo-begin:hi-begin]
        metrics=score(torch.from_numpy(output)[None].cuda(),torch.from_numpy(target)[None].cuda(),
            torch.from_numpy(mixture[lo:hi])[None].cuda(),2,logits)
        d=self.out/'packet';d.mkdir(exist_ok=True);np.savez_compressed(d/'ASSEMBLED.npz',C8_hann=output)
        write(d/'RESULT.json',dict(status='COMPLETE',epoch=3,stages={'C8_hann':dict(metrics=metrics,overlap_matching=receipt)}))
        write(d/'VERIFICATION.json',dict(status='PASS',input_hashes_unchanged=
            sha(first_path)==plan['first_event_hash'] and sha(second_path)==plan['second_clip']['cache_sha256'],
            artifacts={n:sha(d/n) for n in ('ASSEMBLED.npz','RESULT.json')}))
        self.state('PACKET_RECEIVER');receiver(d)
        assert read(d/'PACKET_AUDIT.json')['status']=='PASS'
        return dict(metrics=metrics,receiver=read(d/'PACKET_RESULTS.json'),audit=read(d/'PACKET_AUDIT.json'))

    def c_validation(self,net,predict):
        import numpy as np
        import torch
        from sequence_data import WINDOW,LENGTH,component_gains
        from sequence_eval import predict_eight,score,summarize
        from chunk_evaluation import neighbour_start
        from study import batch
        sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
        from native_data import NativeMixtures
        old=read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')
        data=NativeMixtures(Path(old['preparation']),'validation_pack',1)
        parent=read(ROOT/'local/b99_on_c_20261013_v1/RESULT.json')
        assert data.rows_hash==parent['rows_sha256'] and len(data)==630
        rows=[]
        for i in range(len(data)):
            raw=data[i];row=data.rows[i];k=int(row['count'])
            ids=row['indices'][:k];clips=[data.library.clips[int(j)] for j in ids]
            item=add_native_context(data,i,raw,'cuda')
            _,y,logits,_=predict_eight(net,predict,item)
            result=score(y,item['references'],item['mixture'],k,logits)
            result.update(index=i,categories=[c['category'] for c in clips],nominal_levels_db=row['levels'][:k].tolist())
            for name in ('count','categories','reference_power','predicted_count'):assert result[name]==parent['rows'][i][name],(i,name)
            rows.append(result)
            if i%25==0:self.state('C_DEV',case=i+1,total=630)
        result=dict(rows=rows,rows_sha256=data.rows_hash,**summarize(rows),parent_by_count=parent['by_count'])
        write(self.out/'C_RESULT.json',result)
        return {k:v for k,v in result.items() if k!='rows'}

    def run(self):
        p=read(self.out/'PLAN.json');self.verify(p);training=Path(p['training'])
        while True:
            s=read(training/'STATE.json')
            if s['status']=='COMPLETE_AUDITED':break
            if s['status'] in ('FAILED','STOPPED_WITH_RECOVERY'):raise RuntimeError('Candidate training needs attention')
            self.state('WAITING_CANDIDATE',previous=s['status'],epoch=s.get('epoch'));time.sleep(20)
        with LOCK.open('r') as lock:
            self.state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);self.verify(p)
            import torch
            from train_chunk_candidate import load
            from chunk_temporal_candidate import predict_primary
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            e=read(training/'EPOCH_003.json');a=read(training/'AUDIT_003.json');checkpoint=training/'ACTUAL_003.pt'
            assert a['status']=='PASS' and a['event_sha256']==sha(training/'EPOCH_003.json')
            assert sha(checkpoint)==e['checkpoint_sha256']
            net,saved=load(checkpoint,'cuda',p['architecture'],'added');del saved;net.eval().requires_grad_(False)
            with torch.no_grad():
                full=self.full(net,predict_primary);packet=self.packet(net,predict_primary);c=self.c_validation(net,predict_primary)
            self.verify(p);assert sha(checkpoint)==e['checkpoint_sha256']
            result=dict(status='VERIFIED',architecture=p['architecture'],epoch=3,checkpoint_sha256=e['checkpoint_sha256'],
                full_context=full,packet=packet,c_validation=c,plan_sha256=sha(self.out/'PLAN.json'),
                optimizer_updates_in_evaluation=0,heldout_access=False,independent_final_test=False)
            write(self.out/'RESULT.json',result)
            write(self.out/'AUDIT.json',dict(status='PASS',result_sha256=sha(self.out/'RESULT.json'),
                sources_and_inputs_unchanged=True,count_predictions_unchanged=True))
            write(ROOT/'reports/2026-10-13'/f'{p["architecture"].upper()}_FOLLOWUP.json',result)
            self.state('COMPLETE_AUDITED')


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('action',choices=('register','run'));q.add_argument('--output',type=Path,required=True)
    q.add_argument('--training',type=Path);q.add_argument('--plan',type=Path);args=q.parse_args();worker=Evaluation(args.output)
    try:
        if args.action=='register':worker.register(args.training,args.plan)
        else:
            with (args.output/'.run.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);worker.run()
    except BaseException:
        args.output.mkdir(parents=True,exist_ok=True);write(args.output/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        worker.state('FAILED');raise
