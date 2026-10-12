"""Frozen native e2/C8 FIT48 outputs for a separator-matched tracker."""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha,LOCK
from sequence_data import Sequences,ROLES

HERE=Path(__file__).resolve().parent
INCUMBENT=ROOT/'local/incumbent_tracking_c24_20261013_v1'
PHASE=ROOT/'local/phase_eight_20261011_v1'


def indices(data):
    groups={}
    for i,row in enumerate(data.rows):
        k=int(row['count']);cats=tuple(data.library.clips[int(j)]['category'] for j in row['indices'][:k])
        key=(k,cats,tuple(float(v) for v in row['levels'][:k]))
        group=groups.setdefault(key,[])
        if len(group)<2:group.append(i)
    assert len(groups)==24 and all(len(v)==2 for v in groups.values())
    chosen=sorted(i for group in groups.values() for i in group)
    assert len(chosen)==48 and len(set(chosen))==48
    return chosen


class Preparation:
    def __init__(self,out):self.out=out
    def state(self,status,**kw):write(self.out/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**kw))

    def register(self,plan):
        assert read(INCUMBENT/'STATE.json')['status']=='COMPLETE_AUDITED'
        assert not (self.out/'PLAN.json').exists() and plan.is_file()
        data=Sequences(ROLES[0],1);chosen=indices(data)
        phase=read(PHASE/'PROTOCOL.json');checkpoint=Path(phase['checkpoint'])
        assert sha(checkpoint)==phase['checkpoint_sha256']=='19888f8823fbb6ade6333d40bd849dec3202e522ded016e7265eaced6b52aa49'
        original=read(ROOT/'local/tracking_teacher_20261013_v1/PLAN.json')
        assert chosen==original['indices']
        sources=dict(read(INCUMBENT/'PLAN.json')['source_sha256'])
        sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
        sources[str(plan.resolve().relative_to(ROOT))]=sha(plan)
        files={str(q):sha(q) for q in (checkpoint,PHASE/'PROTOCOL.json',INCUMBENT/'RESULT.json',
            ROOT/'local/tracking_teacher_20261013_v1/PLAN.json')}
        p=dict(registered=time.time(),indices=chosen,training_role=ROLES[0],schedule_epoch=1,
            cases=48,count_distribution={'1':10,'2':24,'3':14},source_sha256=sources,input_sha256=files,
            checkpoint=str(checkpoint),seed=0,optimizer_updates=0,heldout_access=False,
            extraction='Full20.8896ms native U-Net e2 C8 Hann with existing merge;33routing blocks per case',
            source_labels='Local reference PIT assignment for supervised tracking targets only',
            same_fit48_as_B99_teacher=True,independent_recordings=False)
        self.out.mkdir(exist_ok=True);write(self.out/'PLAN.json',p)
        for name,digest in sources.items():
            dest=self.out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dest);assert sha(dest)==digest
        self.state('REGISTERED_NOT_RUNNING')

    def verify(self,p):
        assert all(sha(ROOT/name)==h and sha(self.out/'source_snapshot'/name)==h for name,h in p['source_sha256'].items())
        assert all(sha(name)==h for name,h in p['input_sha256'].items())

    def run(self):
        p=read(self.out/'PLAN.json');self.verify(p)
        with LOCK.open('r') as lock:
            self.state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);self.verify(p)
            import torch
            from models import build,predict
            from full_context_eval import evaluate
            from tracking_cache import save_case
            from diagnose_count_routing import verify_metrics
            torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            net=build('unet_mean').cuda().eval().requires_grad_(False)
            saved=torch.load(p['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
            assert saved['best']['epoch']==2, saved['best']
            net.load_state_dict(saved['model']);del saved
            data=Sequences(ROLES[0],1);assert indices(data)==p['indices'];rows=[]
            for position,index in enumerate(p['indices']):
                item_path=self.out/f'CASE_{index:03d}.json'
                if item_path.exists():
                    item=read(item_path)
                    for key in ('features','scoring','labels'):assert sha(Path(item[key]))==item[key+'_sha256']
                    verify_metrics(item['baseline_metrics'],read(self.out/f'PARENT_{index:03d}.json')['metrics'])
                    assert sha(item['metadata']['output'])==item['metadata']['output_sha256']
                    rows.append(item);continue
                refs=data.components(index,True);features=np.array(data.features[index]);row=data.rows[index];k=int(row['count'])
                assert all(data.library.clips[int(j)]['diagnostic_role']==ROLES[0] for j in row['indices'][:k])
                def progress(done,total):self.state('FIT_TEACHER_INFERENCE',case=position+1,total_cases=48,windows=done,total_windows=total)
                output,result=evaluate(net,predict,refs,features,k,'cuda',progress)
                path=self.out/f'OUTPUT_{index:03d}.npy';np.save(path,output,allow_pickle=False)
                write(self.out/f'PARENT_{index:03d}.json',result)
                metadata=dict(categories=[data.library.clips[int(j)]['category'] for j in row['indices'][:k]],
                    nominal_levels_db=row['levels'][:k].tolist(),source_indices=row['indices'][:k].tolist(),
                    output_sha256=sha(path),output=str(path),teacher_checkpoint_sha256=p['input_sha256'][p['checkpoint']])
                item=save_case(self.out,index,output,refs,k,result['metrics']['predicted_count'],metadata,training=True)
                verify_metrics(item['baseline_metrics'],result['metrics']);rows.append(item)
                del output,refs;gc.collect();torch.cuda.empty_cache()
                print('FIT_TEACHER_CASE_COMPLETE',position+1,48,flush=True)
            self.verify(p)
            for item in rows:
                for key in ('features','scoring','labels'):assert sha(Path(item[key]))==item[key+'_sha256']
                assert item['training']
            result=dict(status='VERIFIED',cases=rows,plan_sha256=sha(self.out/'PLAN.json'),cases_count=48,
                parent_epoch=2,teacher_updates=0,tracker_updates=0,training_role=ROLES[0],heldout_access=False)
            write(self.out/'RESULT.json',result);write(self.out/'AUDIT.json',dict(status='PASS',result_sha256=sha(self.out/'RESULT.json'),
                all48_teacher_metrics_reproduced=True,source_input_hashes=True,role=ROLES[0]))
            write(ROOT/'reports/2026-10-13/NATIVE_TRACKING_TEACHER_PREPARATION_CHECK.json',
                dict(status='PASS',cases=48,role=ROLES[0],parent_epoch=2,plan_sha256=sha(self.out/'PLAN.json'),
                     result_sha256=sha(self.out/'RESULT.json'),optimizer_updates=0,heldout_access=False))
            self.state('COMPLETE_AUDITED',cases=48,optimizer_updates=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','run'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--plan',type=Path)
    a=parser.parse_args();obj=Preparation(a.output)
    try:
        if a.action=='register':obj.register(a.plan)
        else:obj.run()
    except BaseException:
        obj.out.mkdir(exist_ok=True);write(obj.out/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        obj.state('FAILED');raise
