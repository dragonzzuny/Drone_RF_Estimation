"""Read-only full trajectory and independent imported-state checks."""
import argparse
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
import watch_epochs as watch
from audit_followup_queue import audit_loss


def same(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def audit(study):
    torch.set_num_threads(2)
    plan=watch.read(study/'PROTOCOL.json');origin=Path(plan['dependency'])
    if plan['executed_epochs']!=[2,3] or plan['executed_updates_per_arm']!=150:raise ValueError('Budget changed')
    imported=watch.read(study/'IMPORT.json')
    for rel,digest in imported['source_sha256'].items():
        if watch.digest(origin/rel)!=digest:raise ValueError('Imported source changed')
    for arm in plan['arms']:
        for name in ('RESUME_001.pt','ACTUAL_001.pt','SELECTED_001.pt'):
            path=study/arm/name
            if watch.digest(path)!=imported['derived_checkpoint_sha256'][f'{arm}/{name}']:
                raise ValueError('Imported state changed')
            a=torch.load(path,map_location='cpu',weights_only=False)
            source=origin/arm/('LAST.pt' if name=='RESUME_001.pt' else name)
            b=torch.load(source,map_location='cpu',weights_only=False)
            provenance=a.pop('imported_from')
            if provenance['sha256']!=watch.digest(source) or provenance['new_training_updates']!=0:
                raise ValueError('Invalid imported provenance')
            a['protocol_sha256']=plan['dependency_protocol_sha256']
            if not same(a,b):raise ValueError('Imported model, optimizer or RNG changed')
            del a,b
        event=watch.read(study/arm/'EPOCH_001.json')
        original=watch.read(origin/arm/'EPOCH_001.json')
        for key in ('imported_epoch','new_training_updates','imported_receipt_sha256','imported_protocol_sha256'):
            event.pop(key)
        event['protocol_sha256']=plan['dependency_protocol_sha256']
        if event!=original:raise ValueError('Imported first-epoch metrics changed')
        final=torch.load(study/arm/'LAST.pt',map_location='cpu',weights_only=False)
        if any(g['lr']!=1e-4 for g in final['optimizer']['param_groups']):raise ValueError('Optimizer LR changed')
        del final
    result=audit_loss(study)
    result.update(exact_imported_model_optimizer_rng_checked=True,actual_optimizer_learning_rate_checked=True,
        total_fine_tune_updates_per_arm=225,executed_updates_per_arm=150,imported_updates_per_arm=75)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('study','run','public'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve();root.mkdir(parents=True,exist_ok=True)
    try:
        if (root/'PROTOCOL.json').exists():raise ValueError('Duplicate auditor')
        files=[Path(__file__).resolve(),Path(watch.__file__).resolve(),Path(__file__).with_name('audit_followup_queue.py')]
        hashes={str(p.relative_to(watch.ROOT)):watch.digest(p) for p in files}
        registration=dict(source_sha256=hashes,study_protocol_sha256=watch.digest(args.study/'PROTOCOL.json'),time=time.time())
        for path in files:shutil.copyfile(path,root/path.name)
        watch.write(root/'PROTOCOL.json',registration);began=time.time()
        while not (args.study/'COMPLETE.json').exists():
            if (args.study/'FAILURE.json').exists():raise RuntimeError('Continuation failed')
            if time.time()-began>24*3600:raise TimeoutError('Audit wait exceeded')
            watch.write(root/'STATE.json',dict(status='WAITING_COMPLETION',pid=os.getpid(),time=time.time()))
            time.sleep(30)
        for rel,digest in hashes.items():
            if watch.digest(watch.ROOT/rel)!=digest or watch.digest(root/Path(rel).name)!=digest:
                raise ValueError('Frozen auditor changed')
        if watch.digest(args.study/'PROTOCOL.json')!=registration['study_protocol_sha256']:
            raise ValueError('Study protocol changed')
        result=audit(args.study);watch.write(args.public,result);watch.write(root/'RESULT.json',result)
        watch.write(root/'STATE.json',dict(status='PASS',pid=os.getpid(),time=time.time()));print('PASS',flush=True)
    except Exception:
        watch.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
