"""Independent read-only audit for the registered one-epoch learning-rate followup."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback
import watch_epochs as watch
from audit_followup_queue import sources

def audit_loss(study):
    import torch
    torch.set_num_threads(2)
    plan=sources(study);digest=watch.digest(study/'PROTOCOL.json')
    done=watch.read(study/'COMPLETE.json')
    if done['protocol_sha256']!=digest or done['new_updates_per_arm']!=75:raise ValueError('Wrong final loss budget')
    parent=torch.load(plan['parent_checkpoint'],map_location='cpu',weights_only=False)
    if watch.digest(Path(plan['parent_checkpoint']))!=plan['parent_checkpoint_sha256']:raise ValueError('Parent weights changed')
    parent_metric,identities=watch.validate(Path(plan['validation_identity_template']),None)
    def same(first,second):
        return first.keys()==second.keys() and all(torch.equal(first[k],second[k]) for k in first)
    summaries={};hashes={}
    for arm in plan['arms']:
        folder=study/arm;history=[]
        for epoch in range(2):
            value,_=watch.validate(folder/f'VALIDATION_{epoch:03d}.json',identities);history.append(value)
            if epoch==0:
                for a,b in zip(value['by_count'],parent_metric['by_count']):
                    for field in ('mean_nmse','mean_si_sdr'):
                        if not math.isclose(a[field],b[field],rel_tol=1e-5,abs_tol=1e-7):raise ValueError('Initial score changed')
                continue
            receipt=watch.read(folder/f'EPOCH_{epoch:03d}.json')
            best=min(history,key=lambda x:x['selection_nmse'])
            if receipt['arm']!=arm or receipt['epoch']!=epoch or receipt['updates']!=75*epoch or receipt['protocol_sha256']!=digest:
                raise ValueError('Epoch receipt mismatch')
            if receipt['best']['epoch']!=best['epoch'] or not watch.close(receipt['best']['metric'],best['selection_nmse']):
                raise ValueError('Selection mismatch')
            gradients=receipt['preclip_gradient_norms']
            if len(gradients)!=75 or any(not math.isfinite(v) or v<0 for v in gradients):raise ValueError('Gradient budget mismatch')
            if not watch.close(statistics.mean(v>1 for v in gradients),receipt['clipped_fraction']):raise ValueError('Clipping ratio mismatch')
            actual=torch.load(folder/f'ACTUAL_{epoch:03d}.pt',map_location='cpu',weights_only=False)
            if actual['arm']!=arm or actual['epoch']!=epoch or actual['updates']!=75*epoch or actual['protocol_sha256']!=digest:
                raise ValueError('Actual checkpoint mismatch')
            if actual['model'].keys()!=parent['model'].keys():raise ValueError('Architecture state keys changed')
            if any(v.shape!=parent['model'][k].shape or not torch.isfinite(v).all() for k,v in actual['model'].items()):
                raise ValueError('Changed shapes or nonfinite parameters')
            selected=torch.load(folder/f'SELECTED_{epoch:03d}.pt',map_location='cpu',weights_only=False)
            expected=parent if best['epoch']==0 else torch.load(folder/f"ACTUAL_{best['epoch']:03d}.pt",map_location='cpu',weights_only=False)
            if selected['protocol_sha256']!=digest or selected['best']['epoch']!=best['epoch'] or not same(selected['model'],expected['model']):
                raise ValueError('Selected tensors differ from selected epoch')
            hashes[f'{arm}/ACTUAL_{epoch:03d}.pt']=watch.digest(folder/f'ACTUAL_{epoch:03d}.pt')
            del actual,selected,expected
        final=torch.load(folder/'LAST.pt',map_location='cpu',weights_only=False)
        actual=torch.load(folder/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
        if final['updates']!=75 or final['epoch']!=1 or not same(final['model'],actual['model']):raise ValueError('LAST mismatch')
        if {int(v['step']) for v in final['optimizer']['state'].values() if 'step' in v}!={75}:
            raise ValueError('Optimizer counter mismatch')
        best=min(history,key=lambda x:x['selection_nmse'])
        chosen=torch.load(folder/'SELECTED_001.pt',map_location='cpu',weights_only=False)
        stored=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False)
        if final['best']['epoch']!=best['epoch'] or not same(chosen['model'],stored['model']):raise ValueError('BEST transaction mismatch')
        summaries[arm]=dict(selected=best,final=history[-1],new_updates=75,parent_updates=300)
        del final,actual,chosen,stored
    return dict(status='PASS',study_protocol_sha256=digest,complete_sha256=watch.digest(study/'COMPLETE.json'),
                checked_actual_checkpoint_sha256=hashes,results=summaries,
                source_and_parameter_shapes_checked=True,optimizer_counters_checked=True,
                all_validation_rows_checked=True,waveform_reads=0,heldout_read=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('study', 'run', 'public'):
        parser.add_argument('--'+key, required=True, type=Path)
    args = parser.parse_args()
    root = args.run.resolve(); root.mkdir(parents=True, exist_ok=True)
    try:
        if (root/'PROTOCOL.json').exists():
            raise ValueError('Duplicate auditor')
        files = [Path(__file__).resolve(), Path(watch.__file__).resolve(),
                 Path(__file__).with_name('audit_followup_queue.py').resolve()]
        hashes = {str(p.relative_to(watch.ROOT)):watch.digest(p) for p in files}
        registration = dict(source_sha256=hashes,
            study_protocol_sha256=watch.digest(args.study/'PROTOCOL.json'),time=time.time())
        for p in files:shutil.copyfile(p, root/p.name)
        watch.write(root/'PROTOCOL.json', registration)
        began=time.time()
        while not (args.study/'COMPLETE.json').exists():
            if (args.study/'FAILURE.json').exists():raise RuntimeError('Followup failed')
            state=watch.read(args.study/'STATE.json')
            if state['status']=='SKIPPED_REFERENCE_IMPROVED':
                watch.write(root/'STATE.json',dict(status='SKIPPED_REFERENCE_IMPROVED',time=time.time()))
                break
            if time.time()-began>24*3600:raise TimeoutError('Wait exceeded')
            watch.write(root/'STATE.json',dict(status='WAITING_COMPLETION',pid=os.getpid(),time=time.time()))
            time.sleep(30)
        else:
            for rel,digest in hashes.items():
                if watch.digest(watch.ROOT/rel)!=digest or watch.digest(root/Path(rel).name)!=digest:
                    raise ValueError('Auditor source changed')
            if watch.digest(args.study/'PROTOCOL.json')!=registration['study_protocol_sha256']:
                raise ValueError('Study protocol changed')
            result=audit_loss(args.study)
            plan=watch.read(args.study/'PROTOCOL.json')
            if plan['learning_rate']!=1e-4 or plan['epochs_per_arm']!=1:
                raise ValueError('Unexpected learning rate or budget')
            import torch
            for arm in plan['arms']:
                saved=torch.load(args.study/arm/'LAST.pt',map_location='cpu',weights_only=False)
                if any(group['lr']!=1e-4 for group in saved['optimizer']['param_groups']):
                    raise ValueError('Actual optimizer learning rate differs')
                del saved
            result['actual_optimizer_learning_rate_checked']=True
            watch.write(args.public,result);watch.write(root/'RESULT.json',result)
            watch.write(root/'STATE.json',dict(status='PASS',pid=os.getpid(),time=time.time()))
            print(json.dumps(dict(status='PASS')),flush=True)
    except Exception:
        watch.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
