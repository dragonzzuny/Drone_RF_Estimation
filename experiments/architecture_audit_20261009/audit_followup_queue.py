"""Read-only completion checks for the two registered U-Net followups.

Never starts/stops training or reads I/Q. Fit-diagnostic optimizer states were
deliberately discarded, so only their logged budgets/metrics can be audited.
The subsequent full loss study retains actual/selected/optimizer checkpoints.
"""
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


def sources(study):
    protocol=watch.read(study/'PROTOCOL.json')
    for rel,digest in protocol['source_sha256'].items():
        if watch.digest(watch.ROOT/rel)!=digest or watch.digest(study/'source_snapshot'/rel)!=digest:
            raise ValueError('Frozen study source changed: '+rel)
    if watch.digest(Path(protocol['preparation'])/'PREPARATION.json')!=protocol['preparation_sha256']:
        raise ValueError('Prepared RF manifest changed')
    return protocol


def wait_for(study,root,stage):
    began=time.time()
    while not (study/'COMPLETE.json').exists():
        if (study/'FAILURE.json').exists():raise RuntimeError('Predecessor study failed: '+stage)
        if time.time()-began>24*3600:raise TimeoutError('Completion wait exceeded')
        watch.write(root/'STATE.json',dict(status='WAITING_COMPLETION',stage=stage,pid=os.getpid(),time=time.time()))
        time.sleep(30)


def audit_fit(study):
    plan=sources(study);done=watch.read(study/'COMPLETE.json')
    if done['protocol_sha256']!=watch.digest(study/'PROTOCOL.json'):raise ValueError('Fit protocol mismatch')
    if {r['arm'] for r in done['results']}!=set(plan['arms']):raise ValueError('Missing output arm')
    scores={}
    for result in done['results']:
        if (result['updates']!=256 or result['parameters']!=32_142_859
                or not result['finite_gradients'] or not result['weights_discarded']):
            raise ValueError('Wrong fit record')
        history=result['history']
        if [h['step'] for h in history]!=[0,1,8,16,32,64,128,192,256]:
            raise ValueError('Incomplete fit trajectory')
        for h in history:
            if [r['count'] for r in h['rows']]!=[2,2,3,3]:raise ValueError('Different fixed fit cases')
            for c in (2,3):
                rows=[r for r in h['rows'] if r['count']==c]
                if any(len(r['nmse'])!=c or len(r['si_sdr'])!=c for r in rows):raise ValueError('Metric count mismatch')
                nmse=statistics.mean(v for r in rows for v in r['nmse'])
                si=statistics.mean(v for r in rows for v in r['si_sdr'])
                stored=next(v for v in h['by_count'] if v['count']==c)
                if not watch.close(nmse,stored['mean_nmse']) or not watch.close(si,stored['mean_si_sdr']):
                    raise ValueError('Fit aggregate mismatch')
        improved=all(b['mean_nmse']<a['mean_nmse'] for a,b in zip(history[0]['by_count'],history[-1]['by_count']))
        tight=all(r['mean_nmse']<=.1 for r in history[-1]['by_count'])
        if improved!=result['both_counts_improved'] or tight!=result['both_counts_nmse_le_point1']:
            raise ValueError('Fit decision mismatch')
        scores[result['arm']]=dict(initial=history[0]['by_count'],final=history[-1]['by_count'],
                                  both_counts_improved=improved,both_counts_nmse_le_point1=tight)
    return dict(status='PASS',study_protocol_sha256=watch.digest(study/'PROTOCOL.json'),
                complete_sha256=watch.digest(study/'COMPLETE.json'),results=scores,
                limitation='Recorded fit steps only; optimizer states/weights discarded by registered design',
                waveform_reads=0,heldout_read=False)


def audit_loss(study):
    import torch
    torch.set_num_threads(2)
    plan=sources(study);digest=watch.digest(study/'PROTOCOL.json')
    done=watch.read(study/'COMPLETE.json')
    if done['protocol_sha256']!=digest or done['new_updates_per_arm']!=225:raise ValueError('Wrong final loss budget')
    parent=torch.load(plan['parent_checkpoint'],map_location='cpu',weights_only=False)
    if watch.digest(Path(plan['parent_checkpoint']))!=plan['parent_checkpoint_sha256']:raise ValueError('Parent weights changed')
    parent_metric,identities=watch.validate(Path(plan['validation_identity_template']),None)
    def same(first,second):
        return first.keys()==second.keys() and all(torch.equal(first[k],second[k]) for k in first)
    summaries={};hashes={}
    for arm in plan['arms']:
        folder=study/arm;history=[]
        for epoch in range(4):
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
        actual=torch.load(folder/'ACTUAL_003.pt',map_location='cpu',weights_only=False)
        if final['updates']!=225 or final['epoch']!=3 or not same(final['model'],actual['model']):raise ValueError('LAST mismatch')
        if {int(v['step']) for v in final['optimizer']['state'].values() if 'step' in v}!={225}:
            raise ValueError('Optimizer counter mismatch')
        best=min(history,key=lambda x:x['selection_nmse'])
        chosen=torch.load(folder/'SELECTED_003.pt',map_location='cpu',weights_only=False)
        stored=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False)
        if final['best']['epoch']!=best['epoch'] or not same(chosen['model'],stored['model']):raise ValueError('BEST transaction mismatch')
        summaries[arm]=dict(selected=best,final=history[-1],new_updates=225,parent_updates=300)
        del final,actual,chosen,stored
    return dict(status='PASS',study_protocol_sha256=digest,complete_sha256=watch.digest(study/'COMPLETE.json'),
                checked_actual_checkpoint_sha256=hashes,results=summaries,
                source_and_parameter_shapes_checked=True,optimizer_counters_checked=True,
                all_validation_rows_checked=True,waveform_reads=0,heldout_read=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('fit','loss','run','public-directory'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    try:
        if (root/'PROTOCOL.json').exists():raise ValueError('Duplicate completion auditor')
        files=[Path(__file__).resolve(),Path(watch.__file__).resolve()]
        registration=dict(source_sha256={str(f.relative_to(watch.ROOT)):watch.digest(f) for f in files},
                          fit_protocol_sha256=watch.digest(a.fit/'PROTOCOL.json'),
                          loss_protocol_sha256=watch.digest(a.loss/'PROTOCOL.json'),time=time.time())
        for f in files:shutil.copyfile(f,root/f.name)
        watch.write(root/'PROTOCOL.json',registration)
        for stage,study,auditor,filename in [('output_fit',a.fit,audit_fit,'OUTPUT_PARAMETERIZATION_FINAL_AUDIT.json'),
                                           ('robust_loss',a.loss,audit_loss,'ROBUST_UNET_FINAL_AUDIT.json')]:
            wait_for(study,root,stage)
            if watch.digest(study/'PROTOCOL.json')!=registration['fit_protocol_sha256' if stage=='output_fit' else 'loss_protocol_sha256']:
                raise ValueError('Audited protocol changed')
            for rel,digest in registration['source_sha256'].items():
                if watch.digest(watch.ROOT/rel)!=digest or watch.digest(root/Path(rel).name)!=digest:
                    raise ValueError('Auditor source changed')
            result=auditor(study);watch.write(a.public_directory/filename,result);watch.write(root/filename,result)
            print(json.dumps(dict(stage=stage,status=result['status'])),flush=True)
        watch.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
    except Exception:
        watch.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
