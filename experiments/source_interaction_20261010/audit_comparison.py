"""Read-only audit of all actual/selected source-head study checkpoints."""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback
import torch
import train_comparison as worker

w=worker.watch


def same(a,b):
    return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)


def audit(root):
    torch.set_num_threads(2)
    p=w.read(root/'PROTOCOL.json');worker.verify(root,p)
    digest=w.digest(root/'PROTOCOL.json')
    _,ids=w.validate(Path(p['validation_identity_template']),None)
    results={};hashes={}
    for arm in worker.ARMS:
        folder=root/arm
        net=worker.make_model(arm,Path(p['parent_checkpoint']))
        initial={k:v.detach().clone() for k,v in net.state_dict().items()}
        assert sum(q.numel() for q in net.parameters())==p['parameters'][arm]
        del net
        history=[]
        for epoch in range(4):
            path=folder/f'VALIDATION_{epoch:03d}.json'
            metrics,_=w.validate(path,ids);history.append(metrics)
            hashes[f'{arm}/VALIDATION_{epoch:03d}.json']=w.digest(path)
            if epoch==0:continue
            receipt=w.read(folder/f'EPOCH_{epoch:03d}.json')
            if (receipt['arm']!=arm or receipt['updates']!=epoch*75
                    or receipt['protocol_sha256']!=digest or len(receipt['preclip_gradient_norms'])!=75):
                raise ValueError('Epoch budget/identity mismatch')
            for a,b in zip(receipt['validation']['by_count'],metrics['by_count']):
                for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):
                    if not w.close(a[key],b[key]):raise ValueError('Receipt metrics differ')
            actual=torch.load(folder/f'ACTUAL_{epoch:03d}.pt',map_location='cpu',weights_only=False)
            if actual['epoch']!=epoch or actual['updates']!=epoch*75 or actual['arm']!=arm or actual['protocol_sha256']!=digest:
                raise ValueError('Actual checkpoint identity mismatch')
            if actual['model'].keys()!=initial.keys() or any(
                v.shape!=initial[k].shape or not torch.isfinite(v).all() for k,v in actual['model'].items()):
                raise ValueError('Parameter shape/finite check failed')
            chosen=min(history,key=lambda r:r['selection_nmse'])
            selected=torch.load(folder/f'SELECTED_{epoch:03d}.pt',map_location='cpu',weights_only=False)
            expected=initial if chosen['epoch']==0 else torch.load(
                folder/f"ACTUAL_{chosen['epoch']:03d}.pt",map_location='cpu',weights_only=False)['model']
            if (selected['best']['epoch']!=chosen['epoch'] or selected['protocol_sha256']!=digest
                    or not same(selected['model'],expected)):
                raise ValueError('Selected checkpoint differs from prescribed selection')
            hashes[f'{arm}/ACTUAL_{epoch:03d}.pt']=w.digest(folder/f'ACTUAL_{epoch:03d}.pt')
            del actual,selected,expected
        final=torch.load(folder/'LAST.pt',map_location='cpu',weights_only=False)
        actual=torch.load(folder/'ACTUAL_003.pt',map_location='cpu',weights_only=False)
        best=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False)
        selected=torch.load(folder/'SELECTED_003.pt',map_location='cpu',weights_only=False)
        if not same(final['model'],actual['model']) or not same(best['model'],selected['model']):
            raise ValueError('LAST/BEST final transaction differs')
        if final['updates']!=225 or {int(s['step']) for s in final['optimizer']['state'].values()}!={225}:
            raise ValueError('Final optimizer budget differs')
        if any(group['lr']!=1e-5 for group in final['optimizer']['param_groups']):
            raise ValueError('Final optimizer learning rate differs')
        results[arm]=dict(selected=min(history,key=lambda r:r['selection_nmse']),final=history[-1],
                          parameters=p['parameters'][arm],updates=225)
        del initial,final,actual,best,selected
        gc.collect()
    return dict(status='PASS',study_protocol_sha256=digest,
        complete_sha256=w.digest(root/'COMPLETE.json'),results=results,
        checked_checkpoint_sha256=hashes,all_epochs_and_630_rows_checked=True,
        actual_and_selected_tensors_checked=True,optimizer_counters_and_lr_checked=True,
        recorded_iq_reads=0,heldout_read=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('study','run','public'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    try:
        if (root/'PROTOCOL.json').exists():raise ValueError('Duplicate auditor registration')
        registration=dict(auditor_sha256=w.digest(Path(__file__)),worker_sha256=w.digest(Path(worker.__file__)),
                          study_protocol_sha256=w.digest(a.study/'PROTOCOL.json'))
        w.write(root/'PROTOCOL.json',registration);started=time.time()
        while not (a.study/'COMPLETE.json').exists():
            if (a.study/'FAILURE.json').exists():raise RuntimeError('Training failed')
            if time.time()-started>24*3600:raise TimeoutError('Audit wait limit')
            w.write(root/'STATE.json',dict(status='WAITING_COMPLETION',pid=os.getpid(),time=time.time()))
            time.sleep(30)
        if (registration['auditor_sha256']!=w.digest(Path(__file__))
                or registration['worker_sha256']!=w.digest(Path(worker.__file__))
                or registration['study_protocol_sha256']!=w.digest(a.study/'PROTOCOL.json')):
            raise ValueError('Registered audit inputs changed')
        result=audit(a.study);w.write(a.public,result);w.write(root/'RESULT.json',result)
        w.write(root/'STATE.json',dict(status='PASS',pid=os.getpid(),time=time.time()))
        print('PASS',flush=True)
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
