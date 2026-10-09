"""CPU fixed-TRAIN probe for completed WaveNet epochs1/2, with backend checks.

The two epoch-average losses used different mixtures. Evaluate the same24
TRAIN-epoch1 mixtures instead. Reproduce four existing development-validation
cases first; never open reserved confirmation data. No model is trained here.
"""
import argparse
import ast
from collections import defaultdict
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import fit_diagnostic as fit
from native_wavenet import build_native_wavenet
from models import predict
from native_data import NativeMixtures, sha256, write_json
from drone_rf.waveform import waveform_metrics

ROOT=Path(__file__).resolve().parents[2]


def evaluate_one(net,data,index):
    item=data[index]
    batch={k:torch.as_tensor(np.asarray(item[k])[None]) for k in
           ('mixture','references','active','context_features','crop_start')}
    with torch.inference_mode():
        estimates,_=predict(net,batch)
        result=waveform_metrics(estimates,batch['references'],batch['active'],batch['mixture'])
    count=item['construction_count']
    nmse=result['nmse'][0,:count].tolist()
    si=result['si_sdr'][0,:count].tolist()
    if not np.isfinite(nmse).all() or not np.isfinite(si).all():
        raise ValueError('Nonfinite probe score')
    if result['sum_relative_error'].max()>1e-9:
        raise ValueError('CPU mixture sum failed')
    row=data.rows[index]
    clips=[data.library.clips[int(i)] for i in row['indices'][:count]]
    return dict(index=index,count=count,categories=[c['category'] for c in clips],
        pack_ids=[c['pack_id'] for c in clips],nmse=nmse,si_sdr=si,
        reference_power=result['reference_power'][0,:count].tolist(),
        assignment=result['assignment'][0].tolist())


def run(screen, output, reuse_first=None):
    if (output/'PROTOCOL.json').exists():
        raise ValueError('Duplicate CPU probe registration')
    output.mkdir(parents=True,exist_ok=True)
    plan=fit.read(screen/'PROTOCOL.json')
    folder=screen/'wavenet_cycle10'
    if not (folder/'EPOCH_002.json').exists():
        raise ValueError('Both model epochs must finish first')
    train=NativeMixtures(plan['preparation'],'train_pack',1)
    chosen=defaultdict(list)
    for index,row in enumerate(train.rows):
        count=int(row['count'])
        if count not in (2,3):
            continue
        categories=tuple(train.library.clips[int(i)]['category'] for i in row['indices'][:count])
        key=(count,categories)
        cap=3 if count==2 else 12
        if len(chosen[key])<cap:
            chosen[key].append(index)
    indices=sorted(i for group in chosen.values() for i in group)
    if len(indices)!=24 or sum(int(train.rows[i]['count'])==2 for i in indices)!=12:
        raise ValueError('Expected12 two-source and12 three-source fixed TRAIN cases')
    paths={1:folder/'SELECTED_001.pt',2:folder/'LAST.pt'}
    copies={}
    for epoch,path in paths.items():
        digest=sha256(path)
        target=output/f'wave_e{epoch}.pt'
        shutil.copyfile(path,target)
        if sha256(target)!=digest or sha256(path)!=digest:
            raise ValueError('Checkpoint changed during snapshot')
        copies[str(epoch)]=dict(path=str(target),source=str(path),sha256=digest)
    source=dict(plan['source_sha256'])
    source[str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__))
    protocol=dict(status='REGISTERED_FIXED_TRAIN_CPU_PROBE',source_sha256=source,
        screen_protocol_sha256=sha256(screen/'PROTOCOL.json'),
        preparation=plan['preparation'],preparation_sha256=plan['preparation_sha256'],
        checkpoint_copies=copies,train_indices=indices,validation_reproduction_indices=[4,5,2,11],
        train_selection='First3 TRAIN-epoch1 cases per two-source category pair and first12 triple cases, before scoring',
        backend_tolerance=dict(nmse_rtol=1e-3,nmse_atol=1e-5,si_sdr_atol_db=.01),
        assignment_check='active references only; zero-padded target slots are interchangeable',
        independent_test=False,heldout_read=False,training=False)
    reusable=None
    if reuse_first is not None:
        old=fit.read(reuse_first/'PROTOCOL.json')
        if (old['train_indices']!=indices or old['preparation_sha256']!=protocol['preparation_sha256']
                or old['checkpoint_copies']['1']['sha256']!=copies['1']['sha256']
                or old['backend_tolerance']!=protocol['backend_tolerance']):
            raise ValueError('Cannot reuse a different completed first-epoch probe')
        script_rel=str(Path(__file__).relative_to(ROOT))
        for rel,digest in old['source_sha256'].items():
            if sha256(reuse_first/'source_snapshot'/rel)!=digest:
                raise ValueError('Original probe snapshot changed')
            if rel!=script_rel and source[rel]!=digest:
                raise ValueError('Reused evaluation dependencies changed')
        def function_tree(path):
            tree=ast.parse(path.read_text())
            return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_one'))
        if function_tree(Path(__file__))!=function_tree(reuse_first/'source_snapshot'/script_rel):
            raise ValueError('Evaluation function changed')
        reusable=fit.read(reuse_first/'EPOCH_001.json')
        if [r['index'] for r in reusable['rows']]!=indices or len(reusable['backend_reproduction'])!=4:
            raise ValueError('First-epoch probe incomplete')
        for group in reusable['by_count']:
            rows=[r for r in reusable['rows'] if r['count']==group['count']]
            if len(rows)!=12:
                raise ValueError('Reused count balance changed')
            for key,values in (('mean_nmse',[v for r in rows for v in r['nmse']]),
                               ('mean_si_sdr',[v for r in rows for v in r['si_sdr']]),
                               ('weakest_nmse',[r['nmse'][int(np.argmin(r['reference_power']))] for r in rows])):
                if not np.isfinite(values).all() or not np.isclose(np.mean(values),group[key],rtol=1e-12,atol=1e-12):
                    raise ValueError('Reused aggregate mismatch')
        protocol['reused_first_epoch']=dict(path=str(reuse_first/'EPOCH_001.json'),
            sha256=sha256(reuse_first/'EPOCH_001.json'),prior_protocol_sha256=sha256(reuse_first/'PROTOCOL.json'),
            reason='Only the irrelevant inactive-slot order assertion changed; evaluation AST, all dependencies, weights, data and cases match')
    for rel,digest in source.items():
        if sha256(ROOT/rel)!=digest:
            raise ValueError('Source changed')
        target=output/'source_snapshot'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target)
    write_json(output/'PROTOCOL.json',protocol)
    torch.set_num_threads(2)
    val=NativeMixtures(plan['preparation'],'validation_pack',1)
    results={}
    for epoch in (1,2):
        if epoch==1 and reusable is not None:
            results['1']=dict(**reusable,reused_from=protocol['reused_first_epoch'])
            write_json(output/'EPOCH_001.json',results['1'])
            print(json.dumps(dict(epoch=1,status='REUSED_VERIFIED_COMPLETE_RESULT',by_count=reusable['by_count'])),flush=True)
            continue
        saved=torch.load(copies[str(epoch)]['path'],map_location='cpu',weights_only=False)
        if saved['protocol_sha256']!=sha256(screen/'PROTOCOL.json'):
            raise ValueError('Wrong checkpoint protocol')
        if (epoch==1 and saved['best']['epoch']!=1) or (epoch==2 and (saved['epoch']!=2 or saved['updates']!=150)):
            raise ValueError('Wrong actual weight epoch')
        net=build_native_wavenet()
        net.load_state_dict(saved['model'])
        del saved
        gc.collect()
        net.eval()
        expected={r['index']:r for r in fit.read(folder/f'VALIDATION_{epoch:03d}.json')['rows']}
        reproduction=[]
        for index in protocol['validation_reproduction_indices']:
            actual=evaluate_one(net,val,index)
            reference=expected[index]
            if (any(actual[k]!=reference[k] for k in ('count','categories','pack_ids')) or
                    actual['assignment'][:actual['count']]!=reference['assignment'][:actual['count']]):
                raise ValueError('CPU validation identity or active assignment mismatch')
            np.testing.assert_allclose(actual['reference_power'],reference['reference_power'],rtol=1e-10,atol=1e-15)
            np.testing.assert_allclose(actual['nmse'],reference['nmse'],rtol=1e-3,atol=1e-5)
            np.testing.assert_allclose(actual['si_sdr'],reference['si_sdr'],rtol=0,atol=.01)
            reproduction.append(dict(index=index,max_nmse_absolute_difference=float(np.max(np.abs(np.array(actual['nmse'])-reference['nmse']))),
                                     max_si_absolute_difference_db=float(np.max(np.abs(np.array(actual['si_sdr'])-reference['si_sdr'])))))
        rows=[]
        begin=time.time()
        for index in indices:
            rows.append(evaluate_one(net,train,index))
            write_json(output/'STATE.json',dict(status='CPU_FIXED_TRAIN_EVALUATION',epoch=epoch,cases=len(rows),total=24,time=time.time()))
        summary=[]
        for count in (2,3):
            items=[r for r in rows if r['count']==count]
            summary.append(dict(count=count,cases=len(items),
                mean_nmse=float(np.mean([v for r in items for v in r['nmse']])),
                mean_si_sdr=float(np.mean([v for r in items for v in r['si_sdr']])),
                weakest_nmse=float(np.mean([r['nmse'][int(np.argmin(r['reference_power']))] for r in items]))))
        results[str(epoch)]=dict(by_count=summary,rows=rows,backend_reproduction=reproduction,seconds=time.time()-begin)
        write_json(output/f'EPOCH_{epoch:03d}.json',results[str(epoch)])
        print(json.dumps(dict(epoch=epoch,by_count=summary)),flush=True)
        del net
        gc.collect()
    for rel,digest in source.items():
        if sha256(ROOT/rel)!=digest or sha256(output/'source_snapshot'/rel)!=digest:
            raise ValueError('Probe source changed')
    write_json(output/'COMPLETE.json',dict(status='COMPLETE',results=results,
        protocol_sha256=sha256(output/'PROTOCOL.json'),heldout_read=False,
        interpretation='Small fixed TRAIN diagnostic plus4 existing development cases per checkpoint for backend reproduction; no independent test or causal overfitting claim',time=time.time()))
    write_json(output/'STATE.json',dict(status='COMPLETED',time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-first',type=Path)
    a=p.parse_args()
    try:
        run(a.screen.resolve(),a.output.resolve(),a.reuse_first.resolve() if a.reuse_first else None)
    except Exception:
        import traceback
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
