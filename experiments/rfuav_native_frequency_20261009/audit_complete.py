"""Post-completion audit of source/data identities, every score and selection."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from native_data import ROOT,NativeMixtures,sha256,write_json
from report import summarize


def audit(root,output):
    for name in ('PREP','GPU','SPECTRAL','FRAME'):
        if (root/f'{name}_FAILURE.json').exists() or not (root/f'{name}_COMPLETE.json').exists():
            raise RuntimeError(f'{name} incomplete/failed')
    source_checks={}
    for name in ('PREP','GPU','SPECTRAL','FRAME'):
        p=json.loads((root/f'{name}_PROTOCOL.json').read_text())
        for rel,digest in p['source_sha256'].items():
            for path in (ROOT/rel,root/'source_snapshot'/rel):
                if sha256(path)!=digest:raise RuntimeError(f'Changed source: {path}')
        source_checks[name]=len(p['source_sha256'])
    prep=json.loads((root/'PREP_PROTOCOL.json').read_text())
    for name,digest in prep['data_sha256'].items():
        if sha256(name)!=digest:raise RuntimeError('Original metadata/schedule changed')
    gp=json.loads((root/'GPU_PROTOCOL.json').read_text())
    if sha256(gp['parent'])!=gp['parent_sha256']:raise RuntimeError('Parent checkpoint changed')
    if gp['prep_protocol_sha256']!=sha256(root/'PREP_PROTOCOL.json'):raise RuntimeError('Prep/GPU protocol mismatch')
    features=[]
    for role,e in [('validation_pack',1)]+[('train_pack',e) for e in range(1,6)]:
        data=NativeMixtures(root/'preparation',role,e)
        receipt=json.loads(data.cache_stem.with_suffix('.json').read_text())
        features.append(dict(role=role,epoch=e,cases=len(data),rows_sha256=data.rows_hash,
            edge_shifted_cases=len(receipt['shifted_edge_rows']),features_sha256=receipt['features_sha256']))
    validations=[];parent_rows=None
    for epoch in range(6):
        path=root/'gpu'/f'VALIDATION_{epoch:03d}.json'
        result=json.loads(path.read_text());summary=summarize(path);validations.append(dict(epoch=epoch,**summary))
        if parent_rows is None:parent_rows=result['rows']
        for a,b in zip(parent_rows,result['rows']):
            if any(a[k]!=b[k] for k in ('index','count','categories','pack_ids','nominal_levels_db')):
                raise RuntimeError('Mixture identity changed between epochs')
            np.testing.assert_allclose(a['reference_power'],b['reference_power'],rtol=1e-10,atol=1e-15)
            if b['sum_relative_error']>1e-9:raise RuntimeError('Mixture sum error')
        if epoch:
            receipt=json.loads((root/'gpu'/f'EPOCH_{epoch:03d}.json').read_text())
            if receipt['updates']!=75*epoch or receipt['protocol_sha256']!=sha256(root/'GPU_PROTOCOL.json'):
                raise RuntimeError('Wrong update budget')
            selected=min(validations,key=lambda v:v['selection_nmse'])
            if receipt['best']['epoch']!=selected['epoch'] or not np.isclose(receipt['best']['metric'],selected['selection_nmse']):
                raise RuntimeError('Selection used a different criterion')
    spectral=summarize(root/'spectral/VALIDATION.json')
    spectral_rows=json.loads((root/'spectral/VALIDATION.json').read_text())['rows']
    for a,b in zip(parent_rows,spectral_rows):
        if any(a[k]!=b[k] for k in ('index','count','categories','pack_ids','nominal_levels_db')):
            raise RuntimeError('Spectral comparator evaluated other cases')
        np.testing.assert_allclose(a['reference_power'],b['reference_power'],rtol=1e-10,atol=1e-15)
    frame=summarize(root/'spectral/FRAME_VALIDATION.json')
    frame_rows=json.loads((root/'spectral/FRAME_VALIDATION.json').read_text())['rows']
    for a,b in zip(spectral_rows,frame_rows):
        if any(a[k]!=b[k] for k in ('index','count','categories','pack_ids','nominal_levels_db','predicted_count')):
            raise RuntimeError('Frame comparator changed cases/count rule')
        np.testing.assert_allclose(a['reference_power'],b['reference_power'],rtol=1e-10,atol=1e-15)
    templates=json.loads((root/'spectral/TEMPLATES.json').read_text())
    train_ids={c['clip_id'] for c in data.library.clips if c['role']=='train_pack'}
    if set(templates['clip_ids'])!=train_ids or len(templates['clip_ids'])!=3902:
        raise RuntimeError('Template scope mismatch')
    folder=root/'gpu'
    if sha256(folder/'BEST.pt')!=sha256(folder/'SELECTED_005.pt'):
        raise RuntimeError('Selected copy changed')
    torch.set_num_threads(2);checkpoints=[]
    for name in ('SELECTED_001.pt','SELECTED_005.pt','LAST.pt'):
        saved=torch.load(folder/name,map_location='cpu',weights_only=False)
        expected=min(validations[:2] if name=='SELECTED_001.pt' else validations,key=lambda v:v['selection_nmse'])
        if saved['protocol_sha256']!=sha256(root/'GPU_PROTOCOL.json') or saved['best']['epoch']!=expected['epoch']:
            raise RuntimeError('Checkpoint protocol/selection mismatch')
        if name=='LAST.pt' and (saved['epoch']!=5 or saved['updates']!=375):raise RuntimeError('Training incomplete')
        elements=sum(t.numel() for t in saved['model'].values())
        if elements!=32142859 or any(not bool(torch.isfinite(t).all()) for t in saved['model'].values()):
            raise RuntimeError('Nonfinite/changed-capacity checkpoint')
        checkpoints.append(dict(name=name,sha256=sha256(folder/name),elements=elements,best=saved['best']))
        del saved;gc.collect()
    result=dict(status='PASS',auditor_sha256=sha256(Path(__file__)),
        summary_code_sha256=sha256(Path(__file__).with_name('report.py')),
        source_checks=source_checks,original_metadata_checks=len(prep['data_sha256']),
        features=features,validation_epochs=validations,spectral=spectral,frame_spectral=frame,checkpoints=checkpoints,
        selected_epoch=expected['epoch'],heldout_read=False,
        cache_audit_scope='FIR preparation and consuming loaders checked full file hashes; this final audit rechecks sealed metadata/features, not all 67.6GB of IQ again',
        independent_test=False,physical_aircraft_count=False)
    write_json(output,result);print(json.dumps(dict(status='PASS',selected_epoch=result['selected_epoch'],checkpoints=checkpoints)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args();audit(args.run,args.output)
