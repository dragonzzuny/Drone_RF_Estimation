"""Same extreme-batch head-gradient diagnostic for the selected full U-Net.

Independent within-model diagnostic. Gradient magnitudes across the U-Net and
packed WaveNet are not directly comparable: output heads and budgets differ.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

import head_gradient_probe as core
import watch_epochs as watch
from native_data import NativeMixtures
from models import build, predict
from drone_rf.losses import pit_waveform_loss


def run(study, prerequisite, root, public):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate U-Net gradient audit')
    root.mkdir(parents=True,exist_ok=True)
    parent=watch.read(study/'PROTOCOL.json')
    prior=watch.read(prerequisite/'COMPLETE.json')
    if prior['status']!='COMPLETE' or len(prior['rows'])!=32:
        raise ValueError('Prior batch definition incomplete')
    indices=prior['train_indices']
    metadata={r['index']:r for r in prior['rows']}
    if indices!=list(range(indices[0],indices[0]+32)) or indices[0]%32:
        raise ValueError('Not one exact scheduled batch')
    if parent['preparation_sha256']!=prior['preparation_sha256']:
        raise ValueError('Different prepared RF data')
    checkpoint=study/'unet_mean/SELECTED_004.pt'
    source=dict(prior['source_sha256'])
    for rel,digest in parent['source_sha256'].items():
        if rel in source and source[rel]!=digest:
            raise ValueError('Conflicting frozen dependency')
        source[rel]=digest
    source[str(Path(__file__).relative_to(watch.ROOT))]=watch.digest(Path(__file__))
    for rel,digest in source.items():
        if watch.digest(watch.ROOT/rel)!=digest:
            raise ValueError('Source changed')
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(watch.ROOT/rel,target)
    protocol=dict(status='REGISTERED_UNET_HEAD_ONLY_GRADIENT_DIAGNOSIS',source_sha256=source,
        study_protocol_sha256=watch.digest(study/'PROTOCOL.json'),
        prerequisite_sha256=watch.digest(prerequisite/'COMPLETE.json'),
        checkpoint_sha256=watch.digest(checkpoint),checkpoint_arm='unet_mean',checkpoint_epoch=4,
        checkpoint_updates=300,parameters=32_142_859,preparation_sha256=parent['preparation_sha256'],
        train_indices=indices,selection=prior['selection'],
        selected_anchor_index=prior['selected_anchor_index'],selected_anchor_gap_db=prior['selected_anchor_gap_db'],
        modes=prior['modes'],derivative_scope='final Conv2d weight and bias only',optimizer_updates=0,
        validation_read=False,heldout_read=False,post_hoc=True,
        caveat='Selected extreme batch; not full-network or historical gradients; no trained robust-loss result')
    watch.write(root/'PROTOCOL.json',protocol)
    train=NativeMixtures(parent['preparation'],'train_pack',1)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['best']['epoch']!=4 or saved['arm']!='unet_mean' or saved['protocol_sha256']!=protocol['study_protocol_sha256']:
        raise ValueError('Wrong U-Net weights')
    net=build('unet_mean').eval();net.load_state_dict(saved['model']);del saved
    net.requires_grad_(False);net.output.requires_grad_(True)
    parameters=tuple(net.output.parameters())
    vectors={mode:[] for mode in protocol['modes']};rows=[];began=time.time()
    for index in indices:
        item=train[index]
        batch={k:torch.as_tensor(np.asarray(item[k]))[None] for k in
               ('mixture','references','active','context_features','crop_start','construction_count')}
        estimates,logits=predict(net,batch)
        original=pit_waveform_loss(estimates,batch['references'],batch['active'],batch['mixture'])
        nmse,coherence=core.aligned_terms(estimates,batch['references'],batch['active'],batch['mixture'],original['assignment'])
        torch.testing.assert_close((nmse+coherence).mean(),original['source_loss'],rtol=1e-4,atol=1e-6)
        ce=.1*F.cross_entropy(logits,batch['construction_count']-1)
        losses={'original':original['loss']+ce,
                'log1p_nmse_fixed_original_PIT':(torch.log1p(nmse)+coherence).mean()+original['background_loss']+ce}
        row=dict(index=index,count=int(item['construction_count']),categories=metadata[index]['categories'],
                 gap_db=metadata[index]['gap_db'],losses={},head_gradient_norms={})
        for n,(mode,loss) in enumerate(losses.items()):
            grads=torch.autograd.grad(loss,parameters,retain_graph=n==0)
            vector=torch.cat([g.detach().flatten() for g in grads]).numpy().copy()
            if not np.isfinite(vector).all() or not torch.isfinite(loss):
                raise ValueError('Nonfinite gradient or loss')
            vectors[mode].append(vector);row['losses'][mode]=float(loss.detach())
            row['head_gradient_norms'][mode]=float(np.linalg.norm(vector.astype(np.float64)))
        rows.append(row);watch.write(root/'PARTIAL.json',dict(rows=rows,partial=True))
        watch.write(root/'STATE.json',dict(status='CPU_UNET_HEAD_GRADIENT_DIAGNOSIS',cases=len(rows),total=32,
                    seconds=time.time()-began,pid=os.getpid(),time=time.time()))
        del item,batch,estimates,logits,original,nmse,coherence,ce,losses,loss,grads,vector
    np.savez(root/'HEAD_GRADIENTS.npz',**{k:np.stack(v) for k,v in vectors.items()})
    summaries=[core.summarize(rows,vectors[k],k) for k in protocol['modes']]
    for rel,digest in source.items():
        if watch.digest(watch.ROOT/rel)!=digest or watch.digest(root/'source_snapshot'/rel)!=digest:
            raise ValueError('Source changed during audit')
    result=dict(protocol,status='COMPLETE',protocol_sha256=watch.digest(root/'PROTOCOL.json'),rows=rows,
        summary=summaries,gradient_vectors_sha256=watch.digest(root/'HEAD_GRADIENTS.npz'),seconds=time.time()-began)
    watch.write(root/'COMPLETE.json',result);watch.write(public.with_suffix('.json'),result)
    lines=['# 선택 U-Net e4: 같은 극단 batch의 출력층 기울기', '',
        '원 규모 U-Net e4/300업데이트 가중치를 고정했다. 앞선 문맥 모델 진단과 같은 TRAIN batch32를 '
        '사용하되, 아래 해석은 이 U-Net 안에서의 기울기 집중도에 한정한다. 모델·출력층 크기·학습량이 '
        '다르므로 서로 다른 모델의 gradient norm 크기를 우열로 비교하지 않는다.', '',
        '| 손실 진단 | 최대 사례의 개별 norm 합 대비 비율 | 상위 5사례 비율 | 최대 사례와 batch 합의 cosine |',
        '|---|---:|---:|---:|']
    for s in summaries:
        lines.append(f"|{s['mode']}|{s['largest_case_norm_fraction']:.6f}|{s['top5_norm_fraction']:.6f}|{s['largest_case_cosine_with_batch_sum']:.6f}|")
    lines+=['','출력층만 미분했고 optimizer 업데이트는 0회다. 원래 PIT 대응을 고정한 log1p 진단은 '
        '이 손실로 새 모델을 학습한 결과가 아니다. 선택된 극단 batch의 결과를 전체 학습의 기여율이나 '
        '실제 gradient clipping 이력으로 일반화하지 않는다.','']
    watch.write(public.with_suffix('.md'),'\n'.join(lines))
    watch.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
    print(json.dumps(dict(status='COMPLETE',summary=summaries,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('study','prerequisite','run','public'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    try:run(a.study.resolve(),a.prerequisite.resolve(),a.run.resolve(),a.public.resolve())
    except Exception:
        a.run.mkdir(parents=True,exist_ok=True)
        watch.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
