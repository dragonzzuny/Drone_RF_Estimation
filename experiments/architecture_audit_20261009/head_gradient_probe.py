"""Post-hoc output-head gradient audit on one exact scheduled TRAIN batch.

No optimizer update. The frozen long e2 model, original PIT correspondence,
and observed-input whitelist are preserved. Compare original NMSE against a
diagnostic log1p(NMSE) reweighting at the SAME fixed correspondence. Head-only
gradients are not whole-network gradients or a trained robust-loss result.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

import phase_data as data
import phase_packing as pp
import watch_epochs as watch
from native_data import NativeMixtures
from drone_rf.losses import pit_waveform_loss


def aligned_terms(estimates, references, active, mixture, assignment):
    chosen = estimates[:, :3].gather(1, assignment[:, :, None].expand(-1, -1, estimates.shape[-1]))
    mix_power = mixture.abs().square().mean(-1).clamp_min(1e-8)
    ref_power = references.abs().square().mean(-1)
    denominator = torch.where(active, torch.maximum(ref_power, 1e-6*mix_power[:, None]), mix_power[:, None])
    nmse = (chosen-references).abs().square().mean(-1)/denominator
    ec, rc = chosen-chosen.mean(-1, keepdim=True), references-references.mean(-1, keepdim=True)
    cross = (ec*rc.conj()).mean(-1).abs().square()
    variance = ec.abs().square().mean(-1)*rc.abs().square().mean(-1)
    coherence = (1-(cross/variance.clamp_min(1e-16)).clamp(0, 1)) * (active & (ref_power>1e-6*mix_power[:, None]))
    return nmse, coherence


def summarize(rows, vectors, name):
    gradients = np.stack(vectors).astype(np.float64)
    norms = np.linalg.norm(gradients, axis=1)
    summed = gradients.sum(0)
    total_norm = float(np.linalg.norm(summed))
    top = int(norms.argmax())
    cosine = float(np.dot(gradients[top],summed)/(norms[top]*total_norm)) if norms[top]*total_norm>0 else None
    order = np.argsort(-norms)[:5]
    return dict(mode=name, batch_cases=32, head_parameters=gradients.shape[1],
        mean_gradient_norm=total_norm/32, sum_individual_norms=float(norms.sum()),
        largest_case_norm_fraction=float(norms[top]/norms.sum()) if norms.sum()>0 else None,
        largest_case_index=rows[top]['index'], largest_case_gap_db=rows[top]['gap_db'],
        largest_case_cosine_with_batch_sum=cosine,
        top5_norm_fraction=float(norms[order].sum()/norms.sum()) if norms.sum()>0 else None,
        top5_indices=[rows[int(i)]['index'] for i in order])


def run(study, power_audit, root, public):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate gradient audit')
    root.mkdir(parents=True,exist_ok=True)
    parent = watch.read(study/'PROTOCOL.json')
    power = watch.read(power_audit/'COMPLETE.json')
    path = power_audit/'TRAIN_ROWS.jsonl'
    if watch.digest(path)!=power['train_rows_sha256']:
        raise ValueError('Power metadata changed')
    power_rows=[json.loads(line) for line in path.read_text().splitlines()]
    by_index={r['index']:r for r in power_rows if r['epoch']==1}
    worst=max(by_index.values(),key=lambda r:r['gap_db'])
    start=worst['index']//32*32
    indices=list(range(start,start+32))
    checkpoint=study/'long/ACTUAL_002.pt'
    source=dict(parent['source_sha256'])
    source[str(Path(__file__).relative_to(watch.ROOT))]=watch.digest(Path(__file__))
    for rel,digest in source.items():
        if watch.digest(watch.ROOT/rel)!=digest:
            raise ValueError('Frozen source changed')
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(watch.ROOT/rel,dest)
    protocol=dict(status='REGISTERED_HEAD_ONLY_GRADIENT_DIAGNOSIS',source_sha256=source,
        study_protocol_sha256=watch.digest(study/'PROTOCOL.json'),checkpoint_sha256=watch.digest(checkpoint),
        checkpoint_arm='long',checkpoint_epoch=2,checkpoint_updates=150,
        preparation_sha256=parent['preparation_sha256'],power_metadata_sha256=watch.digest(path),
        power_audit_sha256=watch.digest(power_audit/'COMPLETE.json'),train_indices=indices,
        selection='exact epoch1 batch32 containing its largest local reference-power gap',
        selected_anchor_index=worst['index'],selected_anchor_gap_db=worst['gap_db'],
        modes=['original','log1p_nmse_fixed_original_PIT'],
        derivative_scope='final Conv1d output weight and bias only; all other parameters frozen',
        optimizer_updates=0,validation_read=False,heldout_read=False,post_hoc=True,
        caveat='Not actual historical batch gradients, global gradient clipping, or trained alternative-loss performance')
    watch.write(root/'PROTOCOL.json',protocol)
    train=NativeMixtures(parent['preparation'],'train_pack',1)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['epoch']!=2 or saved['arm']!='long' or saved['protocol_sha256']!=protocol['study_protocol_sha256']:
        raise ValueError('Wrong checkpoint')
    net=pp.PhasePackedWaveNet('long').eval();net.load_state_dict(saved['model']);del saved
    net.requires_grad_(False);net.net.output.requires_grad_(True)
    parameters=tuple(net.net.output.parameters())
    vectors={name:[] for name in protocol['modes']}
    rows=[];began=time.time()
    for index in indices:
        item=data.batch(data.example(train,index),'cpu')
        predicted=data.predict(net,item)
        original=pit_waveform_loss(predicted['estimates'],item['references'],item['active'],item['mixture'])
        nmse,coherence=aligned_terms(predicted['estimates'],item['references'],item['active'],item['mixture'],original['assignment'])
        torch.testing.assert_close((nmse+coherence).mean(),original['source_loss'],rtol=1e-4,atol=1e-6)
        ce=.1*F.cross_entropy(predicted['count_logits'],item['construction_count']-1)
        losses={'original':original['loss']+ce,
                'log1p_nmse_fixed_original_PIT':(torch.log1p(nmse)+coherence).mean()+original['background_loss']+ce}
        row=dict(index=index,count=int(item['construction_count']),categories=by_index[index]['categories'],
                 gap_db=by_index[index]['gap_db'],losses={},head_gradient_norms={})
        for n,(name,loss) in enumerate(losses.items()):
            values=torch.autograd.grad(loss,parameters,retain_graph=n==0)
            vector=torch.cat([v.detach().flatten() for v in values]).numpy().copy()
            if not np.isfinite(vector).all() or not torch.isfinite(loss):
                raise ValueError('Nonfinite head gradient or loss')
            vectors[name].append(vector)
            row['losses'][name]=float(loss.detach())
            row['head_gradient_norms'][name]=float(np.linalg.norm(vector.astype(np.float64)))
        rows.append(row)
        watch.write(root/'PARTIAL.json',dict(rows=rows,partial=True))
        watch.write(root/'STATE.json',dict(status='CPU_HEAD_GRADIENT_DIAGNOSIS',cases=len(rows),total=32,
                    seconds=time.time()-began,pid=os.getpid(),time=time.time()))
        del item,predicted,original,nmse,coherence,ce,losses,loss,values,vector
    np.savez(root/'HEAD_GRADIENTS.npz',**{k:np.stack(v) for k,v in vectors.items()})
    summaries=[summarize(rows,vectors[k],k) for k in protocol['modes']]
    for rel,digest in source.items():
        if watch.digest(watch.ROOT/rel)!=digest or watch.digest(root/'source_snapshot'/rel)!=digest:
            raise ValueError('Frozen source changed during diagnosis')
    result=dict(protocol,status='COMPLETE',protocol_sha256=watch.digest(root/'PROTOCOL.json'),
        rows=rows,summary=summaries,gradient_vectors_sha256=watch.digest(root/'HEAD_GRADIENTS.npz'),
        seconds=time.time()-began)
    watch.write(root/'COMPLETE.json',result);watch.write(public.with_suffix('.json'),result)
    lines=['# 출력층 기울기: 큰 전력차를 포함한 고정 학습 batch 진단', '',
        '고정 긴 문맥군 e2/150업데이트에서 첫 epoch의 최대 전력차가 포함된 실제 batch32를 재계산했다. '
        '출력층 weight·bias만 미분했으며 모델을 갱신하지 않았다. 원래 PIT 대응을 두 손실에서 그대로 '
        '유지했다. 따라서 새 손실로 PIT와 모델을 함께 학습한 실험이 아니다.', '',
        '| 손실 진단 | 최대 사례의 개별 norm 합 대비 비율 | 상위 5사례 비율 | 최대 사례와 batch 합의 cosine |',
        '|---|---:|---:|---:|']
    for s in summaries:
        lines.append(f"|{s['mode']}|{s['largest_case_norm_fraction']:.6f}|{s['top5_norm_fraction']:.6f}|{s['largest_case_cosine_with_batch_sum']:.6f}|")
    lines+=['','이 비율은 개별 출력층 기울기 norm들의 합을 기준으로 한다. 전체 모델 기울기의 기여율이나 '
        '실제 gradient clipping 직전 상태로 해석하지 않는다. 선택된 극단 batch의 사후 진단이므로 '
        '전체 학습 batch로 일반화하지 않는다. log1p 가중 변경의 분리 성능은 아직 학습·검증하지 않았다.','']
    watch.write(public.with_suffix('.md'),'\n'.join(lines))
    watch.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
    print(json.dumps(dict(status='COMPLETE',summary=summaries,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('study','power-audit','run','public'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    try:run(a.study.resolve(),a.power_audit.resolve(),a.run.resolve(),a.public.resolve())
    except Exception:
        a.run.mkdir(parents=True,exist_ok=True)
        watch.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
