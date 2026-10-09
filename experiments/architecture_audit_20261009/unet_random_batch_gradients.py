"""Four prospectively sampled TRAIN batches, frozen U-Net output-head audit.

Sampling uses only the registered 75 batch indices, not errors, power, or
gradients. This extends (but does not replace) the selected extreme-batch
diagnostic. It remains a head-only diagnostic with zero optimizer updates.
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


def run(study, prerequisite, power, root, public):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate random-batch registration')
    root.mkdir(parents=True, exist_ok=True)
    parent = watch.read(study/'PROTOCOL.json')
    previous = watch.read(prerequisite/'COMPLETE.json')
    power_audit = watch.read(power/'COMPLETE.json')
    metadata_path = power/'TRAIN_ROWS.jsonl'
    if watch.digest(metadata_path) != power_audit['train_rows_sha256']:
        raise ValueError('Training power metadata changed')
    seed = 20261009
    batches = sorted(np.random.Generator(np.random.PCG64(seed)).choice(75, size=4, replace=False).tolist())
    checkpoint = study/'unet_mean/SELECTED_004.pt'
    if watch.digest(checkpoint) != previous['checkpoint_sha256']:
        raise ValueError('Different selected U-Net')
    source = dict(previous['source_sha256'])
    source[str(Path(__file__).relative_to(watch.ROOT))] = watch.digest(Path(__file__))
    protocol = dict(status='REGISTERED_RANDOM_TRAIN_BATCH_HEAD_AUDIT', source_sha256=source,
        study_protocol_sha256=watch.digest(study/'PROTOCOL.json'),
        preparation_sha256=parent['preparation_sha256'],
        checkpoint_sha256=previous['checkpoint_sha256'], checkpoint_epoch=4, checkpoint_updates=300,
        parameters=32_142_859, power_metadata_sha256=watch.digest(metadata_path),
        previous_extreme_result_sha256=watch.digest(prerequisite/'COMPLETE.json'),
        sampling='uniform without replacement among all75 actual epoch1 batch indices; no error/power filtering',
        generator='NumPy PCG64', sampling_seed=seed, batch_indices=batches, batch_size=32,
        modes=previous['modes'], derivative_scope='final Conv2d weight and bias only',
        optimizer_updates=0, recorded_role='train_pack', validation_read=False, heldout_read=False,
        caveat='Four training batches; not whole-network or historical optimizer gradients', time=time.time())
    for rel, digest in source.items():
        if watch.digest(watch.ROOT/rel) != digest:
            raise ValueError('Frozen dependency changed')
        dest = root/'source_snapshot'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(watch.ROOT/rel, dest)
    watch.write(root/'PROTOCOL.json', protocol)
    all_rows = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    metadata = {r['index']:r for r in all_rows if r['epoch']==1}; del all_rows
    data = NativeMixtures(parent['preparation'], 'train_pack', 1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['best']['epoch'] != 4 or saved['protocol_sha256'] != protocol['study_protocol_sha256']:
        raise ValueError('Parent state mismatch')
    net = build('unet_mean').eval(); net.load_state_dict(saved['model']); del saved
    net.requires_grad_(False); net.output.requires_grad_(True)
    parameters = tuple(net.output.parameters()); started = time.time()
    results = []; arrays = {}; completed = 0
    for batch_index in batches:
        vectors = {mode:[] for mode in protocol['modes']}; rows = []
        for index in range(batch_index*32, (batch_index+1)*32):
            item = data[index]
            batch = {k:torch.as_tensor(np.asarray(item[k]))[None] for k in
                     ('mixture','references','active','context_features','crop_start','construction_count')}
            estimates, logits = predict(net, batch)
            original = pit_waveform_loss(estimates, batch['references'], batch['active'], batch['mixture'])
            nmse, coherence = core.aligned_terms(estimates, batch['references'], batch['active'],
                                                 batch['mixture'], original['assignment'])
            torch.testing.assert_close((nmse+coherence).mean(), original['source_loss'], rtol=1e-4, atol=1e-6)
            ce = .1*F.cross_entropy(logits, batch['construction_count']-1)
            losses = {'original':original['loss']+ce,
                      'log1p_nmse_fixed_original_PIT':(torch.log1p(nmse)+coherence).mean()+original['background_loss']+ce}
            row = dict(index=index, count=int(item['construction_count']), gap_db=metadata[index]['gap_db'],
                       losses={}, head_gradient_norms={})
            for n,(mode,loss) in enumerate(losses.items()):
                grads = torch.autograd.grad(loss, parameters, retain_graph=n==0)
                vector = torch.cat([g.detach().flatten() for g in grads]).numpy().copy()
                if not np.isfinite(vector).all() or not torch.isfinite(loss):
                    raise ValueError('Nonfinite output-head gradient')
                vectors[mode].append(vector); row['losses'][mode]=float(loss.detach())
                row['head_gradient_norms'][mode]=float(np.linalg.norm(vector.astype(np.float64)))
            rows.append(row); completed += 1
            watch.write(root/'STATE.json', dict(status='CPU_RANDOM_BATCH_HEAD_AUDIT', batch_index=batch_index,
                        cases=completed, total=128, seconds=time.time()-started, pid=os.getpid(), time=time.time()))
            del item,batch,estimates,logits,original,nmse,coherence,ce,losses,loss,grads,vector
        summaries = [core.summarize(rows, vectors[mode], mode) for mode in protocol['modes']]
        for mode, values in vectors.items():
            arrays[f'batch{batch_index}_{mode}'] = np.stack(values)
        result = dict(batch_index=batch_index, rows=rows, summary=summaries)
        results.append(result); watch.write(root/f'BATCH_{batch_index:03d}.json', result)
        print(json.dumps(dict(batch_index=batch_index, summary=summaries)), flush=True)
    np.savez(root/'HEAD_GRADIENTS.npz', **arrays)
    for rel,digest in source.items():
        if watch.digest(watch.ROOT/rel)!=digest or watch.digest(root/'source_snapshot'/rel)!=digest:
            raise ValueError('Source changed during audit')
    complete = dict(protocol, status='COMPLETE', protocol_sha256=watch.digest(root/'PROTOCOL.json'),
                    results=results, gradient_vectors_sha256=watch.digest(root/'HEAD_GRADIENTS.npz'),
                    seconds=time.time()-started)
    watch.write(root/'COMPLETE.json',complete); watch.write(public.with_suffix('.json'),complete)
    lines = ['# 무작위 TRAIN batch4: 선택 U-Net 출력층 기울기', '',
        '전체75개 학습 batch 번호에서 PCG64 seed20261009로4개를 비복원 추출하고, 파형·오차를 '
        '계산하기 전에 번호를 고정했다. 기존 극단 batch는 별도 보고로 보존한다. '
        '가중치는 U-Net e4이며 출력층만 미분했다. optimizer 업데이트는0회다.', '',
        '| batch 번호 | 손실 | 최대 사례 norm 비율 | 상위5 비율 | 최대 사례와 batch 합 cosine |',
        '|---|---|---:|---:|---:|']
    for result in results:
        for s in result['summary']:
            lines.append(f"|{result['batch_index']}|{s['mode']}|{s['largest_case_norm_fraction']:.6f}|"
                         f"{s['top5_norm_fraction']:.6f}|{s['largest_case_cosine_with_batch_sum']:.6f}|")
    lines += ['', '비율은 개별 출력층 gradient norm 합을 분모로 한다. 학습 전체의 기여율이나 '
              '전 파라미터의 실제 업데이트를 나타내지 않으며, 새 손실의 복원 개선은 GPU 대조로 별도 검증한다.', '']
    watch.write(public.with_suffix('.md'), '\n'.join(lines))
    watch.write(root/'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('study','prerequisite','power','run','public'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args(); torch.set_num_threads(2)
    try:
        run(a.study.resolve(),a.prerequisite.resolve(),a.power.resolve(),a.run.resolve(),a.public.resolve())
    except Exception:
        a.run.mkdir(parents=True,exist_ok=True)
        watch.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
