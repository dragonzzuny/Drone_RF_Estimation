"""CPU-only fixed TRAIN48 fit diagnosis of the parent and both additional e1s.

The selection rule precedes waveform reads, reusing the existing TRAIN48 rule.
No updates, no model selection, no reserved confirmation data. Three previously
evaluated development cases per model check CPU/GPU metric reproduction only.
"""
import argparse
from collections import Counter
import gc
import os
from pathlib import Path
import statistics as stats
import time
import traceback

import numpy as np
import torch
import train_comparison as worker
from drone_rf.waveform import waveform_metrics

w = worker.watch


@torch.inference_mode()
def evaluate(net, data, index):
    item = data[index]
    batch = {k: torch.as_tensor(np.asarray(item[k])[None]) for k in
             ('mixture', 'references', 'active', 'context_features', 'crop_start')}
    estimates, logits = worker.predict(net, batch)
    score = waveform_metrics(estimates, batch['references'], batch['active'], batch['mixture'])
    n = item['construction_count']
    nmse, si = score['nmse'][0, :n].tolist(), score['si_sdr'][0, :n].tolist()
    if not np.isfinite(nmse + si).all() or float(score['sum_relative_error'].max()) > 1e-9:
        raise ValueError('Invalid CPU waveform metrics')
    row = data.rows[index]
    clips = [data.library.clips[int(i)] for i in row['indices'][:n]]
    power = score['reference_power'][0, :n].tolist()
    return dict(index=index, count=n, categories=[c['category'] for c in clips],
        nmse=nmse, si_sdr=si, reference_power=power, weakest_index=int(np.argmin(power)),
        predicted_count=int(logits.argmax(-1)[0]) + 1,
        sum_relative_error=float(score['sum_relative_error'].max()))


def run(study, root, public):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Duplicate CPU diagnostic')
    p = w.read(study / 'PROTOCOL.json')
    train = worker.NativeMixtures(p['preparation'], 'train_pack', 1)
    used, indices = Counter(), []
    for i, row in enumerate(train.rows):
        n = int(row['count'])
        names = tuple(train.library.clips[int(j)]['category'] for j in row['indices'][:n])
        key = names + tuple(map(float, row['levels'][:n]))
        if used[key] < 2:
            used[key] += 1
            indices.append(i)
    if len(indices) != 48:
        raise ValueError('Existing TRAIN48 selection rule changed')
    sources = dict(p['source_sha256'])
    sources[str(Path(__file__).relative_to(w.ROOT))] = w.digest(Path(__file__))
    registration = dict(status='REGISTERED_CPU_TRAIN48_FIT', source_sha256=sources,
        study_protocol_sha256=w.digest(study / 'PROTOCOL.json'),
        parent_checkpoint_sha256=p['parent_checkpoint_sha256'], preparation_sha256=p['preparation_sha256'],
        train_indices=indices, selection='First two TRAIN e1 per ordered category tuple and nominal levels',
        checkpoints=['parent/e0', 'retained_unet/e1', 'source_interaction/e1'],
        source_epoch_receipts_required=True, model_updates=0, heldout_read=False,
        development_waveform_use='First stored case per count/model, CPU/GPU reproduction only',
        inference='Full CPU FP32 models, two threads, mixture and its own context only',
        interpretation='Already seen training fit, not independent generalization or matched TRAIN/VAL distribution',
        registered_at=time.time())
    w.write(root / 'PROTOCOL.json', registration)
    validation = worker.NativeMixtures(p['preparation'], 'validation_pack', 1)
    records, checks, hashes, head_changes = [], [], {}, {}
    began = time.time()
    for label in registration['checkpoints']:
        arm, epoch_label = label.split('/')
        if arm == 'parent':
            path = Path(p['parent_checkpoint'])
            reference_path = Path(p['validation_identity_template'])
        else:
            receipt_path = study / arm / 'EPOCH_001.json'
            while not receipt_path.exists():
                if (study / 'FAILURE.json').exists() or time.time() - began > 4 * 3600:
                    raise RuntimeError('Main study failed or e1 wait expired')
                w.write(root / 'STATE.json', dict(status='WAITING_COMPLETED_E1', arm=arm,
                    pid=os.getpid(), time=time.time()))
                time.sleep(30)
            receipt = w.read(receipt_path)
            if receipt['protocol_sha256'] != registration['study_protocol_sha256'] or receipt['updates'] != 75:
                raise ValueError('Wrong additional epoch')
            path = study / arm / 'ACTUAL_001.pt'
            reference_path = study / arm / 'VALIDATION_001.json'
        hashes[label] = dict(checkpoint=w.digest(path), validation=w.digest(reference_path))
        if arm == 'parent' and hashes[label]['checkpoint'] != p['parent_checkpoint_sha256']:
            raise ValueError('Parent changed')
        for rel, sha in sources.items():
            if w.digest(w.ROOT / rel) != sha:
                raise ValueError('Frozen source changed')
        net = worker.make_model('retained_unet' if arm == 'parent' else arm).eval()
        if arm == 'source_interaction':
            initial_head = {k: v.detach().clone() for k, v in net.output.state_dict().items()
                            if not k.startswith('base.')}
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if arm != 'parent' and (saved['arm'] != arm or saved['epoch'] != 1 or
                saved['protocol_sha256'] != registration['study_protocol_sha256']):
            raise ValueError('Checkpoint identity mismatch')
        net.load_state_dict(saved['model'])
        del saved
        if arm == 'source_interaction':
            head_changes = {k: dict(initial_norm=float(initial_head[k].norm()),
                final_norm=float(v.norm()), delta_norm=float((v - initial_head[k]).norm()))
                for k, v in net.output.state_dict().items() if k in initial_head}
            del initial_head
        expected = w.read(reference_path)['rows']
        for count in (1, 2, 3):
            ref = next(r for r in expected if r['count'] == count)
            actual = evaluate(net, validation, ref['index'])
            np.testing.assert_allclose(actual['nmse'], ref['nmse'], rtol=1e-3, atol=1e-5)
            np.testing.assert_allclose(actual['si_sdr'], ref['si_sdr'], rtol=0, atol=.01)
            checks.append(dict(model=label, index=ref['index'], passed=True))
        for index in indices:
            records.append(dict(model=label, **evaluate(net, train, index)))
            w.write(root / 'PARTIAL.json', dict(rows=records, partial=True))
            w.write(root / 'STATE.json', dict(status='CPU_TRAIN_INFERENCE', model=label,
                cases=len(records), total=144, pid=os.getpid(), time=time.time()))
        if w.digest(path) != hashes[label]['checkpoint']:
            raise ValueError('Scored checkpoint changed')
        del net
        gc.collect()
    summaries = []
    for label in registration['checkpoints']:
        for count in (1, 2, 3):
            rows = [r for r in records if r['model'] == label and r['count'] == count]
            summaries.append(dict(model=label, count=count, cases=len(rows),
                mean_nmse=stats.mean(v for r in rows for v in r['nmse']),
                mean_si_sdr=stats.mean(v for r in rows for v in r['si_sdr']),
                weakest_nmse=stats.mean(r['nmse'][r['weakest_index']] for r in rows),
                all_nmse_below_point_one=sum(all(v < .1 for v in r['nmse']) for r in rows)))
    result = dict(status='COMPLETE', diagnostic_protocol_sha256=w.digest(root / 'PROTOCOL.json'),
        study_protocol_sha256=registration['study_protocol_sha256'], source_sha256=sources,
        checkpoint_validation_sha256=hashes, train_indices=indices, summaries=summaries,
        rows=records, backend_checks=checks, added_head_weight_changes=head_changes,
        model_updates=0, heldout_read=False, independent_test=False, seconds=time.time() - began)
    w.write(root / 'COMPLETE.json', result)
    w.write(public.with_suffix('.json'), result)
    lines = ['# 보존 U-Net과 추가 e1: 같은 TRAIN48 적합도', '',
        '기존 선택 모델과 두 군의 추가 e1을 고정하고 이미 학습한 같은 48혼합을 CPU로 추론했다. '
        '기종·명목 전력 조건별 첫 두 행이라는 기존 규칙을 파형 읽기 전에 등록했다. '
        '각 모델마다 기존 개발검증 1/2/3성분의 첫 행을 CPU/GPU 점수 재현에만 사용했다. '
        '새 학습·선택·보류 자료 접근은 없다.', '',
        '| 모델 | 성분 수 | 혼합 수 | NMSE ↓ | 복소 SI-SDR ↑ dB | 약신호 NMSE ↓ | 모든 성분 NMSE<0.1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['model']} | {r['count']} | {r['cases']} | {r['mean_nmse']:.6f} | "
            f"{r['mean_si_sdr']:.3f} | {r['weakest_nmse']:.6f} | {r['all_nmse_below_point_one']}/{r['cases']} |")
    lines += ['', 'TRAIN48과 개발검증630은 기종 구성·전력 분포·원기록 조건이 다르다. 평균을 빼서 '
        '일반화 격차나 과적합의 정량값으로 사용하지 않는다. 학습 사례의 적합도와 분리 단서의 학습 '
        '가능성만 진단한다. 추가 출력층 가중치 변화는 JSON에 기록했으며 인과적 성능 기여율을 뜻하지 않는다.', '']
    w.write(public.with_suffix('.md'), '\n'.join(lines))
    w.write(root / 'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))
    print(dict(status='COMPLETE', summaries=summaries), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('study', 'run', 'public'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        run(args.study.resolve(), args.run.resolve(), args.public.resolve())
    except Exception:
        w.write(args.run / 'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
