"""CPU-only checkpoint-change diagnosis; no model inference or optimizer steps."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(root):
    torch.set_num_threads(2)
    study = root / 'local/septda_rf_20261010_v1'
    public = root / 'reports/2026-10-10'
    audit = json.loads((public / 'SEPTDA_RF_AUDIT.json').read_text())
    assert audit['status'] == 'PASS'
    paths = [study / x for x in ('INITIAL.pt', 'ACTUAL_001.pt', 'EPOCH_001.json')]
    hashes = {p.name: digest(p) for p in paths}
    assert all(hashes[k] == audit['hashes'][k] for k in hashes)
    initial = torch.load(paths[0], map_location='cpu', weights_only=False, mmap=True)['model']
    final = torch.load(paths[1], map_location='cpu', weights_only=False, mmap=True)['model']
    assert initial.keys() == final.keys()
    groups = {}
    for key, before in initial.items():
        after = final[key]
        assert before.shape == after.shape and before.dtype == after.dtype
        assert torch.isfinite(before).all() and torch.isfinite(after).all()
        group = key.split('.')[1] if key.startswith('septda.') else 'parent'
        row = groups.setdefault(group, dict(tensors=0, changed_tensors=0, elements=0,
                                            initial_squared_norm=0., delta_squared_norm=0.))
        delta = after.double() - before.double()
        row['tensors'] += 1
        row['changed_tensors'] += int(not torch.equal(before, after))
        row['elements'] += before.numel()
        row['initial_squared_norm'] += float(before.double().square().sum())
        row['delta_squared_norm'] += float(delta.square().sum())
    for row in groups.values():
        row['delta_l2'] = math.sqrt(row['delta_squared_norm'])
        row['relative_delta_l2'] = (math.sqrt(row['delta_squared_norm'] / row['initial_squared_norm'])
                                   if row['initial_squared_norm'] > 0 else None)
    event = json.loads(paths[2].read_text())
    history = event['branch_history']
    assert event['updates'] == len(history) == 75
    assert [r['update'] for r in history] == list(range(1, 76))
    gradients = {}
    for key in ('query_gradient_norm', 'readout_gradient_norm'):
        values = [r[key] for r in history]
        assert all(math.isfinite(x) and x >= 0 for x in values)
        gradients[key] = dict(first=values[0], nonzero_updates=sum(x > 0 for x in values),
                              first10_median=statistics.median(values[:10]),
                              last10_median=statistics.median(values[-10:]))
    assert gradients['query_gradient_norm']['nonzero_updates'] == 74
    assert gradients['readout_gradient_norm']['nonzero_updates'] == 75
    assert all(v['delta_l2'] > 0 for v in groups.values())
    assert not torch.cuda.is_initialized()
    out = dict(status='CHECKED', checkpoint_hashes=hashes, reporter_sha256=digest(Path(__file__)),
        groups=groups, preclip_gradient_history=gradients,
        clipped_updates=sum(x > 1 for x in event['preclip_gradient_norms']),
        updates=75, model_inference=0, iq_reads=0, new_optimizer_steps=0, cuda_initialized=False,
        heldout_read=False,
        interpretation='Nonzero parameter change and gradient connectivity are not proof of useful separation. '
        'Parameter changes include weight decay. Group norms are not comparable across differently sized groups. '
        'Zero-initialized readouts have no defined relative change; report absolute L2.')
    (public / 'SEPTDA_RF_LEARNING_SIGNAL.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    lines = ['# SepTDA 참고 경로의 실제 학습 변화 검사', '',
        '완료한 e1의 초기·실제 가중치를 CPU에서 비교했다. 추가 학습·파형 읽기·추론은 없다. '
        '기존 완료 검산의 SHA-256과 모두 일치한다.', '',
        '| 부분 | 변경 tensor / 전체 | 가중치 변화 L2 | 초기 대비 상대 L2 |',
        '|---|---:|---:|---:|']
    for key, row in groups.items():
        relative = '초기 0: 미정의' if row['relative_delta_l2'] is None else f"{row['relative_delta_l2']:.6g}"
        lines.append(f"|{key}|{row['changed_tensors']}/{row['tensors']}|{row['delta_l2']:.6g}|{relative}|")
    lines += ['', 'query gradient는 첫 단계의 0 readout 때문에 0이며 이후74단계에서 비영이었다. '
        '출력층 gradient는75단계 모두 비영이었다. 모든 주요 경로에 실제 가중치 변화가 있었다.', '',
        '이 검사는 완전히 끊기거나 갱신되지 않은 경로라는 가설을 지지하지 않는다. '
        '유용한 특징을 배웠는지와 출력의 개선은 별개다. 가중치 변화에는 weight decay도 포함되고 '
        '크기가 다른 모듈의 norm을 학습 효과 순위로 해석하지 않는다. '
        '초기 가중치가0인 출력층에 상대 변화율을 부여하지 않는다.', '',
        '[수치와 지문](SEPTDA_RF_LEARNING_SIGNAL.json) · '
        '[고정 TRAIN48의 실제 출력 변화](SEPTDA_RF_TRAIN_DIAGNOSIS_KO.md)']
    (public / 'SEPTDA_RF_LEARNING_SIGNAL_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=out['status'], groups=len(groups), gradients=gradients,
                         clipped_updates=out['clipped_updates']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    run(parser.parse_args().root)
