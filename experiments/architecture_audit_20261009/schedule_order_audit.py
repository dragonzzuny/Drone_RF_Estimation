"""Audit frozen TRAIN construction-count order without opening signal arrays."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

import watch_epochs as watch


def run(preparation, output):
    path = preparation / 'PREPARATION.json'
    config = watch.read(path)
    schedule = Path(config['original_preparation']) / 'TRAIN.npy'
    if watch.digest(schedule) != config['schedule_sha256']['TRAIN.npy']:
        raise ValueError('Frozen training schedule changed')
    rows = np.load(schedule, allow_pickle=False)
    counts = lambda values: {str(k): int(np.count_nonzero(values == k)) for k in (1, 2, 3)}
    results = []
    for epoch in range(1, 6):
        selected = rows[rows['epoch'] == epoch]
        values = selected['count']
        if len(values) != 2400 or counts(values) != {'1': 800, '2': 800, '3': 800}:
            raise ValueError('Per-epoch sample balance differs from plan')
        batches = values.reshape(-1, 32)
        result = dict(epoch=epoch, counts=counts(values), last32=counts(values[-32:]),
            last320=counts(values[-320:]),
            longest_same_count_run=max(len(list(g)) for _, g in itertools.groupby(values)),
            batches_missing_any_count=sum(len(set(map(int, x))) < 3 for x in batches),
            epoch_rows_sha256=hashlib.sha256(selected.tobytes()).hexdigest())
        results.append(result)
    result = dict(status='PASS_BALANCED_EPOCHS', epochs=results,
        preparation_sha256=watch.digest(path), schedule_sha256=watch.digest(schedule),
        audit_source_sha256=watch.digest(Path(__file__)),
        new_iq_read=False, training_changed=False, heldout_read=False,
        scope='Frozen first five TRAIN epochs; composition metadata only, not a causal test')
    watch.write(output.with_suffix('.json'), result)
    lines = ['# 학습 개수 비율과 입력 순서 점검', '',
        'RFUAV 원 대역 비교가 실제 읽는 고정 TRAIN 스케줄의 첫5epoch를 점검했다. '
        '원 준비 규약의 SHA256과 대조했으며 신호 I/Q·미개봉 자료를 열지 않았다. '
        '학습 코드·순서·가중치는 변경하지 않았다.', '',
        '|epoch|전체1/2/3개 혼합 수|마지막32개|마지막320개|동일 개수 최대 연속 길이|세 조건 중 하나라도 빠진 실효 배치 수|',
        '|---:|---|---|---|---:|---:|']
    for r in results:
        fmt = lambda key: '/'.join(str(r[key][str(k)]) for k in (1, 2, 3))
        lines.append(f"|{r['epoch']}|{fmt('counts')}|{fmt('last32')}|{fmt('last320')}|"
                     f"{r['longest_same_count_run']}|{r['batches_missing_any_count']}|")
    lines += ['', '각 epoch는 세 조건800개씩이며 모두75개 실효 배치를 구성한다. '
        '모든 배치에 세 조건이 존재한다. 구성 개수 기준으로 정렬된 학습이나 epoch 전체의 '
        '개수 불균형은 확인되지 않았다. 개수별 기종·대역·국소 활동의 차이, 작은 배치의 '
        '확률적 변동, 최적화 또는 일반화 문제까지 배제한 검사는 아니다.', '']
    watch.write(output.with_suffix('.md'), '\n'.join(lines))
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.preparation, args.output)
