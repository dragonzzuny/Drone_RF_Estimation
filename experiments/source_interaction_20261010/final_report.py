"""Report all epochs and e0-inclusive choices after the independent audit."""
import argparse
from pathlib import Path
import train_comparison as worker

w = worker.watch


def run(study, audit_path, output):
    p = w.read(study / 'PROTOCOL.json')
    worker.verify(study, p)
    audit = w.read(audit_path)
    if audit['status'] != 'PASS' or audit['study_protocol_sha256'] != w.digest(study / 'PROTOCOL.json') or audit['complete_sha256'] != w.digest(study / 'COMPLETE.json'):
        raise ValueError('Completed independently audited study required')
    parent, identities = w.validate(Path(p['validation_identity_template']), None)
    histories, selected = {}, {}
    for arm in worker.ARMS:
        histories[arm] = []
        for epoch in range(4):
            result, _ = w.validate(study / arm / f'VALIDATION_{epoch:03d}.json', identities)
            histories[arm].append(result)
        selected[arm] = min(histories[arm], key=lambda r: r['selection_nmse'])
        if selected[arm]['epoch'] != audit['results'][arm]['selected']['epoch']:
            raise ValueError('Independent selection differs')
    comparisons = []
    candidate = selected['source_interaction']
    for label, reference in [('retained_parent', parent), ('matched_retained_unet', selected['retained_unet'])]:
        for n in (2, 3):
            a, b = [next(g for g in value['by_count'] if g['count'] == n) for value in (reference, candidate)]
            comparisons.append(dict(reference=label, count=n,
                nmse_delta=b['mean_nmse'] - a['mean_nmse'], si_sdr_delta=b['mean_si_sdr'] - a['mean_si_sdr'],
                weakest_nmse_delta=b['weakest_nmse'] - a['weakest_nmse'],
                raw_nmse_strictly_lower=b['mean_nmse'] < a['mean_nmse'],
                complex_si_sdr_strictly_higher=b['mean_si_sdr'] > a['mean_si_sdr']))
    acceptance = candidate['epoch'] > 0 and all(r['raw_nmse_strictly_lower'] and r['complex_si_sdr_strictly_higher'] for r in comparisons)
    result = dict(status='COMPLETE', report_source_sha256=w.digest(Path(__file__)),
        study_protocol_sha256=w.digest(study / 'PROTOCOL.json'), final_audit_sha256=w.digest(audit_path),
        parent=parent, histories=histories, selected=selected, selected_comparisons=comparisons,
        direction_criterion_met=acceptance, completed_epochs_per_arm=3, optimizer_updates_per_arm=225,
        weak_and_single_source_are_reported_not_hidden=True, heldout_read=False, independent_test=False,
        all_results_are_development=True, one_seed=True)
    w.write(output.with_suffix('.json'), result)
    lines = ['# 성분 상호작용 출력층: 완료 대조', '',
        '같은 보존 U-Net 부모·native RFUAV 자료·원 손실·새 AdamW 1e-5·seed 0·각 추가 '
        '3 epoch/225업데이트다. 실효 batch 32·microbatch 1이며 원래 본체를 모두 학습했다. '
        '후보만 37,888개 파라미터를 추가했다. 모든 수치는 같은 630 개발 혼합의 단일 추론 결과다.', '',
        '## 사전 선택 규칙의 결과', '',
        'e0를 포함한 2·3성분 평균 raw NMSE 최소값으로 각 군을 선택했다. 마지막 실제 모델과 구분한다.', '',
        '| 모델 | 선택 추가 epoch | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB | 약신호 NMSE 2/3 ↓ |',
        '|---|---:|---|---|---|']
    for name, r in [('retained_parent', parent), *selected.items()]:
        a, b = [next(g for g in r['by_count'] if g['count'] == n) for n in (2, 3)]
        epoch = 0 if name == 'retained_parent' else r['epoch']
        lines.append(f"| {name} | {epoch} | {a['mean_nmse']:.6f}/{b['mean_nmse']:.6f} | {a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f} | {a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f} |")
    lines += ['', '동일 예산 대조와 부모 대비 두 신호 수의 NMSE 감소·복소 SI-SDR 증가라는 방향 기준: **' +
        ('충족' if acceptance else '미충족') + '**. 한 seed의 개발 선택 결과이며 일반적인 우월성의 입증과 구분한다.', '',
        '## 선택되지 않은 epoch까지 포함한 전체 결과', '',
        '| 군 | 추가 epoch | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB |', '|---|---:|---|---|']
    for epoch in range(4):
        for arm in worker.ARMS:
            r = histories[arm][epoch]
            a, b = [next(g for g in r['by_count'] if g['count'] == n) for n in (2, 3)]
            lines.append(f"| {arm} | {epoch} | {a['mean_nmse']:.6f}/{b['mean_nmse']:.6f} | {a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f} |")
    lines += ['', '## 선택 모델의 1성분 보존과 개수 추정', '',
        '| 군 | 1성분 NMSE ↓ | 1성분 복소 SI-SDR ↑ | 개수 정확도 1/2/3 |', '|---|---:|---:|---|']
    for arm, r in selected.items():
        a = next(g for g in r['by_count'] if g['count'] == 1)
        accuracies = '/'.join(f"{g['construction_count_accuracy']:.3f}" for g in r['by_count'])
        lines.append(f"| {arm} | {a['mean_nmse']:.6f} | {a['mean_si_sdr']:.3f} | {accuracies} |")
    lines += ['', '합성 기록 수의 추정이며 물리 드론 대수 정답이 아니다. 원기록 묶음은 학습 8·개발 5개이고 '
        '세 성분의 기종 조합은 하나다. 원기록 변화와 대역폭 변화가 함께 있다. 반복 개발 선택과 '
        '한 seed의 한계를 유지하며 Autel·예약 확인 파일은 읽지 않았다.', '',
        '최종 검산은 실제/선택 체크포인트 전체 tensor, optimizer 225단계·학습률, 모든 epoch의 '
        '630행, 자료 및 동결 소스의 일치를 확인했다.', '',
        '[검산](SOURCE_INTERACTION_FINAL_AUDIT.json) · [전체 곡선 PDF](SOURCE_INTERACTION_EPOCHS.pdf) · '
        '[조건별 대응 비교](SOURCE_INTERACTION_PAIRED_DIAGNOSTICS.md) · '
        '[학습 적합도·전력 배분 진단](SOURCE_INTERACTION_E1_DIAGNOSIS_KO.md) · '
        '[후속 학습률 검사 규약](SOURCE_HEAD_FOLLOWUP_PLAN_KO.md)', '']
    w.write(output.with_suffix('.md'), '\n'.join(lines))
    print(dict(status='COMPLETE', selected_epochs={k: v['epoch'] for k, v in selected.items()},
               direction_criterion_met=acceptance))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('study', 'audit', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.study.resolve(), args.audit.resolve(), args.output.resolve())
