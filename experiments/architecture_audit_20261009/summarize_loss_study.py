"""Summarize audited actual/selected waveform results, with optional matched LR reference."""
import argparse
from pathlib import Path

import watch_epochs as watch


def summarize(study, audit_path, output, reference=None):
    plan, audit = watch.read(study/'PROTOCOL.json'), watch.read(audit_path)
    if (audit['status'] != 'PASS' or audit['study_protocol_sha256'] != watch.digest(study/'PROTOCOL.json')
            or audit['complete_sha256'] != watch.digest(study/'COMPLETE.json')):
        raise ValueError('Missing matching completion audit')
    parent, identities = watch.validate(Path(plan['validation_identity_template']), None)
    histories, hashes, joint = {}, {}, {}
    cases = []
    def successes(label, path):
        data = watch.read(path)
        watch.validate(path, identities)
        hashes[label] = watch.digest(path)
        for count in (2, 3):
            rows = [r for r in data['rows'] if r['count'] == count]
            cases.append(dict(model=label, count=count, cases=len(rows),
                all_nmse_below_one=sum(all(v < 1 for v in r['nmse']) for r in rows),
                all_absolute_si_positive=sum(all(v > 0 for v in r['si_sdr']) for r in rows),
                all_si_gain_positive=sum(all(v > 0 for v in r['si_sdr_gain']) for r in rows)))
    successes('parent', Path(plan['validation_identity_template']))
    for arm in plan['arms']:
        history = []
        for epoch in range(plan['epochs_per_arm']+1):
            path = study/arm/f'VALIDATION_{epoch:03d}.json'
            metrics, _ = watch.validate(path, identities)
            history.append(metrics)
            hashes[f'{arm}/e{epoch}'] = watch.digest(path)
        chosen = min(history, key=lambda r: r['selection_nmse'])
        if (chosen['epoch'] != audit['results'][arm]['selected']['epoch']
                or history[-1]['epoch'] != audit['results'][arm]['final']['epoch']):
            raise ValueError('Audit selection mismatch')
        histories[arm] = history
        source = {r['count']: r for r in chosen['by_count']}
        joint[arm] = chosen['epoch'] != 0 and all(
            source[r['count']]['mean_nmse'] < r['mean_nmse'] and
            source[r['count']]['mean_si_sdr'] > r['mean_si_sdr']
            for r in parent['by_count'] if r['count'] in (2, 3))
        successes(arm, study/arm/f"VALIDATION_{plan['epochs_per_arm']:03d}.json")
    reference_rows = []
    if reference is not None:
        before = watch.read(reference/'PROTOCOL.json')
        for key in ('parent_checkpoint_sha256', 'preparation_sha256', 'parameters', 'seed',
                    'effective_batch', 'microbatch', 'validation_cases', 'examples_per_epoch'):
            if before[key] != plan[key]:
                raise ValueError('Nonmatched learning-rate comparison: '+key)
        if before['train_schedule_epochs'][0] != plan['train_schedule_epochs'][0]:
            raise ValueError('Different first TRAIN schedule')
        for rel in set(before['source_sha256']) & set(plan['source_sha256']):
            if before['source_sha256'][rel] != plan['source_sha256'][rel]:
                raise ValueError('Shared implementation changed: '+rel)
        for arm in plan['arms']:
            path = reference/arm/'VALIDATION_001.json'
            old, _ = watch.validate(path, identities)
            hashes[f'reference/{arm}/e1'] = watch.digest(path)
            now = histories[arm][1]
            for count in (2, 3):
                a = next(r for r in old['by_count'] if r['count'] == count)
                b = next(r for r in now['by_count'] if r['count'] == count)
                reference_rows.append(dict(arm=arm, count=count,
                    nmse_before=a['mean_nmse'], nmse_after=b['mean_nmse'],
                    si_before=a['mean_si_sdr'], si_after=b['mean_si_sdr'],
                    both_better=b['mean_nmse'] < a['mean_nmse'] and b['mean_si_sdr'] > a['mean_si_sdr']))
    result = dict(status='COMPLETE', protocol_sha256=watch.digest(study/'PROTOCOL.json'),
        audit_sha256=watch.digest(audit_path), source_sha256=watch.digest(Path(__file__)),
        validation_sha256=hashes, parent=parent, histories=histories,
        selected_jointly_improves_parent=joint, final_all_source_successes=cases,
        same_actual_e1_learning_rate_comparison=reference_rows,
        waveform_reads=0, heldout_read=False, independent_test=False)
    watch.write(output.with_suffix('.json'), result)
    lines = ['# U-Net 미세조정: 완료된 파형 결과', '',
        f"같은 RFUAV 원기록 분할·같은 대역 합성·native 100MS/s·원 규모 {plan['parameters']:,}파라미터다. "
        f"동일한 부모 e4/300업데이트와 새 optimizer에서 두 손실을 각각 추가 {plan['epochs_per_arm']} epoch/"
        f"{plan['new_updates_per_arm']}업데이트 학습했다. seed 0의 개발검증 630개이며 독립 확인 실험이 아니다.", '',
        '아래 표는 시작 모델과 각 추가 epoch의 실제 가중치 성능이다. 선택 모델과 최종 실제 모델을 구분한다.', '',
        '| 모델 | 추가 epoch | NMSE 2/3 ↓ | SI-SDR 2/3 ↑ dB | 약신호 NMSE 2/3 ↓ | 합성 개수 정확도 2/3 |',
        '|---|---:|---|---|---|---|']
    def row(label, metric):
        a, b = [next(r for r in metric['by_count'] if r['count'] == c) for c in (2, 3)]
        return (f"| {label} | {metric['epoch'] if label!='시작 모델' else 0} | "
                f"{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f} | "
                f"{a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f} | "
                f"{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f} | "
                f"{a['construction_count_accuracy']:.1%}/{b['construction_count_accuracy']:.1%} |")
    lines.append(row('시작 모델', parent))
    for epoch in range(1, plan['epochs_per_arm']+1):
        for arm in plan['arms']:
            lines.append(row(arm, histories[arm][epoch]))
    lines += ['', '원래 규약의 선택 기준은 e0 포함 2/3성분 평균 raw NMSE 최소다.', '']
    for arm in plan['arms']:
        chosen = audit['results'][arm]['selected']['epoch']
        lines.append(f"- {arm}: 선택 추가 epoch {chosen}; 시작점의 네 평균 파형 지표 동시 개선: {joint[arm]}.")
    lines += ['', '1성분 조건의 시작 모델 및 마지막 실제 모델도 함께 보고한다.', '',
              '| 모델 | 1성분 NMSE ↓ | 1성분 SI-SDR ↑ dB | 합성 개수 정확도 |',
              '|---|---:|---:|---:|']
    for label, metric in [('시작 모델', parent)]+[(arm, histories[arm][-1]) for arm in plan['arms']]:
        r = next(r for r in metric['by_count'] if r['count'] == 1)
        lines.append(f"| {label} | {r['mean_nmse']:.6f} | {r['mean_si_sdr']:.3f} | {r['construction_count_accuracy']:.1%} |")
    lines += ['', '같은 1성분 조건에서 입력 복사 기준의 NMSE는 0이었다. '
              '[단순 기준선](TRIVIAL_WAVEFORM_BASELINES.md)과 함께 해석한다.']
    lines += ['', '마지막 실제 가중치의 모든 성분 동시 성공 사례도 함께 보고한다. 각 조건은 210개 혼합이다.', '',
        '| 모델 | 성분 수 | 모든 성분 NMSE<1 | 모든 성분 절대 SI-SDR>0 | 모든 성분 혼합 대비 SI-SDR 향상 |',
        '|---|---:|---:|---:|---:|']
    for r in cases:
        lines.append(f"| {r['model']} | {r['count']} | {r['all_nmse_below_one']}/210 | "
                     f"{r['all_absolute_si_positive']}/210 | {r['all_si_gain_positive']}/210 |")
    if reference_rows:
        lines += ['', '## 학습률 비교: 양쪽 모두 실제 추가 e1/75업데이트', '',
            '| 손실 | 성분 수 | NMSE 5e-4 → 1e-4 | SI-SDR 5e-4 → 1e-4 dB | 두 지표 개선 |',
            '|---|---:|---|---|---|']
        for r in reference_rows:
            lines.append(f"| {r['arm']} | {r['count']} | {r['nmse_before']:.6f} → {r['nmse_after']:.6f} | "
                         f"{r['si_before']:.3f} → {r['si_after']:.3f} | {r['both_better']} |")
    lines += ['', '원 신호 수는 합성한 기록 기여 수이며 물리 드론 대수 정답이 아니다. 전체 창 PIT 대응은 기종 식별이나 긴 구간 추적의 증거가 아니다. '
        '학습 손실 척도 변화, 원기록 수와 한 seed, 제한된 추가 학습량의 한계를 유지한다. 실패한 조건도 포함하며 확인용 기록은 열지 않았다.', '']
    watch.write(output.with_suffix('.md'), '\n'.join(lines))
    print(dict(status='COMPLETE', selected_jointly_improves_parent=joint, all_source_cases=cases))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('study', 'audit', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--reference', type=Path)
    a = p.parse_args()
    summarize(a.study.resolve(), a.audit.resolve(), a.output.resolve(), a.reference.resolve() if a.reference else None)
