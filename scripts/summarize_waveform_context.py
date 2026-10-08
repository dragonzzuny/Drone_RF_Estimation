"""Audit saved epoch metrics without loading a model, I/Q, or held-out data.

Raw means remain primary. Local power-gap summaries are descriptive diagnostics
and never filter cases or change the frozen checkpoint selection rule.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


ARMS = ('short_context', 'long_context')


def read(path):
    return json.loads(path.read_text())


def near(a, b):
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-11):
        raise ValueError(f'Saved aggregate differs from rows: {a} vs {b}')


def mean(values):
    values = list(values)
    if not values or any(v is None or not math.isfinite(v) for v in values):
        raise ValueError('Empty or nonfinite aggregate; do not silently discard')
    return statistics.mean(values)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (position - left)


def audit_validation(data):
    rows = data['rows']
    if len(rows) != 630 or sorted(r['index'] for r in rows) != list(range(630)):
        raise ValueError('Validation case list changed')
    diagnostics = []
    for group in data['by_count']:
        count = group['count']
        cases = [r for r in rows if r['count'] == count]
        if len(cases) != 210:
            raise ValueError('Validation count stratum changed')
        nmse = mean(v for r in cases for v in r['nmse'])
        si = mean(v for r in cases for v in r['si_sdr'])
        weak = mean(r['nmse'][r['weakest_index']] for r in cases)
        accuracy = mean(int(r['predicted_count'] == count) for r in cases)
        near(nmse, group['mean_nmse'])
        near(si, group['mean_si_sdr'])
        near(weak, group['weakest_nmse'])
        near(accuracy, group['construction_count_accuracy'])
        if count > 1:
            near(mean(v for r in cases for v in r['si_sdr_gain']), group['mean_si_sdr_gain'])
        case_nmse = [mean(r['nmse']) for r in cases]
        gap_groups = []
        if count > 1:
            for low, high in ((0, 10), (10, 20), (20, math.inf)):
                subset = []
                for row in cases:
                    powers = row['reference_power']
                    if not min(powers) > 0:
                        raise ValueError('Nonpositive reference power')
                    gap = 10 * math.log10(max(powers) / min(powers))
                    if (gap >= low if low == 0 else gap > low) and gap <= high:
                        subset.append(row)
                gap_groups.append(dict(lower_db=low, upper_db=None if math.isinf(high) else high,
                    cases=len(subset), mean_nmse=mean(v for r in subset for v in r['nmse']) if subset else None,
                    mean_si_sdr=mean(v for r in subset for v in r['si_sdr']) if subset else None))
            if sum(g['cases'] for g in gap_groups) != 210:
                raise ValueError('Power-gap diagnostics lost cases')
        diagnostics.append(dict(count=count, cases=210,
            case_mean_nmse_median=statistics.median(case_nmse),
            case_mean_nmse_p90=percentile(case_nmse, .9),
            case_mean_nmse_max=max(case_nmse),
            predicted_count_histogram=[sum(r['predicted_count'] == k for r in cases) for k in (1, 2, 3)],
            mean_background_nmse=mean(r['background_nmse'] for r in cases),
            local_power_gap=gap_groups))
    near(mean(g['mean_nmse'] for g in data['by_count'] if g['count'] > 1), data['selection_nmse'])
    if max(r['sum_relative_error'] for r in rows) > 1e-9:
        raise ValueError('Mixture sum check failed')
    return diagnostics


def summarize(run, arm_names=ARMS):
    protocol = read(run / 'PROTOCOL.json')
    if list(arm_names) != protocol['arms']:
        raise ValueError('Requested arm names disagree with frozen protocol')
    digest = hashlib.sha256((run / 'PROTOCOL.json').read_bytes()).hexdigest()
    arms = {}
    metadata = None
    for arm in arm_names:
        records = []
        for path in sorted((run / arm).glob('EPOCH_*.json')):
            receipt = read(path)
            epoch = receipt['epoch']
            if epoch != len(records) + 1 or receipt['updates'] != epoch * protocol['updates_per_epoch']:
                raise ValueError('Nonconsecutive epochs or update budget mismatch')
            if receipt['protocol_sha256'] != digest:
                raise ValueError('Protocol hash mismatch')
            validation = read(run / arm / f'VALIDATION_{epoch:03d}.json')
            if {k: v for k, v in validation.items() if k != 'rows'} != receipt['metrics']:
                raise ValueError('Receipt and validation disagree')
            identities = [{k: row[k] for k in ('index', 'count', 'categories', 'pack_ids',
                'nominal_levels_db', 'reference_power', 'input_si_sdr', 'weakest_index')}
                for row in validation['rows']]
            if metadata is None:
                metadata = identities
            elif identities != metadata:
                raise ValueError('Validation inputs differ across epochs/arms')
            record = dict(receipt, diagnostics=audit_validation(validation))
            if records:
                previous = records[-1]
                record['change_from_previous_epoch'] = dict(
                    average_loss=receipt['average_loss'] - previous['average_loss'],
                    by_count=[dict(count=g['count'],
                        nmse_percent=100 * (g['mean_nmse'] / old['mean_nmse'] - 1),
                        si_sdr_db=g['mean_si_sdr'] - old['mean_si_sdr'])
                        for g, old in zip(receipt['metrics']['by_count'], previous['metrics']['by_count'])])
            records.append(record)
        initial = read(run / arm / 'VALIDATION_000.json')
        audit_validation(initial)
        arms[arm] = dict(initial={k: v for k, v in initial.items() if k != 'rows'}, epochs=records)
    common = min(len(arms[a]['epochs']) for a in arm_names)
    selected = {}
    for arm in arm_names:
        candidates = [arms[arm]['initial']] + [r['metrics'] for r in arms[arm]['epochs'][:common]]
        best = min(candidates, key=lambda m: m['selection_nmse'])
        selected[arm] = best
        if common:
            saved = arms[arm]['epochs'][common - 1]['best']
            if saved['epoch'] != best['epoch']:
                raise ValueError('Checkpoint selection rule disagrees with receipts')
            near(saved['metric'], best['selection_nmse'])
    short = {g['count']: g for g in selected[arm_names[0]]['by_count']}
    long = {g['count']: g for g in selected[arm_names[1]]['by_count']}
    accepted = all(long[k]['mean_nmse'] < short[k]['mean_nmse'] and
                   long[k]['mean_si_sdr'] > short[k]['mean_si_sdr'] for k in (2, 3))
    return dict(updated_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        status='COMPLETED' if (run / 'COMPLETE.json').exists() else 'INTERIM',
        common_completed_epoch=common, updates_per_arm=common * protocol['updates_per_epoch'],
        protocol_sha256=digest, protocol=protocol, arms=arms, selected_at_common_budget=selected,
        long_context_meets_prespecified_acceptance=accepted, all_rows_reaggregated=True,
        same_validation_metadata_verified=True, raw_means_primary=True,
        local_power_gap_analysis='posthoc descriptive; no exclusions or reselection',
        independent_test=False, physical_aircraft_count=False, heldout_iq_access=False)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_completion(run, result):
    """CPU-only audit of this project's trusted local training checkpoints."""
    import torch
    torch.set_num_threads(2)
    if result['status'] != 'COMPLETED' or result['common_completed_epoch'] != result['protocol']['epochs']:
        raise ValueError('Cannot audit completion before the full matched budget finishes')
    done = read(run / 'COMPLETE.json')
    if done['protocol_sha256'] != result['protocol_sha256'] or done['updates_per_arm'] != result['updates_per_arm']:
        raise ValueError('Completion marker disagrees with receipts')
    root = Path(__file__).resolve().parents[1]
    protocol = result['protocol']
    for name, digest in protocol['source_sha256'].items():
        for path in (root / name, run / 'source_snapshot' / name):
            if file_digest(path) != digest:
                raise ValueError(f'Frozen source changed: {path}')
    for name, digest in protocol['data_sha256'].items():
        if file_digest(Path(name)) != digest:
            raise ValueError(f'Frozen data schedule changed: {name}')
    checkpoints = {}
    for arm in protocol['arms']:
        folder = run / arm
        hashes = {name: file_digest(folder / name) for name in ('BEST.pt', 'LAST.pt', 'SELECTED_005.pt')}
        if hashes['BEST.pt'] != hashes['SELECTED_005.pt']:
            raise ValueError('Selected milestone does not match BEST')
        best = torch.load(folder / 'BEST.pt', map_location='cpu', weights_only=False)
        last = torch.load(folder / 'LAST.pt', map_location='cpu', weights_only=False)
        chosen = result['selected_at_common_budget'][arm]
        if last['epoch'] != 5 or last['updates'] != result['updates_per_arm']:
            raise ValueError('Checkpoint update budget differs from receipts')
        for checkpoint in (best, last):
            if checkpoint['protocol_sha256'] != result['protocol_sha256'] or checkpoint['best']['epoch'] != chosen['epoch']:
                raise ValueError('Checkpoint protocol or selection differs')
            near(checkpoint['best']['metric'], chosen['selection_nmse'])
            if not all(torch.isfinite(v).all().item() for v in checkpoint['model'].values()):
                raise ValueError('Nonfinite model state')
        preflight = read(folder / 'GPU_PREFLIGHT.json')
        elements = sum(v.numel() for v in best['model'].values())
        if elements != preflight['parameters']:
            raise ValueError('Saved model state does not match full preflight capacity')
        checkpoints[arm] = dict(sha256=hashes, selected_epoch=chosen['epoch'],
            updates=last['updates'], state_elements=elements, finite=True)
        del best, last
    return dict(status='PASS', frozen_live_and_snapshot_sources=len(protocol['source_sha256']),
        data_schedule_hashes=len(protocol['data_sha256']), checkpoints=checkpoints,
        loaded_on_cpu=True, raw_iq_read=False)


def markdown(result):
    lines = ['# 복소 I/Q 문맥 대조: epoch별 검산 결과', '',
        f"갱신: {result['updated_at']} · 상태: {result['status']}", '',
        'RFUAV 같은 원 RF 대역 내 중심 정렬 합성, 원기록 그룹 분할의 개발 검증 630혼합.',
        '각 epoch 2,400학습 혼합/300업데이트, seed 0. 두 군 각각 32,355,203파라미터이며 같은 초기 가중치다.',
        '복소 입력은 0.63872ms/10.48576ms, 채점 창은 양쪽 0.63872ms. 두 군 모두 같은 약 21ms 시간 평균 전력 특징을 받는다.', '',
        '| 복소 문맥 | epoch | 학습 손실 | 2성분 NMSE ↓ | 3성분 NMSE ↓ | 2성분 SI-SDR dB ↑ | 3성분 SI-SDR dB ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm, label in zip(ARMS, ('짧은 문맥', '긴 문맥')):
        for row in result['arms'][arm]['epochs']:
            groups = {g['count']: g for g in row['metrics']['by_count']}
            lines.append(f"| {label} | {row['epoch']} | {row['average_loss']:.6f} | "
                f"{groups[2]['mean_nmse']:.6f} | {groups[3]['mean_nmse']:.6f} | "
                f"{groups[2]['mean_si_sdr']:.3f} | {groups[3]['mean_si_sdr']:.3f} |")
    lines += ['', f"공통 완료: {result['common_completed_epoch']} epoch / 각 {result['updates_per_arm']}업데이트.",
        '체크포인트 선택은 epoch 0을 포함한 2·3성분 평균 NMSE 최솟값이며 1성분 성능은 따로 보고한다.', '',
        '| 선택 군 | 선택 epoch | 1성분 NMSE | 2성분 NMSE | 3성분 NMSE | 2성분 SI-SDR | 3성분 SI-SDR |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        row = result['selected_at_common_budget'][arm]
        groups = {g['count']: g for g in row['by_count']}
        lines.append(f"| {arm} | {row['epoch']} | {groups[1]['mean_nmse']:.6f} | "
            f"{groups[2]['mean_nmse']:.6f} | {groups[3]['mean_nmse']:.6f} | "
            f"{groups[2]['mean_si_sdr']:.3f} | {groups[3]['mean_si_sdr']:.3f} |")
    verdict = '충족' if result['long_context_meets_prespecified_acceptance'] else '미충족'
    lines += ['', f'긴 문맥의 사전 개선 기준(2·3성분 모두 NMSE 감소와 SI-SDR 증가): **{verdict}**.']
    selected_epochs = {a: result['selected_at_common_budget'][a]['epoch'] for a in ARMS}
    if all(e > 0 for e in selected_epochs.values()):
        selected_diagnostics = {a: {g['count']: g for g in result['arms'][a]['epochs'][selected_epochs[a] - 1]['diagnostics']} for a in ARMS}
        lines += ['', '사후 조건별 진단: 채점 창에서 가장 강한/약한 정답의 전력차를 사용한다. 합성 설정값이나 물리적 송신 활동 라벨이 아니다.', '',
            '| 성분 수 | 국소 전력차 | 사례 수 | 짧은 NMSE | 긴 NMSE | 짧은 SI-SDR | 긴 SI-SDR |',
            '|---|---|---:|---:|---:|---:|---:|']
        for count in (2, 3):
            for short, long in zip(selected_diagnostics['short_context'][count]['local_power_gap'], selected_diagnostics['long_context'][count]['local_power_gap']):
                if short['cases'] != long['cases']:
                    raise ValueError('Local power-gap strata differ')
                label = '0–10dB' if short['lower_db'] == 0 else '10 초과–20dB' if short['upper_db'] == 20 else '20dB 초과'
                if short['cases']:
                    lines.append(f"| {count} | {label} | {short['cases']} | {short['mean_nmse']:.6f} | {long['mean_nmse']:.6f} | {short['mean_si_sdr']:.3f} | {long['mean_si_sdr']:.3f} |")
        lines += ['', '이 조건별 표는 전체 사례를 유지한 사후 진단이며, 유리한 부분집합으로 사전 평균 지표를 대체하거나 확인적 개선으로 주장하지 않는다.']
    lines += [
        '', 'NMSE는 출력의 배율·위상을 정답으로 보정하지 않은 복소 I/Q 오차다. 현재 복소 SI-SDR은 평균을 제거하고 상수 복소 배율(전체 위상 회전 포함)을 허용한다.',
        '따라서 정확한 I/Q 복원에는 NMSE를 주 지표로 쓰며, 분리 품질을 확인하는 복소 SI-SDR을 함께 보고한다. 일정한 비영 배율만 바꾸면 복소 SI-SDR은 변하지 않는다.',
        '배율과 분리 품질의 구분에 관한 원 논문: [Le Roux 외, SDR – half-baked or well done?](https://arxiv.org/html/1811.02508v1). 우리 복소 확장은 원 논문의 실수 음성 지표와 구분한다.',
        '', '모든 저장 평가 행을 재집계했고 동일한 검증 사례·참조 전력·업데이트 예산을 확인했다.',
        '아주 작은 참조 전력도 제외하지 않았다. JSON의 국소 전력차·중앙값 진단은 사후 설명용이며 주 평균·선택 기준을 바꾸지 않는다.',
        '한 seed의 개발 비교이며 미지 기종 평가, 실제 동시 수신 분리, 물리 드론 수 추정 결과가 아니다.',
        '원 중심주파수 간격은 보존하지 않았다. 학습과 검증의 VTSBW 조건 차이, 기록 잡음 포함 정답, 입력 가림에 따른 GroupNorm 변화가 있다.',
        '이 대조로 모든 긴 문맥 모델의 성패를 일반화하거나 기존 전이 STFT 모델 대비 구조 우월성을 주장하지 않는다.', '']
    if 'completion_audit' in result:
        lines += ['완료 검산: 봉인 소스·스냅샷·학습/검증 목록 해시, 양쪽 체크포인트의 1,500업데이트·선택 epoch·유한한 원 규모 가중치, 최종 선택 사본 일치를 확인했다.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output filename prefix')
    parser.add_argument('--verify-completion', action='store_true', help='Audit final local checkpoints on CPU; requires torch')
    args = parser.parse_args()
    result = summarize(args.run)
    if args.verify_completion:
        result['completion_audit'] = audit_completion(args.run, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    args.output.with_suffix('.md').write_text(markdown(result))
    print(json.dumps({k: result[k] for k in ('status', 'common_completed_epoch',
        'updates_per_arm', 'long_context_meets_prespecified_acceptance', 'all_rows_reaggregated')}))
