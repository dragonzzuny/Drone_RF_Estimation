"""Recompute native study CPU diagnoses without reopening any I/Q recordings."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from native_data import ROOT, sha256, write_json


def close(a, b):
    if a is None or b is None:
        if a is not b:
            raise ValueError('Undefined aggregate was replaced')
    else:
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def audit(root, output):
    protocols = {}
    for name in ('COUNT', 'REFERENCE'):
        if not (root / f'{name}_COMPLETE.json').exists() or (root / f'{name}_FAILURE.json').exists():
            raise ValueError(f'{name} incomplete or failed')
        protocol = json.loads((root / f'{name}_PROTOCOL.json').read_text())
        for rel, digest in protocol['source_sha256'].items():
            for path in (ROOT / rel, root / 'source_snapshot' / rel):
                if sha256(path) != digest:
                    raise ValueError(f'Changed diagnostic source {path}')
        protocols[name] = dict(sha256=sha256(root / f'{name}_PROTOCOL.json'),
                               source_files=len(protocol['source_sha256']))
    parent_path = root / 'gpu/VALIDATION_000.json'
    parent = json.loads(parent_path.read_text())['rows']
    if [r['index'] for r in parent] != list(range(630)):
        raise ValueError('Incorrect reference row ordering')
    truth = np.array([r['count'] for r in parent])
    count_path = root / 'count/RESULT.json'
    count = json.loads(count_path.read_text())
    cp = json.loads((root / 'COUNT_PROTOCOL.json').read_text())
    if (count['protocol_sha256'] != protocols['COUNT']['sha256'] or
        count['train_cases'] != 12000 or count['validation_cases'] != 630 or
        cp['trigger_sha256'] != sha256(root / 'gpu/VALIDATION_001.json') or
        cp['preparation_protocol_sha256'] != sha256(root / 'PREP_PROTOCOL.json')):
        raise ValueError('Count protocol or cohort changed')
    preparation_hash = sha256(root / 'preparation/PREPARATION.json')
    if count['native_preparation_sha256'] != preparation_hash:
        raise ValueError('Count preparation changed')
    count_results = {}
    for name, saved in count['results'].items():
        predicted = np.asarray(saved['predictions'])
        if predicted.shape != truth.shape or not np.isin(predicted, [1, 2, 3]).all():
            raise ValueError('Invalid count predictions')
        matrix = [[int(np.sum((truth == a) & (predicted == b))) for b in (1, 2, 3)] for a in (1, 2, 3)]
        if matrix != saved['confusion_matrix'] or saved['labels'] != [1, 2, 3]:
            raise ValueError('Count confusion matrix mismatch')
        close(float(np.mean(truth == predicted)), saved['accuracy'])
        for k in (1, 2, 3):
            close(float(np.mean(predicted[truth == k] == k)), saved['accuracy_by_count'][str(k)])
        count_results[name] = {k: v for k, v in saved.items() if k != 'predictions'}
        model_path = root / 'count' / f'{name}.joblib'
        if model_path.exists():
            count_results[name]['saved_classifier_sha256'] = sha256(model_path)
    reference_path = root / 'REFERENCE_NATIVE_POWER.json'
    reference = json.loads(reference_path.read_text())
    if (reference['protocol_sha256'] != protocols['REFERENCE']['sha256'] or
        reference['native_preparation_sha256'] != preparation_hash or
        reference['deployable'] or reference['performance_bound'] or reference['heldout_read']):
        raise ValueError('Reference-only diagnostic mislabeled')
    rows = reference['rows']
    if [r['index'] for r in rows] != [r['index'] for r in parent if r['count'] > 1]:
        raise ValueError('Missing or duplicated reference diagnostic case')
    max_sum_error = max_roundtrip = 0.
    for row in rows:
        base = parent[row['index']]
        for key in ('count', 'categories', 'pack_ids', 'weakest_index'):
            if row[key] != base[key]:
                raise ValueError('Reference diagnostic identity mismatch')
        close(row['reference_power'], base['reference_power'])
        max_roundtrip = max(max_roundtrip, max(row['roundtrip_nmse']))
        for values in row['modes'].values():
            if len(values['nmse']) != row['count'] or not np.isfinite(values['nmse']).all():
                raise ValueError('Invalid reference NMSE')
            max_sum_error = max(max_sum_error, values['sum_relative_error'])
    if max_sum_error > 1e-9 or max_roundtrip > 1e-9:
        raise ValueError('Reference waveform invariants failed')
    for group in reference['summary']:
        subset = [r for r in rows if r['count'] == group['count']]
        if len(subset) != 210 or group['cases'] != 210:
            raise ValueError('Incorrect diagnostic count balance')
        for mode, saved in group['modes'].items():
            si = [v for r in subset for v in r['modes'][mode]['si_sdr']]
            close(float(np.mean([v for r in subset for v in r['modes'][mode]['nmse']])), saved['mean_nmse'])
            close(float(np.mean(si)) if all(v is not None for v in si) else None, saved['mean_si_sdr'])
            close(float(np.mean([r['modes'][mode]['nmse'][r['weakest_index']] for r in subset])), saved['weakest_nmse'])
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'pandas')}
    warning_path = root / 'count_import_probe.stderr'
    result = dict(status='PASS', auditor_sha256=sha256(Path(__file__)), protocols=protocols,
                  parent_validation_sha256=sha256(parent_path), count_result_sha256=sha256(count_path),
                  reference_result_sha256=sha256(reference_path), count=count_results,
                  reference_assisted_only=reference['summary'], max_sum_error=max_sum_error,
                  max_roundtrip_nmse=max_roundtrip, environment_versions=versions,
                  count_import_warning_sha256=sha256(warning_path) if warning_path.exists() else None,
                  training_accuracy_audit='Logged training-fit accuracy; this audit recomputes validation aggregates from all predictions, not classifier fitting',
                  independent_test=False, raw_iq_reopened=False, physical_aircraft_count=False)
    write_json(output.with_suffix('.json'), result)
    lines = ['# 원 주파수 배치 실험의 CPU 진단', '',
        '다음 두 검사는 개발 검증을 본 뒤 추가한 탐색 진단이다. 독립 시험이나 실제 드론 대수 추정 성과가 아니다.', '',
        '## 혼합만 사용하는 구성 개수 분류', '',
        'TRAIN 12,000혼합으로 StandardScaler와 C=1 다항 로지스틱 회귀를 학습했다. 같은 630개발 검증에 고정 적용했다. '
        '65차원 시간 평균 또는 평균+표준편차130차원을 사용하며, 시간 순서·호핑 주기는 학습하지 않는다.', '',
        '| 입력 | 학습 정확도 | 개발 검증 정확도 | 3성분 재현율 |', '|---|---:|---:|---:|']
    for name, v in count_results.items():
        train = f"{v['training_accuracy']:.2%}" if 'training_accuracy' in v else '—'
        lines.append(f"| {name} | {train} | {v['accuracy']:.2%} | {v['accuracy_by_count']['3']:.2%} |")
    lines += ['', '두 학습 분류기는 새 기록·대역폭 조건에서 일반화에 실패했다. 기록 묶음과 VTSBW 변화가 얽혀 있어 원인을 대역폭 하나로 확정하지 않는다. '
        '분류기는 비채택하며 U-Net 파형 출력이나 개수 head를 바꾸지 않았다. 정답 개수는 추론 입력이 아니다.', '',
        '## 정답을 사용한 전력 분배 진단 — 실제 분리 성능과 구분', '',
        '같은 420개발 혼합에서 정답 I/Q와 정답 구성 개수를 사용해 마스크를 만들고 혼합 위상을 유지했다. '
        '참값을 모르는 실제 추론에 사용할 수 없으며 최적 성능 상한·하한도 아니다.', '',
        '| 정답 정보 | 구성 수 | NMSE | 복소 SI-SDR dB | 약신호 NMSE |', '|---|---:|---:|---:|---:|']
    for group in reference['summary']:
        for name, values in group['modes'].items():
            lines.append(f"| {name} | {group['count']} | {values['mean_nmse']:.6f} | {values['mean_si_sdr']:.3f} | {values['weakest_nmse']:.6f} |")
    lines += ['', '정답의 국소 전력 정보를 주면 오차를 더 줄일 수 있었다. 이 정보를 혼합만으로 추정할 수 있다는 증거나, '
        '현재 잔차가 잡음의 식별 한계에 도달했다는 증거는 아니다.', '',
        '## 검산 및 실행 환경', '',
        '두 실행의 코드 보존본·준비 규약 해시, 개수 예측630행/혼동행렬, 정답 보조420행의 자료 동일성·집계·STFT 왕복·혼합 합을 재검산했다. '
        '검산에는 새 I/Q를 읽지 않았다. 학습 정확도는 저장된 학습 결과이며 이 검산에서 분류기 학습을 다시 수행하지 않았다.', '',
        f'실행 환경: {versions}. pandas가 불러온 선택 의존성 numexpr/bottleneck에서 NumPy ABI 경고가 발생했다. '
        '실제 ndarray 분류기 fit/predict 검사는 통과했고 두 본 학습도 수렴 경고 없이 정상 완료했다. 환경 패키지는 변경하지 않았고 경고 로그를 보존했다.', '',
        '[실제 추론 결과](NATIVE_RF_FINAL.md) · [규약과 코드](../../experiments/rfuav_native_frequency_20261009/README.md)', '']
    output.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='PASS', count_cases=630, reference_cases=len(rows), versions=versions)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.output)
