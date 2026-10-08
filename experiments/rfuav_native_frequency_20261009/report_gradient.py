"""Audit scalar gradient geometry and record the resulting matched ablation."""
import argparse
import json
from pathlib import Path
import numpy as np
from native_data import ROOT, sha256, write_json


def run(parent, output):
    root=parent/'gradient'
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    result=json.loads((root/'COMPLETE.json').read_text())
    if result['protocol_sha256']!=sha256(root/'PROTOCOL.json') or (root/'FAILURE.json').exists():
        raise ValueError('Invalid gradient completion')
    for rel,digest in protocol['source_sha256'].items():
        if sha256(ROOT/rel)!=digest or sha256(root/'source_snapshot'/rel)!=digest:
            raise ValueError('Gradient source changed')
    for path,digest in protocol['data_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Gradient metadata changed')
    if sha256(parent/'gpu/SELECTED_005.pt')!=protocol['checkpoint_sha256']:
        raise ValueError('Gradient model changed')
    if [r['index'] for r in result['rows']]!=protocol['indices'] or len(result['rows'])!=96:
        raise ValueError('Gradient cases changed')
    for group in result['by_count']:
        rows=[r for r in result['rows'] if r['count']==group['count']]
        if len(rows)!=32:raise ValueError('Wrong count balance')
        np.testing.assert_allclose(np.mean([r['cosine']<0 for r in rows]),group['negative_case_fraction'])
    for g in result['rows']+result['by_count']+[result['overall']]:
        if not all(np.isfinite(g[k]) for k in ('wave_norm','weighted_count_norm','dot','cosine','weighted_count_to_wave_norm')):
            raise ValueError('Invalid gradient scalar')
        np.testing.assert_allclose(g['cosine'],g['dot']/max(g['wave_norm']*g['weighted_count_norm'],1e-30),rtol=1e-12)
        np.testing.assert_allclose(g['weighted_count_to_wave_norm'],g['weighted_count_norm']/max(g['wave_norm'],1e-30),rtol=1e-12)
    trigger=result['overall']['cosine']<0 and result['overall']['weighted_count_to_wave_norm']>=1
    if trigger!=result['next_test_triggered']:raise ValueError('Changed next-test rule')
    report=dict(status='PASS',protocol=protocol,result=result,
        audit=dict(source_files=len(protocol['source_sha256']),auditor_sha256=sha256(__file__),
            result_sha256=sha256(root/'COMPLETE.json'),
            scope='source/data/model hashes and scalar geometry; aggregate vectors were computed by frozen worker and are not reconstructed from per-case norms'))
    write_json(output.with_suffix('.json'),report)
    lines=['# 개수 감독과 파형 복원의 공통 인코더 기울기 진단','',
        '선택된 native e2 모델을 고정하고, 학습 epoch1의 구성 수별32개씩 총96혼합을 사용했다. '
        '개발 검증 파형을 새로 읽거나 모델을 업데이트하지 않았다. 모델 선택 자체에는 앞선 개발 검증 이력이 있다.', '',
        '공통 시간 문맥 인코더의 603,136파라미터에서 기존 파형 손실과 **0.1×개수 교차엔트로피**의 기울기를 각각 계산했다. '
        '아래는 각 집단에서 기울기 벡터를 평균한 뒤 얻은 각도와 크기 비다. 전체32,142,859파라미터의 기울기 비교는 아니다.','',
        '| 구성 수 | 혼합 수 | 평균 기울기 cosine | 가중 개수/파형 기울기 norm | 개별 반대 방향 비율 |',
        '|---|---:|---:|---:|---:|']
    for g in result['by_count']:
        lines.append(f"| {g['count']} | {g['cases']} | {g['cosine']:.6f} | {g['weighted_count_to_wave_norm']:.3f} | {g['negative_case_fraction']:.2%} |")
    g=result['overall']
    lines.append(f"| 전체 | 96 | {g['cosine']:.6f} | {g['weighted_count_to_wave_norm']:.3f} | — |")
    lines+=['','이 지점의 공통 인코더에서 두 목적의 평균 기울기가 반대 방향이었다. '
        '유한한 학습 표본·단일 체크포인트의 국소 최적화 진단이며, 이것만으로 검증 성능 저하의 원인이나 보정 효과를 확정하지 않는다.','',
        '## 등록한 직접 대조','',
        '진단 전에 정한 `전체 cosine < 0` 및 `가중 개수 norm ≥ 파형 norm` 조건을 만족했다. '
        '따라서 같은 부모 가중치·원 규모·seed0·혼합 목록·새 AdamW·5 epoch/375업데이트를 유지하고, '
        '개수 head 입력에만 stop-gradient를 적용하는 군을 등록했다. 개수 head 자체는 계속 학습하며 공통 인코더는 파형 손실로 학습한다. '
        '완료된 native 원 대조군의 같은 예산 결과와 비교한다.','',
        '이 변경은 비지도 학습이나 새 신경망 구조가 아니다. supervised 파형 복원과 개수 분류 사이의 기울기 전달 범위를 바꾸는 실험이다. '
        '초기 추론 출력 동일성, 개수 head 기울기 유지, 인코더의 개수 기울기 차단 및 파형 기울기 유지를 GPU에서 검사한다. '
        '최종 채택은 두·세 성분에서 NMSE와 SI-SDR이 함께 개선되는지로 판단하고 약신호·개수 성능도 공개한다.','',
        '[고정 모델 추론 및 적합도 진단](NATIVE_RF_INFERENCE_DIAGNOSIS.md) · '
        '[실행 코드](../../experiments/rfuav_native_frequency_20261009/detach_count.py)','']
    output.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='PASS',cases=96,next_test_triggered=trigger,overall=result['overall'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.run,args.output)
