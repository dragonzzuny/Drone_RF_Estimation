"""Conditional error geometry from all saved three-source initial outputs."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2];public=ROOT/'reports/2026-10-10'
path=public/'SUCCESSIVE_INITIAL48_RESULT.json';value=json.loads(path.read_text());rows=[]
assert json.loads((public/'SUCCESSIVE_INITIAL48_AUDIT.json').read_text())['status']=='PASS'
for row in value['rows']:
    if row['count']!=3:continue
    metric=row['successive'];assert metric['sum_relative_error']<1e-10
    order=[metric['assignment'].index(slot) for slot in (0,1,2)]
    a,b,c=[metric['absolute_error_power'][j] for j in order]
    p=metric['reference_power'][order[2]]
    low=(math.sqrt(a)-math.sqrt(b))**2;high=(math.sqrt(a)+math.sqrt(b))**2
    cosine=(c-a-b)/(2*math.sqrt(a*b))
    assert low-1e-7<=c<=high+1e-7 and -1-1e-6<=cosine<=1+1e-6
    rows.append(dict(index=row['index'],matched_references_in_output_order=order,
        first_error_power=a,second_error_power=b,last_error_power=c,last_reference_power=p,
        last_nmse=c/p,conditional_lower_nmse_given_first_two_error_norms=low/p,
        derived_real_error_cosine=cosine))
assert len(rows)==14
result=dict(status='CHECKED_SAVED_ROWS',rows=rows,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    median_derived_error_cosine=statistics.median(r['derived_real_error_cosine'] for r in rows),
    median_conditional_lower_nmse=statistics.median(r['conditional_lower_nmse_given_first_two_error_norms'] for r in rows),
    conditional_lower_nmse_above_point1=sum(r['conditional_lower_nmse_given_first_two_error_norms']>.1 for r in rows),
    heldout_read=False,additional_inference=0,updates=0,
    limitation='Conditional geometry holds current first-two error norms fixed; not a learned performance limit or receiver-noise floor. Error cosine derived from conservation, not separately measured waveform cross-products.')
(public/'SUCCESSIVE_ERROR_GEOMETRY.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# 마지막 잔여 오차와 앞선 두 추출 오차의 관계','',
    '고정TRAIN14의 초기 세 성분 출력은 합이 혼합과 일치하고 배경 출력은0이다. '
    '출력에 대응하는 정답 순서로 오차를 e1,e2,e3라고 하면 e3=−(e1+e2)다. '
    '표본 평균 오차 에너지를 E1,E2,E3라 할 때 다음 관계가 성립한다.','',
    '```','E3 = E1 + E2 + 2 Re <e1, e2>',
    '(sqrt(E1) − sqrt(E2))² ≤ E3 ≤ (sqrt(E1) + sqrt(E2))²','```','',
    '마지막 정답 에너지로 나누면 해당 출력 NMSE의 조건부 범위를 얻는다. '
    '첫 두 오차의 에너지를 **현재 값에 고정했을 때**, 14사례 중6사례는 최대로 상쇄하더라도NMSE0.1보다 큰 하한을 갖는다. '
    '따라서 그6사례에서0.1이하를 목표로 한다면 첫 두 추출 오차의 크기 자체도 변해야 한다. '
    '학습하면 이 오차 크기가 변하므로 전체 모델의 달성 가능한 한계를 뜻하지 않는다.','',
    f"보존식으로 유도한 두 오차의 실수 내적 cosine 중앙값은 {result['median_derived_error_cosine']:.6f}다. "
    '오차끼리 상당 부분 상쇄돼도 약한 정답 에너지에 비하면 잔여가 클 수 있다. '
    '이는 원 신호 간 상관이나 강신호 누출 비율을 측정한 값과 다르다. '
    '별도 파형 재추론 없이 저장된 에너지와 합 일치 진단값에서 유도했다.',
    '', '등록한 공동 학습은 마지막 출력의 손실을 앞선 두 단계까지 전달한다. '
    '이 해석은 그 연결의 필요성을 설명하지만 실제 DEV 개선을 보장하지 않는다.',
    '', '[14사례 전체 수치](SUCCESSIVE_ERROR_GEOMETRY.json)']
(public/'SUCCESSIVE_ERROR_GEOMETRY_KO.md').write_text('\n'.join(lines)+'\n')
print({k:v for k,v in result.items() if k!='rows'})
