"""All output roles in the initial successive TRAIN48 diagnostic."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2];public=ROOT/'reports/2026-10-10'
path=public/'SUCCESSIVE_INITIAL48_RESULT.json';value=json.loads(path.read_text())
assert json.loads((public/'SUCCESSIVE_INITIAL48_AUDIT.json').read_text())['status']=='PASS'
summary=[]
for count in (1,2,3):
    rows=[r for r in value['rows'] if r['count']==count]
    for slot in (0,1,2):
        matched=[(r,j) for r in rows for j,s in enumerate(r['successive']['assignment'][:count]) if s==slot]
        summary.append(dict(count=count,output_slot=slot,matched_active_sources=len(matched),
            weakest_matches=sum(r['successive']['assignment'][r['successive']['weakest_index']]==slot for r in rows),
            mean_nmse=statistics.mean(r['successive']['nmse'][j] for r,j in matched) if matched else None,
            mean_si_sdr=statistics.mean(r['successive']['si_sdr'][j] for r,j in matched) if matched else None,
            both_improved_sources=sum(r['successive']['nmse'][j]<r['parent']['nmse'][j] and
                r['successive']['si_sdr'][j]>r['parent']['si_sdr'][j] for r,j in matched)))
result=dict(status='CHECKED_SAVED_ROWS',summary=summary,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    heldout_read=False,additional_inference=0,optimizer_steps=0,
    limitation='PIT-aligned output roles on reused TRAIN48, not a learned aircraft identity or causal isolation')
(public/'SUCCESSIVE_INITIAL_ROLES.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# 순차 출력의 역할별 초기 오차','',
    '이미 검산된 TRAIN48 전체 행을 출력 역할별로 다시 집계했다. 첫·둘째는 추출 출력, 셋째는 마지막 잔여다. '
    '출력과 정답의 대응은 평가용 전체 창 PIT이며 실제 추론에 기종을 주는 것이 아니다.','',
    '| 성분 수 | 출력 번호 | 정답 대응 수 | 최약 정답 대응 수 | NMSE ↓ | 복소SI-SDR ↑ dB | 부모 대비 두 지표 개선 수 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for row in summary:
    n='해당 없음' if row['mean_nmse'] is None else f"{row['mean_nmse']:.6f}"
    s='해당 없음' if row['mean_si_sdr'] is None else f"{row['mean_si_sdr']:.3f}"
    lines.append(f"|{row['count']}|{row['output_slot']+1}|{row['matched_active_sources']}|{row['weakest_matches']}|{n}|{s}|{row['both_improved_sources']}|")
lines+=['','세 성분14혼합 모두의 최약 정답이 마지막 잔여 출력에 대응했고 평균NMSE는1.549152였다. '
    'NMSE=1은0출력의 기준이므로, 해당 TRAIN 사례 집계에서는 진폭을 포함한 오차가 크다. '
    '복소SI-SDR이 좋아지는 현상만으로 이 오차를 해결했다고 보지 않는다. '
    '초기 구조의 관찰이며 공동 학습이 개선할지는 등록한 후속에서 검증한다.',
    '', '[전 조건 JSON](SUCCESSIVE_INITIAL_ROLES.json) · [전체48행](SUCCESSIVE_INITIAL48_RESULT.json)']
(public/'SUCCESSIVE_INITIAL_ROLES_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary))
