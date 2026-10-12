"""Apply the already registered final-three-epoch criteria to audited rows."""
import itertools
import json
from pathlib import Path
import statistics
from compare_saved_rows import comparison, sha, read

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'local/sequence_assignment_20261013_v1'
PUBLIC=ROOT/'reports/2026-10-13'


def groups(value):return {v['count']:v for v in value['by_count']}


def assignment_summary(value):
    perms=list(itertools.permutations(range(3)));result=[]
    for k in (1,2,3):
        rows=[v for v in value['assignment_diagnostics'] if len(v['categories'])==k]
        result.append(dict(count=k,sequences=len(rows),
            active_order_changes=sum(len({tuple(perms[j][:k]) for j in v['assignment_cost']['local_indices']})>1 for v in rows),
            shared_minus_local_loss=statistics.mean(v['assignment_cost']['shared_min_loss']-v['assignment_cost']['local_min_loss'] for v in rows)))
    return result


def main():
    assert read(OUT/'STATE.json')['status']=='COMPLETE_AUDITED'
    initial=read(OUT/'INITIALIZATION.json');base=groups(initial['selected_validation'])
    events={};short={};long={};hashes={}
    for arm in ('independent','shared'):
        d=OUT/arm;e=read(d/'EPOCH_003.json');a=read(d/'AUDIT_003.json')
        assert a['status']=='PASS' and a['event_sha256']==sha(d/'EPOCH_003.json')
        assert e['epoch']==3 and e['updates']==288 and e['arm']==arm
        assert e['initial_checkpoint_sha256']==initial['checkpoint_sha256']
        for filename,key in (('SHORT_003.json','short_sha256'),('LONG_003.json','long_sha256')):
            assert sha(d/filename)==e[key]
            hashes[str((d/filename).relative_to(ROOT))]=sha(d/filename)
        events[arm]=e;short[arm]=read(d/'SHORT_003.json');long[arm]=read(d/'LONG_003.json')
    assert short['independent']['rows_sha256']==short['shared']['rows_sha256']
    matched=comparison(short['independent']['rows'],short['shared']['rows'])
    checks=[];candidate=groups(short['shared'])
    for k in (2,3):
        checks.extend([dict(criterion=f'{k}_source_nmse_vs_parent',passed=candidate[k]['mean_nmse']<base[k]['mean_nmse']),
            dict(criterion=f'{k}_weakest_nmse_vs_parent',passed=candidate[k]['weakest_nmse']<base[k]['weakest_nmse']),
            dict(criterion=f'{k}_si_sdr_vs_parent',passed=candidate[k]['mean_si_sdr'] is not None
                 and base[k]['mean_si_sdr'] is not None and candidate[k]['mean_si_sdr']>base[k]['mean_si_sdr'])])
    checks.extend([dict(criterion='short_macro_nmse_vs_matched_control',
        passed=short['shared']['selection_nmse']<short['independent']['selection_nmse']),
        dict(criterion='long_merged_macro_nmse_vs_matched_control',
        passed=long['shared']['merged']['selection_nmse']<long['independent']['merged']['selection_nmse'])])
    result=dict(status='VERIFIED',additional_epochs=3,updates_per_arm=288,initial_epoch=99,
        parent=initial['selected_validation'],arms={arm:dict(short={k:v for k,v in short[arm].items() if k!='rows'},
            long_merged={k:v for k,v in long[arm]['merged'].items() if k!='rows'},
            raw_concatenation={k:v for k,v in long[arm]['raw_concatenation'].items() if k!='rows'},
            assignment=assignment_summary(long[arm]),event=events[arm]) for arm in events},
        registered_checks=checks,shared_meets_waveform_criteria=all(v['passed'] for v in checks),
        paired_descriptive=matched,input_sha256=hashes,protocol_sha256=sha(OUT/'PROTOCOL.json'),
        current_best_changed=False,heldout_access=False,
        limitation='One seed, same-record B development; no architecture change; 24 long sequences; no significance or independent-recording claim')
    (PUBLIC/'SEQUENCE_FINAL_COMPARISON.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 연속 구간 출력 대응: 동일3epoch 비교 완료','',
        '부모99에서 같은 혼합 일정·fresh AdamW1e-5·288업데이트씩 학습했다. 본체와 손실 가중치는 같다.',
        '공통 순열군만 세 연속 창에 하나의 출력 대응을 적용한다. 파형 입력 길이는 각0.63872ms로 같다.','',
        '|조건|두 NMSE↓|세 NMSE↓|두/세 SI-SDR↑ dB|두/세 최약 NMSE↓|긴 병합 macroNMSE↓|',
        '|---|---:|---:|---:|---:|---:|']
    def line(label,s,l):
        g=groups(s)
        si=[f"{g[k]['mean_si_sdr']:.3f}" if g[k]['mean_si_sdr'] is not None else '정의불가' for k in (2,3)]
        return f"|{label}|{g[2]['mean_nmse']:.6f}|{g[3]['mean_nmse']:.6f}|{' / '.join(si)}|{g[2]['weakest_nmse']:.6f} / {g[3]['weakest_nmse']:.6f}|{l:.6f}|"
    lines.append(line('부모99',initial['selected_validation'],read(OUT/'INITIAL_LONG.json')['merged']['selection_nmse']))
    for arm,label in (('independent','창별 대응 추가3'),('shared','공통 대응 추가3')):
        lines.append(line(label,short[arm],long[arm]['merged']['selection_nmse']))
    lines+=['',f"등록된 파형 채택 기준: **{'모두 통과' if result['shared_meets_waveform_criteria'] else '미충족'}**.",
        '체크포인트 자동 교체는 하지 않았다. 패킷 보존·누출은 별도 고정 사례 결과와 함께 판단한다.','',
        '|판정 항목|결과|','|---|---|']
    lines.extend(f"|{v['criterion']}|{'통과' if v['passed'] else '미충족'}|" for v in checks)
    lines+=['','순서 변경 횟수는 비활성 출력끼리의 순서 변경을 제외해 계산했다.',
        '순서 보정 없이 연결한 파형이 좋아져도 최종 병합 파형이 나쁘면 배포 성능 개선으로 해석하지 않는다.',
        '같은 원녹음 시간분할·한 seed의 개발 실험이다. 여러 혼합 창을 독립 녹음 반복으로 세지 않는다.',
        '3epoch는 기존100epoch 모델의 미세조정 예산이며 처음부터의 수렴 비교가 아니다.','']
    (PUBLIC/'SEQUENCE_FINAL_COMPARISON_KO.md').write_text('\n'.join(lines))
    print(json.dumps(dict(checks=checks,accepted=result['shared_meets_waveform_criteria']),indent=2))


if __name__=='__main__':main()
