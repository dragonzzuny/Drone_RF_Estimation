"""Reaggregate all saved TRAIN6 overlap cases and audit the execution receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def read(p):
    return json.loads(p.read_text())


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def audit(run,folder):
    p=read(run/'PROTOCOL.json');r=read(run/'COMPLETE.json')
    assert r['status']=='COMPLETE' and r['protocol_sha256']==digest(run/'PROTOCOL.json')
    assert r['inferences']==18 and r['updates']==0 and not r['heldout_read'] and not r['gpu_use']
    assert p['indices']==[0,1,4,5,2,11] and p['offsets']==[0,-16384,16384]
    assert len(r['rows'])==30 and len(r['diagnostics'])==6
    assert {(q['index'],q['mode']) for q in r['rows']}=={(i,m) for i in p['indices'] for m in p['modes']}
    for rel,sha in p['source_sha256'].items():
        assert digest(ROOT/rel)==digest(run/'source_snapshot'/rel)==sha
    assert digest(Path(p['parent']))==p['parent_sha256']
    assert digest(Path(p['preparation'])/'PREPARATION.json')==p['preparation_sha256']
    assert digest(folder/'WINDOW_OVERLAP_PLAN_KO.md')==p['plan_sha256']
    for q in r['rows']:
        assert q['sum_relative_error']<1e-10
        assert q['assignment']==next(x for x in r['rows'] if x['index']==q['index'] and x['mode']=='central')['assignment']
        for error,power,nmse in zip(q['absolute_error_power'],q['reference_power'],q['nmse']):
            assert abs(error/power-nmse)<1e-12
    for q in r['diagnostics']:
        assert q['reference_and_mixture_overlap_bitwise_equal'] and q['common_samples']==31104
        assert len(q['windows'])==3
        for window in q['windows']:
            assert sorted(window['prediction_only_order'][:3])==[0,1,2] and window['prediction_only_order'][3]==3
    for s in r['by_count']:
        rows=[q for q in r['rows'] if q['mode']==s['mode'] and q['count']==s['count']]
        assert len(rows)==s['cases']==2
        values=dict(nmse=statistics.mean(statistics.mean(q['nmse']) for q in rows),
            si_sdr=statistics.mean(statistics.mean(q['si_sdr']) for q in rows),
            weakest_nmse=statistics.mean(q['nmse'][q['weakest_index']] for q in rows))
        for key,value in values.items():
            assert abs(value-s[key])<1e-12
    def get(mode,count):
        return next(s for s in r['by_count'] if s['mode']==mode and s['count']==count)
    comparisons=[]
    for mode in p['modes'][1:]:
        deltas=[]
        for n in (1,2,3):
            a,b=get('central',n),get(mode,n)
            deltas.append(dict(count=n,**{k:b[k]-a[k] for k in ('nmse','si_sdr','weakest_nmse')}))
        passed=all(x['nmse']<0 and x['si_sdr']>0 and x['weakest_nmse']<=0 for x in deltas if x['count'] in (2,3))
        comparisons.append(dict(mode=mode,deltas=deltas,joint_train_subset_improvement=passed))
    report=dict(status='PASS',protocol_sha256=r['protocol_sha256'],complete_sha256=digest(run/'COMPLETE.json'),
        auditor_sha256=digest(Path(__file__)),all_30_rows_reaggregated=True,source_and_parent_hashes_checked=True,
        recorded_overlap_identity_checks_verified=True,waveforms_independently_reinferred=False,
        independent_pit_disagreements=sum(q['assignment']!=q['independent_pit_assignment'] for q in r['rows']),
        comparisons=comparisons,heldout_read=False,updates=0)
    write(folder/'WINDOW_OVERLAP_AUDIT.json',report)
    text=['# 같은 표본의 창 위치·병합 진단 결과','',
        '고정 TRAIN6, 원 규모 native 부모 U-Net, CPU 18회 추론이다. 세 창이 공유하는 311.04μs의 동일 I/Q를 평가했다. '
        '신규 학습이나 개발·독립 확인 결과가 아니다. 정답은 지표 계산에만 사용했고 추론 순서 정렬·평균에는 쓰지 않았다.','',
        '| 방법 | 두 성분 NMSE ↓ | 두 성분 복소 SI-SDR ↑ dB | 세 성분 NMSE ↓ | 세 성분 복소 SI-SDR ↑ dB |',
        '|---|---:|---:|---:|---:|']
    for mode in p['modes']:
        a,b=get(mode,2),get(mode,3)
        text.append(f"| {mode} | {a['nmse']:.6f} | {a['si_sdr']:.3f} | {b['nmse']:.6f} | {b['si_sdr']:.3f} |")
    text+=['','두·세 평균 NMSE 감소, 복소 SI-SDR 증가, 최약 NMSE 비증가의 공동 조건:']
    text += [f"- {q['mode']}: {'충족' if q['joint_train_subset_improvement'] else '미충족'}" for q in comparisons]
    text+=['',f"정답을 사용한 독립 PIT 대응과 중앙 예측의 고정 대응이 달랐던 행: {report['independent_pit_disagreements']}/30. 모든 수치는 고정 대응을 사용한다.",
        '', '30행의 상대·절대 오차 관계, 전 조건 집계, 코드·체크포인트 해시, 실행 시 동일 표본 검사 기록을 검산했다. '
        '검산에서 파형 추론을 다시 수행한 것은 아니다. 개별 창 이동은 문맥 위치·정규화도 바꾸므로 경계 padding만의 효과로 해석하지 않는다.',
        '', '평균으로 일부 지표가 좋아져도 고정 TRAIN6의 관찰이며 전체 DEV 성능은 별도 검증 대상이다. '
        '단일 창의 638.72μs 지표나 기존 DEV 표와 직접 섞어 비교하지 않는다.',
        '', '[실행 전 계획](WINDOW_OVERLAP_PLAN_KO.md) · [선행 근거](WINDOW_OVERLAP_RELATED_WORK_KO.md) · '
        '[전체 개별 수치](WINDOW_OVERLAP_RESULT.json) · [검산](WINDOW_OVERLAP_AUDIT.json)']
    (folder/'WINDOW_OVERLAP_KO.md').write_text('\n'.join(text)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--folder',type=Path,required=True);a=parser.parse_args()
    print(json.dumps(audit(a.run.resolve(),a.folder.resolve()),ensure_ascii=False))
