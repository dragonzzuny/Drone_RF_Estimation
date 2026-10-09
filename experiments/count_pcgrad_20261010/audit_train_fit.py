"""Reaggregate fixed TRAIN48 diagnoses, verify provenance, and report limits.

This checks saved numerical receipts, not an independent second inference run.
It never reads I/Q or the reserved confirmation recordings.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit(run,public):
    p=read(run/'PROTOCOL.json');result=read(run/'COMPLETE.json')
    assert result==read(public/'COUNT_TRAIN_FIT.json') and result['status']=='COMPLETE'
    assert result['protocol_sha256']==digest(run/'PROTOCOL.json')
    assert result['model_updates']==0 and result['heldout_read'] is False
    for rel,sha in p['source_sha256'].items():assert digest(ROOT/rel)==sha
    prior_path=public/'SOURCE_INTERACTION_TRAIN_FIT.json'
    assert digest(prior_path)==p['prior_sha256']
    prior=read(prior_path);indices=p['indices']
    assert indices==prior['train_indices'] and len(indices)==len(set(indices))==48
    for arm,sha in p['checkpoints'].items():
        folder=ROOT/f'local/count_{arm}_20261010_v1'
        assert digest(folder/'ACTUAL_001.pt')==sha
        a=read(public/f'COUNT_{arm.upper()}_FINAL_AUDIT.json')
        assert a['status']=='PASS' and a['complete_sha256']==digest(folder/'COMPLETE.json')
    models=('parent/e0','retained_unet/e1','pcgrad/e1','cagrad/e1')
    rows=result['rows'];assert len(rows)==192
    for model in models:
        selected=[r for r in rows if r['model']==model]
        assert [r['index'] for r in selected]==indices
        assert [sum(r['count']==c for r in selected) for c in (1,2,3)]==[10,24,14]
        for r in selected:
            assert len(r['nmse'])==len(r['si_sdr'])==len(r['reference_power'])==r['count']
            assert all(math.isfinite(v) and v>=0 for v in r['nmse'])
            assert all(math.isfinite(v) and v>0 for v in r['reference_power'])
            assert all(math.isfinite(v) for v in r['si_sdr'])
            assert r['weakest_index']==min(range(r['count']),key=lambda i:r['reference_power'][i])
            assert r['predicted_count'] in (1,2,3) and 0<=r['sum_relative_error']<1e-10
        if model in models[:2]:assert selected==[r for r in prior['rows'] if r['model']==model]
    summaries=[]
    for model in models:
        for count in (1,2,3):
            part=[r for r in rows if r['model']==model and r['count']==count]
            summaries.append(dict(model=model,count=count,cases=len(part),
                mean_nmse=statistics.mean(v for r in part for v in r['nmse']),
                mean_si_sdr=statistics.mean(v for r in part for v in r['si_sdr']),
                weakest_nmse=statistics.mean(r['nmse'][r['weakest_index']] for r in part)))
    assert summaries==result['summaries']
    backend=result['backend_checks'];assert len(backend)==6
    assert {r['model'] for r in backend}=={'pcgrad','cagrad'}
    assert max(r['nmse_max_error'] for r in backend)<2e-5
    assert max(r['si_sdr_max_error'] for r in backend)<2e-3
    evidence=dict(status='PASS',complete_sha256=digest(run/'COMPLETE.json'),
        protocol_sha256=digest(run/'PROTOCOL.json'),auditor_sha256=digest(Path(__file__)),
        all_192_rows_reaggregated=True,unchanged_train48_and_prior_rows_checked=True,
        source_and_checkpoint_hashes_checked=True,backend_spot_checks=6,
        independently_reran_inference=False,gpu_use=False,recorded_iq_reads=0,heldout_read=False)
    (public/'COUNT_TRAIN_FIT_AUDIT.json').write_text(json.dumps(evidence,indent=2)+'\n')
    text=['# 개수별 업데이트 방법의 고정 TRAIN48 적합도',
        '', 'PCGrad·CAGrad의 실제 추가 1epoch 모델을 같은 고정 TRAIN48에서 CPU로 평가했다. '
        '기존 부모·같은 예산 원 손실 대조의 결과는 앞선 검산 자료에서 그대로 가져왔다. '
        '추가 학습이나 확인용 파일 접근은 없었다.', '',
        '| 모델 | 성분 수 | 혼합 수 | NMSE ↓ | 복소 SI-SDR ↑ dB | 최약 NMSE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        text.append(f"| {r['model']} | {r['count']} | {r['cases']} | {r['mean_nmse']:.6f} | {r['mean_si_sdr']:.3f} | {r['weakest_nmse']:.6f} |")
    text+=['','PCGrad와 CAGrad 모두 이 학습 부분집합에서는 부모보다 두·세 성분 평균 NMSE가 줄고 복소 SI-SDR이 높아졌다. '
        '반면 개발검증에서는 부모보다 악화했다. 모델이 전혀 학습되지 않았다는 설명과는 맞지 않는다. '
        '원 손실 대조보다 일관되게 나아진 것도 아니며, 기록 차이·대역폭 차이·과적합 중 하나를 원인으로 확정하지 않는다.',
        '', '48혼합은 기존에 고정한 학습 진단 부분집합이며 독립 평가 자료가 아니다. '
        '개발 630혼합과 창별 전력·활동 분포도 다르므로 두 표의 차이를 순수한 일반화 격차로 해석하지 않는다. '
        '학습 예제 적합도와 개발 성능을 구분해 해석한다.', '',
        '두 새 모델마다 개발 사례를 개수별 하나씩 재추론해 CPU/GPU 지표 차이를 점검했다. '
        f"NMSE 최대 차이는 {max(r['nmse_max_error'] for r in backend):.3g}, "
        f"복소 SI-SDR 최대 차이는 {max(r['si_sdr_max_error'] for r in backend):.3g}dB였다. "
        '이 점검은 모든 파형의 독립 재추론을 뜻하지 않는다.', '',
        '[전체 수치](COUNT_TRAIN_FIT.json) · [192행·체크포인트·소스 검산](COUNT_TRAIN_FIT_AUDIT.json) · '
        '[개발검증 전체 결과](COUNT_OPTIMIZATION_REPORT_KO.md)']
    (public/'COUNT_TRAIN_FIT_KO.md').write_text('\n'.join(text)+'\n')
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True);args=parser.parse_args()
    print(audit(args.run,args.public))
