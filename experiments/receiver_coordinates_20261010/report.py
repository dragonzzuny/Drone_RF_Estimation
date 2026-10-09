"""Receipt audit and complete TRAIN6 coordinate report, no I/Q reads."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(root,public):
    p=read(root/'PROTOCOL.json');result=read(root/'COMPLETE.json')
    assert result==read(public/'RECEIVER_COORDINATE_TRAIN6.json') and result['status']=='COMPLETE'
    assert result['protocol_sha256']==sha(root/'PROTOCOL.json') and result['updates']==0
    assert result['inferences']==30 and result['heldout_read'] is False
    assert result['inference_alignment_used_references'] is False
    for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==sha(root/'source_snapshot'/rel)==h
    assert sha(p['parent'])==p['parent_sha256']
    assert sha(Path(p['preparation'])/'PREPARATION.json')==p['preparation_sha256']
    rows=result['rows'];assert len(rows)==36
    assert set(r['index'] for r in rows)==set(p['indices']) and len(p['indices'])==6
    for index in p['indices']:
        part=[r for r in rows if r['index']==index]
        assert [r['steps'] for r in part]==p['steps']+[None]
        assert part[0]['equivariance_relative_error']==0
        for row in part:
            n=row['count'];assert n in (1,2,3)
            assert len(row['nmse'])==len(row['si_sdr'])==len(row['reference_power'])==n
            assert row['reference_power']==part[0]['reference_power']
            assert all(math.isfinite(v) and v>=0 for v in row['nmse'])
            assert all(math.isfinite(v) for v in row['si_sdr'])
            assert 0<=row['sum_relative_error']<1e-10
            if row['steps'] is not None:assert sorted(row['prediction_only_alignment'])==[0,1,2]
    summaries=[]
    for n in (1,2,3):
        for step in p['steps']+[None]:
            part=[r for r in rows if r['count']==n and r['steps']==step];assert len(part)==2
            summaries.append(dict(count=n,steps=step,cases=2,
                mean_nmse=statistics.mean(v for r in part for v in r['nmse']),
                mean_si_sdr=statistics.mean(v for r in part for v in r['si_sdr']),
                weakest_nmse=statistics.mean(r['nmse'][r['weakest_index']] for r in part),
                mean_equivariance_error=statistics.mean(r['equivariance_relative_error'] for r in part) if step is not None else None))
    audit=dict(status='PASS',complete_sha256=sha(root/'COMPLETE.json'),protocol_sha256=sha(root/'PROTOCOL.json'),
        auditor_sha256=sha(Path(__file__)),all_36_rows_reaggregated=True,source_snapshot_and_parent_hashes_checked=True,
        independently_reran_inference=False,gpu_use=False,recorded_iq_reads=0,heldout_read=False,summaries=summaries)
    (public/'RECEIVER_COORDINATE_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# 수신 기준 좌표 변화: 고정 TRAIN6 진단','',
        '기존 TRAIN48의 개수별 첫 두 혼합만 사용했다. 추가 학습 없이 보존 부모의30회 CPU 추론을 검사했다. '
        '혼합과 모든 성분에 동일한 ±1.5625/±3.125MHz 복소 발진자를 적용하고 추정 파형을 원 좌표로 되돌려 평가했다. '
        '실제 RF 위치와 신호 사이 간격은 수신 기준 좌표와 함께 표현하면 유지된다.','',
        '| 성분 수 | 좌표 단계 | 평균 NMSE ↓ | 복소 SI-SDR ↑ dB | 최약 NMSE ↓ |',
        '|---:|---|---:|---:|---:|']
    for r in summaries:
        label=str(r['steps']) if r['steps'] is not None else '5좌표 평균'
        lines.append(f"|{r['count']}|{label}|{r['mean_nmse']:.6f}|{r['mean_si_sdr']:.3f}|{r['weakest_nmse']:.6f}|")
    lines+=['', '5좌표 평균은 정답 없이 예측 파형끼리 세 출력의 순서를 맞춘 뒤 평균했다. '
        '정답은 최종 평가에만 사용했다. 두 성분에서는 평균 NMSE와 복소 SI-SDR이 함께 좋아졌으나 '
        '세 성분에서는 NMSE가 악화했다. 세 성분 약신호 지표만 골라 개선 성공으로 표시하지 않는다.', '',
        '이 작은 진단은 주파수 이동에 완전히 등변인 모델이 아님을 보이지만 고정 위치 의존이 주요 병목이라는 증거는 아니다. '
        '수신 좌표 증강 학습의 효과도 아직 시험하지 않았다. 이 평균 추론을 최선 구성으로 채택하거나 전체 자료 학습을 새로 예약하지 않는다. '
        '현재 우선순위는 진행 중인 시간·주파수 두 축 구조의 같은 학습량 비교다.', '',
        '[RF Transformer 부록 F.1](https://arxiv.org/html/2603.09201v1#A6.SS1)은 작은 간섭 자료의 위상·주파수·크기 변환을 사용한다. '
        '그 논문은 특정 간섭 성분에 변환을 주며, 이번에 모든 성분의 수신 기준을 함께 바꾼 진단과 같은 실험은 아니다. '
        '드론 자료에서 증강 효과가 검증됐다는 근거로 사용하지 않는다.', '',
        '[전체36행](RECEIVER_COORDINATE_TRAIN6.json) · [검산](RECEIVER_COORDINATE_AUDIT.json) · '
        '[I/Q·문맥 변환 수치 검사](RECEIVER_COORDINATE_CPU_CHECK.json)']
    (public/'RECEIVER_COORDINATE_TRAIN6_KO.md').write_text('\n'.join(lines)+'\n')
    return {k:v for k,v in audit.items() if k!='summaries'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();print(run(a.run,a.public))
