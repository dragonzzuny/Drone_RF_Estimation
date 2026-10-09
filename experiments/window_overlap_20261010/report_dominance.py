"""Verify saved TRAIN48 target-side coverage; no new I/Q reads or inference."""
import argparse
import json
import hashlib
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, public):
    repo=Path(__file__).resolve().parents[2]
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    result=json.loads((root/'COMPLETE.json').read_text())
    assert result==json.loads((public/'SOURCE_DOMINANCE_RESULT.json').read_text())
    assert result['protocol_sha256']==digest(root/'PROTOCOL.json')
    assert result['status']=='COMPLETE' and result['updates']==0
    for rel,sha in protocol['source_sha256'].items():
        assert digest(repo/rel)==digest(root/'source_snapshot'/rel)==sha
    rows=result['rows'];assert len(rows)==100
    assert sorted({r['index'] for r in rows})==sorted(protocol['indices'])
    for index in protocol['indices']:
        case=[r for r in rows if r['index']==index];n=case[0]['count']
        assert len(case)==n and sorted(r['slot'] for r in case)==list(range(n))
        assert sum(r['weakest'] for r in case)==1
        assert min(case,key=lambda r:r['waveform_power'])['weakest']
        assert abs(sum(r['winning_bin_fraction'] for r in case)-1)<1e-12
        for r in case:
            assert [d['threshold'] for d in r['dominance']]==[.5,.8,.95]
            for key in ('source_energy_fraction','bin_fraction'):
                values=[d[key] for d in r['dominance']]
                assert all(0<=v<=1+1e-12 for v in values)
                assert values[0]+1e-12>=values[1]>=values[2]-1e-12
    for s in result['summary']:
        selected=[r for r in rows if r['count']==s['count'] and (s['group']=='all' or r['weakest'])]
        assert len(selected)==s['contributions']
        expected=dict(mean_winning_bin_fraction=statistics.mean(r['winning_bin_fraction'] for r in selected),
            mean_source_energy_in_winning_bins=statistics.mean(r['source_energy_in_winning_bins'] for r in selected),
            median_source_energy_in_winning_bins=statistics.median(r['source_energy_in_winning_bins'] for r in selected),
            median_energy_in_80percent_bins=statistics.median(r['dominance'][1]['source_energy_fraction'] for r in selected))
        for k,v in expected.items():assert abs(s[k]-v)<1e-12
    receipt=dict(status='PASS',protocol_sha256=digest(root/'PROTOCOL.json'),complete_sha256=digest(root/'COMPLETE.json'),
        auditor_sha256=digest(Path(__file__)),cases=48,contributions=100,source_hashes_checked=True,
        aggregation_checked=True,recorded_iq_reads=0,independently_recomputed_stft=False,gpu_use=False,heldout_read=False)
    (public/'SOURCE_DOMINANCE_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    lines=['# 시간·주파수 칸의 성분 우세도: TRAIN48 진단','',
        '기존 고정 TRAIN48 전체의 정답 성분 100개를 분석했다. 두 성분24혼합, 세 성분14혼합이다. 학습·DEV 평가·새 모델 성과가 아니다. STFT512/hop128과 현재 native RF 처리를 그대로 사용했다.','',
        '| 혼합 | 최약 성분 수 | 우세 칸에 들어 있는 자기 에너지 평균 | 중앙값 | 자기 전력 비율≥80% 칸의 자기 에너지 중앙값 |',
        '|---|---:|---:|---:|---:|']
    for s in result['summary']:
        if s['count']>1 and s['group']=='weakest':
            lines.append(f"|{s['count']}성분|{s['contributions']}|{100*s['mean_source_energy_in_winning_bins']:.2f}%|{100*s['median_source_energy_in_winning_bins']:.2f}%|{100*s['median_energy_in_80percent_bins']:.2f}%|")
    lines += ['',
        '세 성분 최약 파형의 상당 부분은 다른 성분이 더 강한 칸과 겹친다. 각 칸을 최강 성분 하나에만 배정하는 라벨은 이 부분의 약한 기여를 표현하지 못한다. 이는 부드러운 기여도 라벨을 검토할 근거이며, 특정 학습법의 개선을 입증하지 않는다.',
        '', '여기서 비율의 분모는 개별 정답 STFT 전력의 합이다. 위상 간섭을 포함한 혼합 STFT 전력과 같다고 가정하지 않는다. 모든 칸을 포함하며 정답에 수신 잡음도 들어 있다. 잡음 구간을 제거한 결과나 분리 가능성의 상한으로 해석하지 않는다. 고정 소수 학습 혼합의 분포라 모집단 통계·독립 검증 결과가 아니다.',
        '', '[전체 100행](SOURCE_DOMINANCE_RESULT.json) · [집계·소스 검산](SOURCE_DOMINANCE_AUDIT.json)']
    (public/'SOURCE_DOMINANCE_KO.md').write_text('\n'.join(lines)+'\n')
    print(receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.public.resolve())
