"""Audited descriptive comparison; no refitting or model selection."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(epoch):
    out = ROOT / 'reports/2026-10-13'
    control = ROOT / 'local/b_timeblock_spectral_20261013_v1'
    parent = ROOT / 'local/timeblock_extend100_20261012_v1'
    a, b = read(control/'RESULT.json'), read(parent/f'VALIDATION_{epoch:03d}.json')
    audit, event = read(parent/f'AUDIT_{epoch:03d}.json'), read(parent/f'EPOCH_{epoch:03d}.json')
    assert audit['status'] == 'PASS' and audit['event_sha256'] == sha(parent/f'EPOCH_{epoch:03d}.json')
    assert event['validation_sha256'] == sha(parent/f'VALIDATION_{epoch:03d}.json')
    assert read(control/'AUDIT.json')['result_sha256'] == sha(control/'RESULT.json')
    assert a['fixed_dev_schedule_sha256'] == b['rows_sha256']
    assert len(a['rows']) == len(b['rows']) == 630
    groups = defaultdict(list)
    for x, y in zip(a['rows'], b['rows']):
        assert all(x[k] == y[k] for k in ('index', 'count', 'categories', 'weakest_index'))
        np.testing.assert_allclose(x['reference_power'], y['reference_power'], rtol=1e-12, atol=1e-12)
        groups[(x['count'], tuple(x['categories']))].append((x, y))
    rows = []
    for (count, categories), pairs in sorted(groups.items()):
        x = np.array([p[0]['nmse'] for p in pairs]); y = np.array([p[1]['nmse'] for p in pairs])
        undefined = sum(v is None for p in pairs for v in p[0]['si_sdr'])
        rows.append(dict(count=count, categories=list(categories), cases=len(pairs),
            spectral_mean_nmse=float(x.mean()), unet_mean_nmse=float(y.mean()),
            spectral_median_nmse=float(np.median(x)), unet_median_nmse=float(np.median(y)),
            unet_lower_source_nmse=int((y<x).sum()), total_sources=x.size,
            spectral_undefined_si_sources=undefined))
    result = dict(status='VERIFIED',unet_epoch=epoch,by_composition=rows,
        spectral_summary=a['by_count'],unet_summary=b['by_count'],
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in
            [control/'RESULT.json',control/'AUDIT.json',parent/f'VALIDATION_{epoch:03d}.json',parent/f'AUDIT_{epoch:03d}.json']},
        split='B: same recordings, disjoint time blocks; not independent recording generalization',
        dictionary='Known categories; templates fitted once using only 1296 B fit contexts',
        statistics='Descriptive paired cases; overlapping reused recordings are not independent samples')
    (out/f'B_SPECTRAL_VS_E{epoch:03d}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=[f'# B 분할의 스펙트럼 비교 방법과 U-Net epoch{epoch}', '',
        '같은 고정 DEV630 사례·전력·위상·주파수 배치·정답·파형 지표를 사용했다. '
        '기존 NNLS/스펙트럼 마스크 알고리즘의 설정을 유지하고, 템플릿만 B 학습 구간1296개에서 한 번 추정했다. '
        '이전 C 템플릿은 현재 B 평가 시간을 포함할 수 있어 재사용하지 않았다.', '',
        '|신호 수|주파수 템플릿 NMSE↓|U-Net NMSE↓|상대 감소|', '|---|---:|---:|---:|']
    for x,y in zip(a['by_count'],b['by_count']):
        lines.append(f"|{x['count']}|{x['mean_nmse']:.6f}|{y['mean_nmse']:.6f}|{100*(1-y['mean_nmse']/x['mean_nmse']):.2f}%|")
    lines += ['', '|조합|사례 수|템플릿 NMSE↓|U-Net NMSE↓|U-Net 오차가 낮은 성분 수|', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"|{' + '.join(row['categories'])}|{row['cases']}|{row['spectral_mean_nmse']:.6f}|{row['unet_mean_nmse']:.6f}|{row['unet_lower_source_nmse']}/{row['total_sources']}|")
    lines += ['',
        '조합별 예외도 보존한다. Mini3+Mini4Pro에서는 U-Net의 평균·중앙값 NMSE가 '
        '템플릿 방식보다 크다. 나머지 세 가지 두 성분 조합과 세 성분 조합은 U-Net의 평균이 낮다. '
        '단일 Mini 계열에서는 템플릿 방식이 더 낮은 오차인 성분이 많아 평균만으로 모든 사례의 우월성을 주장하지 않는다.', '',
        '두 성분 템플릿 결과에는 SI-SDR을 정의할 수 없는 성분22개가 있어 전체 평균을 숫자로 대체하지 않았다. '
        '단일 성분에서는 입력을 그대로 전달해 높은 SI-SDR을 얻는 사례가 있으므로 단일·다중 성분을 따로 해석한다.', '',
        '템플릿 방식은 알려진 기종 사전과 관측 수신 대역을 사용한다. '
        '하나의 고전 비교 방법을 이긴 결과이며 NMF·모든 신경망 대비 우월성을 뜻하지 않는다. '
        'U-Net은 같은 DEV로 epoch를 선택했고 이 표 역시 개발 평가다. '
        '동일 녹음의 여러 창·혼합은 독립 녹음 반복이 아니므로 유의확률이나 일반화 신뢰구간을 제시하지 않는다.', '']
    (out/f'B_SPECTRAL_VS_E{epoch:03d}_KO.md').write_text('\n'.join(lines))
    print(json.dumps(result['by_composition'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,required=True);main(p.parse_args().epoch)
