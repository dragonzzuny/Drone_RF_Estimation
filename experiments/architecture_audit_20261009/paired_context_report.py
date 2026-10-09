"""Descriptive paired development differences; no inference or significance test."""
import argparse
from pathlib import Path
import statistics as stats

import phase_watch
import watch_epochs as watch


def run(study, output):
    snapshot = phase_watch.snapshot(study)
    common = snapshot['common_epoch']
    if common < 1:
        raise ValueError('No common completed epoch')
    results = []
    identities = None
    for epoch in range(1, common+1):
        values, hashes = [], {}
        for arm in ('local','long'):
            p = study/arm/f'VALIDATION_{epoch:03d}.json'
            _, identities = watch.validate(p, identities)
            values.append(sorted(watch.read(p)['rows'],key=lambda r:r['index']))
            hashes[arm] = watch.digest(p)
        for count in (2,3):
            pairs = [(a,b) for a,b in zip(*values) if a['count']==count]
            if len(pairs)!=210 or any(a['index']!=b['index'] for a,b in pairs):
                raise ValueError('Mixture correspondence changed')
            nmse = [stats.mean(b['nmse'])-stats.mean(a['nmse']) for a,b in pairs]
            si = [stats.mean(b['si_sdr'])-stats.mean(a['si_sdr']) for a,b in pairs]
            weak_nmse,weak_si,all_better = [],[],[]
            for a,b in pairs:
                j = a['weakest_index']
                if j!=b['weakest_index']:
                    raise ValueError('Weak reference identity changed')
                weak_nmse.append(b['nmse'][j]-a['nmse'][j])
                weak_si.append(b['si_sdr'][j]-a['si_sdr'][j])
                all_better.append(all(b['nmse'][i]<a['nmse'][i] and
                    b['si_sdr'][i]>a['si_sdr'][i] for i in range(count)))
            results.append(dict(epoch=epoch,count=count,cases=210,
                validation_sha256=hashes,
                mean_nmse_delta=stats.mean(nmse),median_nmse_delta=stats.median(nmse),
                mean_si_delta=stats.mean(si),median_si_delta=stats.median(si),
                mixture_mean_both_better=sum(n<0 and s>0 for n,s in zip(nmse,si)),
                weakest_both_better=sum(n<0 and s>0 for n,s in zip(weak_nmse,weak_si)),
                every_source_both_better=sum(all_better)))
    report = dict(status='COMPLETE_SNAPSHOT',common_completed_epoch=common,rows=results,
        source_sha256=watch.digest(Path(__file__)),protocol_sha256=snapshot['protocol_sha256'],
        waveform_reads=0,heldout_read=False,selection_changed=False,independent_test=False,
        difference='long minus local; negative NMSE and positive SI-SDR differences favor long',
        scope='Same actual epoch, no checkpoint cherry-picking. Correlated crops, one training seed; descriptive only',
        significance_test=False)
    watch.write(output.with_suffix('.json'),report)
    lines=['# 긴 복소 문맥: 같은 혼합·같은 epoch의 대응 비교','',
        '각행은 같은 epoch의 실제 가중치와 같은210개 개발 혼합을 비교한다. '
        '차이는 long−local이며 NMSE는 음수, SI-SDR은 양수일 때 긴 문맥군이 좋다. '
        '두 군 모두 전체 혼합 RMS와 시간평균 전력 특징을 받는다. '
        '한seed·서로 상관된 원기록 창의 사후 기술 통계이며 독립 시험이나 유의성 검정이 아니다.','',
        '| e | 성분 수 | 평균 NMSE 차이 | NMSE 차이 중앙값 | 평균 SI-SDR 차이 dB | SI 차이 중앙값 dB | 혼합 평균 둘 다 개선 | 약한 성분 둘 다 개선 | 모든 성분 각각 둘 다 개선 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"|{r['epoch']}|{r['count']}|{r['mean_nmse_delta']:.6f}|{r['median_nmse_delta']:.6f}|"
            f"{r['mean_si_delta']:.3f}|{r['median_si_delta']:.3f}|{r['mixture_mean_both_better']}/210|"
            f"{r['weakest_both_better']}/210|{r['every_source_both_better']}/210|")
    lines+=['','마지막 세 열은 서로 다른 판정이며, 혼합 평균이 좋아졌다고 모든 성분이 개선된 것은 아니다. '
        '각 성분은 기존 전체 창 PIT 대응으로 같은 정답 성분에 정렬돼 있다. '
        '한 epoch에서의 우위를 안정적 학습 효과나 낮은 절대 복원 오차로 대신하지 않는다.','']
    watch.write(output.with_suffix('.md'),'\n'.join(lines))
    print(f'Checked {common} common epochs; no new waveform reads or updates')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.study.resolve(),a.output.resolve())
