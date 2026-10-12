"""Reproducible report of all fixed-final tracking results, including adverse cases."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/2026-10-13'
RUN=ROOT/'local/learned_tracking_20261013_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result=read(RUN/'RESULT.json')
    assert sha(RUN/'RESULT.json')==read(RUN/'AUDIT.json')['result_sha256']
    audit=ROOT/'local/tracking_direct_audit_20261013_v1'
    assert sha(audit/'RESULT.json')==read(audit/'AUDIT.json')['result_sha256']
    assert read(audit/'RESULT.json')['status']=='PASS'
    base=read(ROOT/'local/constrained_routing_20261013_v1/RESULT.json')
    lines=['# 고정 U-Net 파형의 학습 기반 출력 추적 결과', '',
        '2026-10-13. 사전 고정한 최종50epoch/600업데이트,seed0. 전체 U-Net99는 고정했고',
        '875,521파라미터 Transformer가 출력 PSD·상대전력·시간 위치로 동일 신호 관계를 학습했다.',
        '정답은 FIT48 지도학습과 평가에만 사용한다. 추론은 기존 추정 파형의 조각 순열만 변경한다.', '',
        'B24는 같은 원녹음의 시간 분리, C24는 다른 원녹음과 VTSBW가 함께 바뀐 반복 개발자료다.',
        '각각 단일5/두성분12/세성분7조건의20.8896ms 전체 파형이며 짧은630 평가와 직접 순위 비교하지 않는다.', '',
        '|자료|성분 수|방법|NMSE↓|복소SI-SDR↑ dB|최약NMSE↓|',
        '|---|---:|---|---:|---:|---:|']
    comparisons=[]
    for group,key in [('B24','B'),('C24','C')]:
        old=base['groups'][group]
        for k in (1,2,3):
            for label,m in [('U-Net99+C8',old['summaries']['parent']['by_count'][k-1]),
                            ('고정 PSD 군집',old['summaries']['candidate']['by_count'][k-1]),
                            ('학습 추적기50',result[key]['by_count'][str(k)])]:
                lines.append(f"|{group}|{k}|{label}|{m['mean_nmse']:.6f}|{m['mean_si_sdr']:.6f}|{m['weakest_nmse']:.6f}|")
        prior={r['index']:r for r in old['cases']}
        for row in result[key]['rows']:
            before=prior[row['index']]['candidate']
            delta=float(np.mean(row['nmse'])-np.mean(before['nmse']))
            comparisons.append(dict(group=group,index=row['index'],count=row['count'],
                categories=row['categories'],nominal_levels_db=row['nominal_levels_db'],
                nmse_change_vs_psd=delta,improved=delta < -1e-10,worse=delta>1e-10,
                parent=prior[row['index']]['parent'],psd=before,learned50=row))
    lines.extend(['', '|자료|성분 수|개선/악화/동일 조건 수 대 PSD|','|---|---:|---:|'])
    for group in ('B24','C24'):
        for k in (1,2,3):
            rows=[r for r in comparisons if r['group']==group and r['count']==k]
            good=sum(r['improved'] for r in rows);bad=sum(r['worse'] for r in rows)
            lines.append(f'|{group}|{k}|{good}/{bad}/{len(rows)-good-bad}|')
    lines.extend(['',
        'B 두/세 성분 NMSE가 고정 PSD 대비37.6%/50.6% 감소했다. 세 성분7조건 중6개가',
        '개선되고1개는 동일했다. 두 성분에는4개 악화 조건이 남는다. C의 두/세 평균과 최약',
        'NMSE도 개선됐으나 단일 평균은 .096837→.097217로 소폭 악화했다. 모든 조건의',
        '공동 무악화 기준을 충족하지 않으므로 기존 C630 CURRENT_BEST를 교체하지 않는다.', '',
        '별도 CPU 경로에서48개 원본 추정 I/Q의 특징,군집 순서,직접 파형 점수를 재현했다.',
        '학습 구간의 affinity 손실 .364917→.009648 및 FIT 두/세 NMSE .082152/.268225는',
        '학습 동작의 진단이다. 독립 일반화 근거로 사용하지 않는다. 입력 자료의 다양성이나',
        '교사 파형 품질에 대한 한계는 남으며,완전 분리·기체 식별 개선을 입증한 결과가 아니다.', '',
        '50epoch는48개 사전 계산 특징을 각12업데이트씩 반복한600업데이트다. 전체 파형',
        'U-Net 학습50epoch와 예산이 같지 않다. 47epoch의 좋은 두 성분 결과를 사후 선택하지 않았다.',
        '기존 음성 분리의 학습 임베딩·군집·시간 배정과 연결되는 방법으로,추적 개념 자체의',
        '최초성을 주장하지 않는다. 기여 후보는 RF 원기록 기여 파형의 장구간 배정 문제 진단과',
        '이 문제를 줄이는 학습 방식 및 조건별 검증이다.', '',
        '다음 등록 평가는 기존 C630 최선 native e2에 같은 최종50 추적기를 수정 없이 적용한다.',
        '[후속 계획](INCUMBENT_TRACKING_PLAN_KO.md).', '',
        '[직접 I/Q 감사](LEARNED_TRACKING_DIRECT_AUDIT.json) · [전체 결과](LEARNED_TRACKING_RESULT.json)',
        '· [전 epoch 기록](LEARNED_TRACKING_EPOCH_REPORTS_KO.md) · [선행 연구](ROUTING_PRIOR_ART_2026_KO.md)'])
    (REPORT/'LEARNED_TRACKING_RESULT_KO.md').write_text('\n'.join(lines)+'\n')
    (REPORT/'LEARNED_TRACKING_CASE_COMPARISON.json').write_text(json.dumps(dict(
        cases=comparisons,result_sha256=sha(RUN/'RESULT.json'),direct_audit_sha256=sha(audit/'RESULT.json')),
        ensure_ascii=False,indent=2)+'\n')
    epochs=[read(RUN/f'EPOCH_{e:03d}.json') for e in range(1,51)]
    ep=['# 학습 추적기 매epoch 기록','', '모든 지표는 B24 전체20.8896ms. 각 칸은 두/세 성분. 최종50 고정.', '',
        '|Epoch|NMSE↓|복소SI-SDR↑ dB|최약NMSE↓|직전 대비 NMSE|', '|---:|---:|---:|---:|---:|']
    prev=read(RUN/'B_000.json')['by_count']
    for d in epochs:
        a,b=d['by_count']['2'],d['by_count']['3'];x,y=prev['2'],prev['3']
        ep.append(f"|{d['epoch']}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.4f}/{b['mean_si_sdr']:.4f}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{a['mean_nmse']-x['mean_nmse']:+.6f}/{b['mean_nmse']-y['mean_nmse']:+.6f}|")
        prev=d['by_count']
    (REPORT/'LEARNED_TRACKING_EPOCH_REPORTS_KO.md').write_text('\n'.join(ep)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,k,color in zip(axs,(2,3),('#0072B2','#D55E00')):
        ax.plot(range(1,51),[v['by_count'][str(k)]['mean_nmse'] for v in epochs],color=color,label='Learned tracking')
        for name,label,style in [('parent','U-Net99 + C8',':'),('candidate','Fixed PSD clustering','--')]:
            value=base['groups']['B24']['summaries'][name]['by_count'][k-1]['mean_nmse']
            ax.axhline(value,color='0.35',ls=style,label=label)
        ax.set(xlabel='Tracker epoch (12 updates each)',ylabel='Mean I/Q NMSE (lower is better)',
            title=f'{k} sources; {12 if k==2 else 7} B development mixtures',ylim=(0,None))
        ax.grid(alpha=.15);ax.legend(fontsize=8)
    p=REPORT/'figures/learned_tracking_curve'
    for ext in ('pdf','svg','png'):fig.savefig(p.with_suffix('.'+ext),dpi=180)
    plt.close(fig)
    manifest=dict(source_sha256=sha(Path(__file__)),input_sha256=sha(RUN/'RESULT.json'),
        exports={ext:sha(p.with_suffix('.'+ext)) for ext in ('pdf','svg','png')},
        scope='All50 epochs, B24 only; full waveform teacher frozen; final50 fixed',visual_review='PENDING')
    p.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('WROTE reports and figure; visual review still required')


if __name__=='__main__':main()
