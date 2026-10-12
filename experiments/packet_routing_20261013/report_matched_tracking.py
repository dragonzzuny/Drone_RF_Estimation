"""Summarize completed matched ablation and incumbent transfer without selection."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/2026-10-13'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    original=read(ROOT/'local/learned_tracking_20261013_v1/RESULT.json')
    positionless=read(ROOT/'local/positionless_tracking_20261013_v2/RESULT.json')
    check=ROOT/'local/positionless_direct_audit_20261013_v2'
    assert read(check/'STATE.json')['status']=='COMPLETE_AUDITED'
    assert sha(check/'RESULT.json')==read(check/'AUDIT.json')['result_sha256']
    lines=['# 동일 예산 위치 정보 대조 결과','',
        'seed0,FIT48,50epoch/600업데이트,같은 초기화·자료 순서·AdamW 설정이다.',
        '위치 있는 기존 대조는 재학습하지 않았다. 위치 없는 모델은 time_gain=0을 처음부터 고정했다.',
        'B24는 매epoch,C24는 고정 최종50만 평가했다. CPU48 원본 I/Q 재검산을 모두 통과했다.','',
        '|자료|성분|위치 정보|NMSE↓|복소SI-SDR↑ dB|최약NMSE↓|','|---|---:|---|---:|---:|---:|']
    for group in ['B','C']:
        for k in ['1','2','3']:
            for name,d in [('사용',original),('미사용',positionless)]:
                m=d[group]['by_count'][k]
                lines.append(f"|{group}24|{k}|{name}|{m['mean_nmse']:.6f}|{m['mean_si_sdr']:.6f}|{m['weakest_nmse']:.6f}|")
    lines+=['','B 세 성분에서 위치 사용 모델의 NMSE·SI-SDR·최약NMSE가 더 좋았다.',
        'B 두 성분은 위치 사용의 평균NMSE·SI-SDR이 더 좋으나 최약NMSE는 미사용이 더 좋다.',
        'C 두 성분은 미사용이 세 지표 모두 좋다. C 세 성분은 사용의 평균NMSE·SI-SDR이 좋고',
        '최약NMSE는 미사용이 좋다. 시간 위치 정보가 모든 조건에서 유리하다는 결론은 성립하지 않는다.',
        '두 모델 모두 전역 self-attention을 사용하므로 전체 문맥의 유무를 비교한 실험이 아니다.',
        '한 seed의 반복 개발 비교이며 절대 위치 의존과 RF 시간 패턴 학습을 아직 구분하지 못했다.','',
        '등록 수정 이력: 최초v1은 실행 전 저장 경로 충돌을 발견해 GPU lock 대기 중 취소했다.',
        'optimizer0회이며 공개 기존 결과 SHA는 유지됐다. 수정v2에서 실제600업데이트를 완료했다.',
        '[매epoch](POSITIONLESS_TRACKING_EPOCH_REPORTS_KO.md) · [원자료](POSITIONLESS_TRACKING_RESULT.json)',
        '· [직접 I/Q 검산](POSITIONLESS_TRACKING_DIRECT_AUDIT.json)']
    (REPORT/'POSITIONLESS_TRACKING_RESULT_KO.md').write_text('\n'.join(lines)+'\n')
    run=ROOT/'local/incumbent_tracking_c24_20261013_v1'
    assert sha(run/'RESULT.json')==read(run/'AUDIT.json')['result_sha256']
    d=read(run/'RESULT.json')
    lines=['# 기존 native e2의 전체 C24 추적 비교','',
        '기존 짧은 C630 최선 U-Net을 고정하고20.8896ms C24를 평가했다.',
        '원래 짧은1/2/3성분 각1조건 재현과 전체24의 GPU/NumPy 파형 점수 검산을 통과했다.',
        '학습0회. PSD는 고정 특징의 제약 군집이며 learned50은 U-Net99 출력으로 학습한 추적기다.','',
        '|성분|방법|NMSE↓|복소SI-SDR↑ dB|최약NMSE↓|','|---:|---|---:|---:|---:|']
    for k in ['1','2','3']:
        for name in ['parent','psd','learned50']:
            m=d['by_method'][name][k]
            lines.append(f"|{k}|{name}|{m['mean_nmse']:.6f}|{m['mean_si_sdr']:.6f}|{m['weakest_nmse']:.6f}|")
    lines+=['','고정 PSD가 학습 추적기 직접 이식보다 세 지표의 두/세 평균이 좋았다.',
        'native e2의 파형 특성에 맞추지 않은 추적기 이식은 채택하지 않는다. 원인이 교사 출력',
        '분포 차이라는 것은 가설이며,해당 교사로 다시 학습한 비교가 필요하다.',
        'PSD도 parent 대비 단일1/5,두성분2/12,세성분1/7조건에서 평균NMSE가 악화했다.',
        '따라서 평균 개선과 조건별 실패를 함께 보고한다. 단일/두/세 예측 개수 정확도는',
        '.8/.333333/.571429로 복원과 별개 한계가 남는다.',
        '같은 C24에서 native e2+PSD는 B99+학습50보다 두/세 NMSE·SI-SDR·최약NMSE가 모두 좋다.',
        '단일 SI-SDR은 더 낮다. 짧은 C630이나 B24와 섞어 전체 최고 성능으로 주장하지 않는다.',
        '[전체 조건·순열·점수](INCUMBENT_TRACKING_RESULT.json) · [등록 계획](INCUMBENT_TRACKING_PLAN_KO.md)']
    (REPORT/'INCUMBENT_TRACKING_RESULT_KO.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10,'svg.fonttype':'none'})
    runs=[ROOT/'local/learned_tracking_20261013_v1',ROOT/'local/positionless_tracking_20261013_v2']
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,k in zip(axs,['2','3']):
        for path,label,color in zip(runs,['With position','Without position'],['#0072B2','#D55E00']):
            values=[read(path/f'EPOCH_{e:03d}.json')['by_count'][k]['mean_nmse'] for e in range(1,51)]
            ax.plot(range(1,51),values,label=label,color=color)
        ax.set(xlabel='Tracker epoch (12 updates each)',ylabel='Mean I/Q NMSE (lower is better)',
            title=f'{k} sources; B development\nSame 600 optimizer updates',ylim=(0,None))
        ax.grid(alpha=.15);ax.legend(fontsize=9)
    p=REPORT/'figures/matched_tracking_position'
    for ext in ['pdf','svg','png']:fig.savefig(p.with_suffix('.'+ext),dpi=180)
    plt.close(fig)
    p.with_suffix('.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),
        exports={ext:sha(p.with_suffix('.'+ext)) for ext in ['pdf','svg','png']},
        matched_seed=0,updates_each=600,visual_review='PENDING'),indent=2)+'\n')
    print('Reports complete, figure needs visual review')


if __name__=='__main__':main()
