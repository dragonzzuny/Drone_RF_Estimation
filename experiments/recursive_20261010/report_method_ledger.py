"""Same-parent additional-75-update DEV ledger; preserve actual failed e1."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/architecture_audit_20261009'))
import watch_epochs as watch

def run():
    old=watch.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    parent,ids=watch.validate(Path(old['baseline']),None)
    entries=[
        ('원 손실','source_interaction_comparison_20261010_v1/retained_unet',32142859,32142859,2400),
        ('PCGrad','count_pcgrad_20261010_v1',32142859,32142859,2400),
        ('CAGrad','count_cagrad_20261010_v1',32142859,32142859,2400),
        ('파형 업데이트 보호','wave_update_guard_20261010_v1',32142859,32142859,2400),
        ('주파수·시간 보강층','tf_axis_20261010_v1',37406475,37406475,2400),
        ('보강층 학습률 증가','tf_axis_adaptation_20261010_v1/joint_adapter_lr',37406475,37406475,2400),
        ('보강층만 학습','tf_axis_adaptation_20261010_v1/frozen_backbone',37406475,5263616,2400),
        ('두 창 지도학습','paired_window_20261010_v1/paired_supervision',32142859,32142859,4800),
        ('두 창+일관성','paired_window_20261010_v1/paired_consistency',32142859,32142859,4800),
        ('친화도0 대조','source_affinity_20261010_v1/affinity_control',32143899,32143899,2400),
        ('최강 성분 친화도','source_affinity_20261010_v1/hard_affinity',32143899,32143899,2400),
        ('부드러운 성분 친화도','source_affinity_20261010_v1/soft_affinity',32143899,32143899,2400),
        ('같은 입력2회 대조','successive_pit_20261010_v1/retained_two_pass_control',32142859,32142859,2400),
        ('공유 U-Net 순차 추출','successive_pit_20261010_v1/successive_pit',32142859,32142859,2400)]
    rows=[];pending=[]
    for label,relative,parameters,trainable,windows in entries:
        folder=ROOT/'local'/relative;vp=folder/'VALIDATION_001.json';ep=folder/'EPOCH_001.json'
        if not vp.exists() or not ep.exists():pending.append(label);continue
        event=watch.read(ep);assert event['epoch']==1 and event['updates']==75
        value,_=watch.validate(vp,ids)
        passed=True;deltas=[]
        for count in (2,3):
            a,b=[next(r for r in v['by_count'] if r['count']==count) for v in (value,parent)]
            dn=a['mean_nmse']-b['mean_nmse'];dw=a['weakest_nmse']-b['weakest_nmse']
            ds=None if a['mean_si_sdr'] is None else a['mean_si_sdr']-b['mean_si_sdr']
            passed=passed and dn<0 and ds is not None and ds>0 and dw<=0
            deltas.append(dict(count=count,nmse_delta=dn,si_sdr_delta=ds,weakest_nmse_delta=dw))
        rows.append(dict(label=label,parameters=parameters,trainable=trainable,
            unique_long_mixtures=2400,distinct_window_views=windows,updates=75,
            actual_e1=value,parent_joint_criterion=bool(passed),parent_deltas=deltas,
            validation_sha256=hashlib.sha256(vp.read_bytes()).hexdigest(),
            epoch_receipt_sha256=hashlib.sha256(ep.read_bytes()).hexdigest()))
    result=dict(status='SNAPSHOT',rows=rows,pending=pending,parent=parent,
        parent_checkpoint_sha256=old['parent_checkpoint_sha256'],all_same_630_identities_checked=True,
        data_reads=0,inference=0,heldout_read=False,independent_test=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='Same additional updates does not imply same total pretraining, parameters, window views, or computation. Parent criterion does not replace matched-control requirement.')
    public=ROOT/'reports/2026-10-10';watch.write(public/'METHOD_LEDGER.json',result)
    lines=['# 방법 개발의 실제1epoch 결과: 같은 부모와 추가75업데이트','',
        '각 방법의 실제e1을 같은630개 DEV 혼합의 정답·기종·기록·전력 지문으로 대조했다. '
        '선택이e0여도 실패한e1을 숨기지 않는다. 모든 군의 원본 긴 학습혼합은2400개이며, 두 창 군만 서로 다른4800창을 본다. '
        '추가업데이트 수가 같아도 파라미터·계산량·총 사전학습 이력이 같은 것은 아니다.','',
        '| 방법 | 총/학습 파라미터M | NMSE2/3 ↓ | 복소SI-SDR2/3 ↑ dB | 최약NMSE2/3 ↓ | 부모 공동 기준 |',
        '|---|---|---|---|---|---|']
    def fmt(value,n=6):return '미정의' if value is None else f'{value:.{n}f}'
    for row in [dict(label='학습 전 부모',parameters=32142859,trainable=0,actual_e1=parent,parent_joint_criterion=None)]+rows:
        a,b=row['actual_e1']['by_count'][1:]
        met='기준' if row['parent_joint_criterion'] is None else '충족' if row['parent_joint_criterion'] else '미충족'
        lines.append(f"|{row['label']}|{row['parameters']/1e6:.3f}/{row['trainable']/1e6:.3f}|{fmt(a['mean_nmse'])}/{fmt(b['mean_nmse'])}|{fmt(a['mean_si_sdr'],3)}/{fmt(b['mean_si_sdr'],3)}|{fmt(a['weakest_nmse'])}/{fmt(b['weakest_nmse'])}|{met}|")
    lines+=['','부모 공동 기준은 두·세 성분 모두의 평균NMSE 감소·복소SI-SDR 증가·최약NMSE 비악화다. '
        '각 후보의 정식 채택에는 자기 실험의 같은 예산 대조군도 넘어야 하며, 이 표의 부모 기준으로 대신하지 않는다. '
        '단일 성분 지표와 개수 정확도는JSON의 세 개수 전체 집계에 보존했다.',
        '', '아직 e1 수치가 없는 등록 조건: '+', '.join(pending)+'.',
        '', '복원 정답은 공통 관측RF대역의 원기록 기여 파형이며 수신 잡음을 포함한다. '
        '단일seed·반복DEV·5개 개발 기록 묶음·한 세 기종 조합의 한계를 유지한다. '
        'CPU TRAIN 진단이나 정답 이용 제거 결과를 이 DEV 비교에 섞지 않는다. '
        '네 위상 평균 추론은 네 번 실행하는 별도 설정이므로 여기의 단일 추론 부모와 구분한다.',
        '', '[전체 집계·지문](METHOD_LEDGER.json)']
    (public/'METHOD_LEDGER_KO.md').write_text('\n'.join(lines)+'\n')
    print(dict(completed=len(rows),pending=pending))

if __name__=='__main__':run()
