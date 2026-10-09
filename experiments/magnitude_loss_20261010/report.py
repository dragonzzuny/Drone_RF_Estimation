"""Publish audited actual/selected waveform and spectral results, including failures."""
import argparse
from pathlib import Path
import statistics
import math
import audit_train

w=audit_train.w


def condition_tables(root,p):
    groups={}
    for arm,path in (('parent',Path(p['baseline'])),('retained_control_e1',Path(p['study'])/'retained_unet/VALIDATION_001.json'),
                     ('magnitude_e1',root/'VALIDATION_001.json')):
        for row in w.read(path)['rows']:
            if row['count']==1:continue
            total=sum(row['reference_power'])
            for i,(name,power,nmse,si) in enumerate(zip(row['categories'],row['reference_power'],row['nmse'],row['si_sdr'])):
                sir=10*math.log10(power/(total-power))
                bin_name='<-20' if sir<-20 else '[-20,-10)' if sir<-10 else '[-10,0)' if sir<0 else '>=0'
                for kind,label in (('category',name),('local_sir_db',bin_name)):
                    key=(arm,row['count'],kind,label)
                    groups.setdefault(key,[]).append((nmse,si))
    return [dict(arm=a,count=n,stratifier=k,label=label,components=len(v),
                 mean_nmse=statistics.mean(x[0] for x in v),mean_si_sdr=statistics.mean(x[1] for x in v))
            for (a,n,k,label),v in sorted(groups.items())]


def run(root,output):
    audit=audit_train.audit(root)
    assert audit['status']=='PASS'
    p=w.read(root/'PROTOCOL.json');r=w.read(root/'COMPLETE.json');spectral=w.read(root/'SPECTRAL.json')
    parent,_=w.validate(Path(p['baseline']),None)
    control,_=w.validate(Path(p['study'])/'retained_unet/VALIDATION_001.json',None)
    # A numerically reproduced e0 is the SAME model, not an improvement.
    adopted=r['selected']['epoch']==1 and r['criterion_met']
    result=dict(status='PASS',protocol_sha256=w.digest(root/'PROTOCOL.json'),
        audit=audit,actual=r['actual'],selected=r['selected'],criterion_met=r['criterion_met'],
        adopted=adopted,spectral_by_arm=[{k:v for k,v in a.items() if k!='rows'} for a in spectral['arms']],
        category_and_actual_local_sir=condition_tables(root,p),
        same_parent_e0_never_an_improvement=True,heldout_read=False,independent_test=False)
    w.write(output.with_suffix('.json'),result)
    lines=['# 비로그 STFT 크기 손실: native RF 1 epoch 완료', '',
        '같은 전체32,142,859파라미터 U-Net·보존 native e2·TRAIN2400혼합·75업데이트·새 AdamW1e-5·실효batch32/microbatch1·seed0다. '
        '대조는 앞서 완료·검산한 동일 일정의 retained_unet e1이다. 후보는 최종 I/Q의 상대 STFT 크기L1을0.1로 추가했다. '
        '원 파형PIT·RF 중심 간격·입력·전처리·개수 손실은 유지했다.', '',
        '| 모델 | 개수 | I/Q NMSE ↓ | 복소 SI-SDR ↑ dB | 최약 성분 NMSE ↓ | 합성 개수 정확도 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,value in [('보존 부모 e0',parent),('원 손실 대조 e1',control),('크기 손실 후보 실제 e1',r['actual'])]:
        for g in value['by_count']:
            lines.append(f"| {name} | {g['count']} | {g['mean_nmse']:.6f} | {g['mean_si_sdr']:.3f} | {g['weakest_nmse']:.6f} | {100*g['construction_count_accuracy']:.2f}% |")
    lines += ['',f"선택은 **e{r['selected']['epoch']}**이며, 새 후보 채택은 **{adopted}**다. 같은 시작 모델e0의 수치 재현을 새로운 개선으로 세지 않는다. "
        '실제e1과 부모·같은 예산 대조의 차이를 모두 보존한다.', '',
        '## 최종 복원 파형의 STFT 진단', '',
        '| 모델 | 개수 | 상대 크기 L1 ↓ | 크기 NMSE ↓ | 크기·위상 교차 항 ↓ | STFT 복소 NMSE ↓ | 스펙트럼 RMS 비 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a in spectral['arms']:
        for g in a['by_count']:
            lines.append(f"| {a['arm']} | {g['count']} | {g['mean_relative_l1']:.6f} | {g['mean_magnitude_nmse']:.6f} | {g['mean_phase_interaction']:.6f} | {g['mean_spectral_complex_nmse']:.6f} | {g['mean_spectral_rms_ratio']:.6f} |")
    lines += ['', '크기 NMSE와 크기·위상 교차 항의 합은 STFT 복소 NMSE와 같다. 후자는 두 크기를 가중치로 포함한 위상 차이 항이며 순수 위상 오차나 독립적인 원인별 기여율이 아니다. '
        'STFT 창의 가중·중첩 때문에 시간영역 I/Q NMSE와의 숫자 일치도 가정하지 않는다. RMS 비는 누출·왜곡까지 포함한 전체 출력 에너지 비이며 정답 성분의 회귀계수가 아니다.', '',
        'JSON에는 기종별·실제 창별 SIR(<−20, −20~−10, −10~0, ≥0dB) 점수도 보존했다. '
        'SIR은 해당 정답 에너지를 다른 정답 에너지 합으로 나눈 값이며, 명목 합성 전력차 또는 상관 교차항을 포함한 혼합 전력으로 대신하지 않는다. '
        '같은 원기록의 반복 성분이므로 성분 수를 독립 표본 수로 해석하지 않는다.', '',
        '## 비교 범위', '',
        'RFUAV 한 자료의 같은 원 RF 대역·100MS/s·원기록 TRAIN8/DEV5묶음이다. 개발630혼합에서1/2/3각210개를 평가했다. '
        '두 성분4기종 조합·세 성분1조합이며, 반복 개발검증·한 seed다. 완전 수렴·다른 seed·독립 확인평가를 입증한 결과로 확대하지 않는다. '
        'Autel과 예약 확인 파일은 열지 않았다. 정답은 손실과 평가에만 쓰고 추론 입력에는 넣지 않았다.', '',
        '실제/선택 가중치, optimizer75단계와 모멘트,630개 파형 평가 행, 세 모델의 스펙트럼1890행, 원 소스·자료 의존성 해시를 검산했다. '
        '일반적인 크기 손실 효과 전체나 선행 음성 연구를 반증한 비교는 아니다.', '',
        '[사전 규약과 원문](MAGNITUDE_PLAN_KO.md) · [CPU 수식·기울기 검사](MAGNITUDE_CPU_CHECK.json) · [실행 결과](MAGNITUDE_RESULT.json)', '']
    w.write(output.with_suffix('.md'),'\n'.join(lines))
    print(dict(status='PASS',selected_epoch=r['selected']['epoch'],adopted=adopted),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.study.resolve(),a.output.resolve())
