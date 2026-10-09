"""Read-only audit of selective auxiliary and the four fixed comparisons."""
import argparse
import importlib.util
from pathlib import Path
import audit_train

# Legacy worker imports prepend other experiment directories containing report.py.
spec=importlib.util.spec_from_file_location('magnitude_reporting',Path(__file__).with_name('report.py'))
common=importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

w=audit_train.w


def run(root,output):
    audit=audit_train.audit(root);assert audit['status']=='PASS'
    p=w.read(root/'PROTOCOL.json');r=w.read(root/'COMPLETE.json')
    check=w.ROOT/'reports/2026-10-10/SELECTIVE_MAGNITUDE_CPU_CHECK.json'
    assert w.digest(check)==p['selective_cpu_check_sha256'] and w.read(check)['status']=='PASS'
    assert p['objective_callback'].startswith('selective.objective') and p['updates']==75
    all_count_root=Path(p['predecessor']);all_count=w.read(all_count_root/'COMPLETE.json')
    assert all_count['protocol_sha256']==w.digest(all_count_root/'PROTOCOL.json')
    parent,ids=w.validate(Path(p['baseline']),None)
    control,_=w.validate(Path(p['study'])/'retained_unet/VALIDATION_001.json',ids)
    previous,_=w.validate(all_count_root/'VALIDATION_001.json',ids)
    assert previous==all_count['actual']
    models=[('보존 부모 e0',parent),('원 손실 대조 e1',control),('전체 개수 크기 손실 e1',previous),('다중 성분만 크기 손실 e1',r['actual'])]
    deltas=[]
    for count in (1,2,3):
        a,b=[next(g for g in v['by_count'] if g['count']==count) for v in (previous,r['actual'])]
        deltas.append(dict(count=count,nmse_delta=b['mean_nmse']-a['mean_nmse'],complex_si_delta=b['mean_si_sdr']-a['mean_si_sdr'],weakest_nmse_delta=b['weakest_nmse']-a['weakest_nmse']))
    adopted=r['selected']['epoch']==1 and r['criterion_met']
    spectra=w.read(root/'SPECTRAL.json')
    result=dict(status='PASS',protocol_sha256=w.digest(root/'PROTOCOL.json'),audit=audit,
        selected=r['selected'],actual=r['actual'],adopted=adopted,
        previous_all_count_deltas=deltas,models=[dict(name=n,**v) for n,v in models],
        spectral_by_arm=[{k:v for k,v in a.items() if k!='rows'} for a in spectra['arms']],
        category_and_actual_local_sir=common.condition_tables(root,p),
        heldout_read=False,independent_test=False,adaptive_followup=True)
    w.write(output.with_suffix('.json'),result)
    lines=['# 단일 성분 크기 감독 제거: 완료한1요인 대조','',
        '전체 U-Net32,142,859파라미터·같은 보존 native e2·TRAIN2400·75업데이트·새 AdamW1e-5·실효batch32/microbatch1·seed0다. '
        '단일 성분800혼합도 원 손실로 학습하고 두·세 성분에만 같은0.1비로그 크기 항을 적용했다. '
        '앞선 후보 가중치를 이어 쓰지 않았다. 추론에 실제 개수·기종·원신호를 넣지 않았다.','',
        '| 모델 | 성분 수 | NMSE ↓ | 복소 SI-SDR ↑ dB | 최약 NMSE ↓ | 개수 정확도 |','|---|---:|---:|---:|---:|---:|']
    for name,value in models:
        for g in value['by_count']:
            lines.append(f"|{name}|{g['count']}|{g['mean_nmse']:.6f}|{g['mean_si_sdr']:.3f}|{g['weakest_nmse']:.6f}|{100*g['construction_count_accuracy']:.2f}%|")
    lines+=['',f"선택은e{r['selected']['epoch']}, 새 모델 채택은 **{adopted}**다. 부모와 동일한e0를 개선으로 세지 않는다.",'',
        '## 전체 개수 크기 감독 대비 실제e1 변화','',
        '| 성분 수 | NMSE 변화 | 복소 SI-SDR 변화 dB | 최약 NMSE 변화 |','|---|---:|---:|---:|']
    for g in deltas:lines.append(f"|{g['count']}|{g['nmse_delta']:+.6f}|{g['complex_si_delta']:+.4f}|{g['weakest_nmse_delta']:+.6f}|")
    lines+=['','두·세 성분의 개선 여부와 부모·원 손실 대조를 넘는지는 별도 판단이다. '
        '이는 앞선 결과를 본 뒤 정한 개발 대조이며, 감독 적용 범위의 효과를 이 설정에서만 확인한다. '
        '하나의 결과로 학습 간섭의 보편적인 원인을 확정하지 않는다.','',
        '원 소스·자료·부모·대조 결과 해시, 실제/선택 가중치와optimizer75단계·모멘트,630파형 행 및1890스펙트럼 행을 검산했다. '
        '각 기종·실제SIR 조건과 스펙트럼 크기·위상 항은JSON에 보존했다. 스펙트럼 교차 항은 크기를 가중치로 포함하며 순수 위상 오차와 다르다.','',
        'RFUAV 한 자료·동일 원 대역·100MS/s·원기록TRAIN8/DEV5묶음·두 성분4기종 조합/세 성분1조합이다. '
        '반복 개발검증·한 seed이며 독립 확인·일반적인 수렴·물리 드론 개수 검증을 대신하지 않는다. Autel과 예약 확인 파일은 열지 않았다.','',
        '[사전 규약](SELECTIVE_MAGNITUDE_PLAN_KO.md) · [앞선 크기 손실 대조](MAGNITUDE_FINAL.md) · [출력 기울기 진단](MAGNITUDE_GRADIENT_KO.md)','']
    w.write(output.with_suffix('.md'),'\n'.join(lines))
    print(dict(status='PASS',adopted=adopted,selected_epoch=r['selected']['epoch'],deltas=deltas),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.study.resolve(),a.output.resolve())
