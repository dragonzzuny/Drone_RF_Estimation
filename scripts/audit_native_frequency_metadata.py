"""Audit frequency geometry of the frozen admitted schedules, without IQ access.

Nyquist intervals are coordinate bounds, not measurements of occupied bandwidth
or receiver passband. No source data, model, split or ongoing trial is changed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run):
    protocol=json.loads((run/'PROTOCOL.json').read_text())
    files={Path(name).name:Path(name) for name in protocol['data_sha256']}
    config=json.loads(files['PREPARATION.json'].read_text())
    manifest=Path(config['manifest'])
    paths=[files['PREPARATION.json'],manifest,files['TRAIN.npy'],files['VALIDATION.npy']]
    for path in paths:
        if digest(path)!=protocol['data_sha256'][str(path)]:
            raise ValueError(f'Frozen metadata changed: {path}')
    clips=json.loads(manifest.read_text())['clips']
    schedules=[]
    for name,role in [('TRAIN.npy','train_pack'),('VALIDATION.npy','validation_pack')]:
        data=np.load(files[name],mmap_mode='r',allow_pickle=False)
        counts=Counter()
        for row in data:
            sources=[clips[int(i)] for i in row['indices'][:int(row['count'])]]
            if any(c['role']!=role or c['fs_hz']!=100_000_000 for c in sources):
                raise ValueError('Unexpected source role or sample rate')
            signature=tuple(sorted((c['category'],c['center_hz'],c['fs_hz']) for c in sources))
            if len({int(center/1e9) for _,center,_ in signature})!=1:
                raise ValueError('Mixed original RF bands')
            counts[signature]+=1
        rows=[]
        for signature,cases in sorted(counts.items()):
            centers=[c for _,c,_ in signature]
            low=max(c-fs/2 for _,c,fs in signature)
            high=min(c+fs/2 for _,c,fs in signature)
            if high<=low:
                raise ValueError('No common digital frequency interval')
            common_center=(low+high)/2
            rows.append(dict(categories=[c for c,_,_ in signature],count=len(signature),cases=cases,
                centers_hz=centers,center_spread_hz=max(centers)-min(centers),
                common_nyquist_lower_hz=low,common_nyquist_upper_hz=high,
                common_nyquist_width_hz=high-low,provisional_common_center_hz=common_center,
                shifts_to_common_center_hz=[c-common_center for c in centers]))
        schedules.append(dict(schedule=name,role=role,cases=len(data),
            epochs=sorted(map(int,np.unique(data['epoch']))),geometry=rows))
    if schedules[0]['cases']!=protocol['epochs']*protocol['examples_per_epoch'] or schedules[1]['cases']!=630:
        raise ValueError('Actual schedule budget changed')
    return dict(status='METADATA_ONLY_COMPLETE',metadata_sha256={str(p):digest(p) for p in paths},
        schedules=schedules,raw_iq_read=False,heldout_read=False,ongoing_trial_changed=False,
        native_offset_replay_implemented=False,
        interval_meaning='theoretical digital Nyquist support; receiver usable passband and occupied signal bandwidth not measured',
        current_training_geometry='same-band center-aligned; native center offsets not preserved')


def markdown(result):
    lines=['# 원 수신 중심주파수 배치: 메타데이터 점검','',
        '현재 전력 배분 대조가 실제 사용하는 12,000학습·630검증 목록과 봉인된 캐시 명세만 읽었다.',
        'I/Q·보류 기종은 읽지 않았고, 학습·자료·분할을 변경하지 않았다. 주파수 차이 보존 합성은 아직 구현·평가하지 않았다.','',
        '아래 간격은 **수신 중심주파수 차이**다. 신호의 실제 점유 대역이나 반송파 간격을 측정한 값은 아니다.',
        '공통 폭은 표본률로 계산한 이론적 디지털 Nyquist 범위의 교집합이다. 실제 수신기 통과대역 검증과 필터 여유 폭이 필요하다.','',
        '| 목록 | 조합 | 혼합 수 | 수신 중심 MHz | 중심 차이 MHz | 공통 디지털 범위 MHz |',
        '|---|---|---:|---|---:|---|']
    for schedule in result['schedules']:
        for row in schedule['geometry']:
            if row['count']==1:
                continue
            lines.append(f"| {schedule['role']} | {' + '.join(row['categories'])} | {row['cases']} | "
                f"{', '.join(f'{c/1e6:g}' for c in row['centers_hz'])} | {row['center_spread_hz']/1e6:g} | "
                f"{row['common_nyquist_lower_hz']/1e6:g}–{row['common_nyquist_upper_hz']/1e6:g} |")
    lines+=['','현재 중심 정렬과 원 주파수 배치를 유지한 합성은 서로 다른 평가 조건이다.',
        '후속 비교에서는 공통 RF 관측 대역을 먼저 고정하고, 각 원기록에서 대응 대역을 필터링한 뒤 동일 주파수 좌표로 변환한다. 혼합과 개별 정답에 같은 RF 필터 응답을 적용한다.',
        '100MS/s 기록을 먼저 그대로 이동시키면 Nyquist 범위를 벗어난 에너지가 접혀 들어올 수 있으므로, 이동 전 대역 제한과 경계 구간 처리를 검증해야 한다.',
        '공통 대역만 남긴 경우 목표는 그 관측 대역 안의 기여 파형이다. 잘린 원기록 전체를 복원했다고 표현하지 않는다.',
        '전력차 설정은 선택한 관측 대역의 긴 구간에서 통일하고 국소 실제 전력차를 기록한다. 기존 중심 정렬 결과와 직접 동일 조건 성능으로 비교하지 않는다.',
        '원 주파수 배치 유지 자체가 실제 동시 수신 재현이나 분리 성능 향상을 입증하지 않는다. 점유 중첩·필터링 영향·미학습 기록 성능은 후속 검증 항목이다.','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=audit(args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    args.output.with_suffix('.md').write_text(markdown(result))
    print(json.dumps(dict(status=result['status'],schedules=[dict(schedule=s['schedule'],cases=s['cases']) for s in result['schedules']])))
