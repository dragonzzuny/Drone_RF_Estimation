"""Small public summary of verified numerical stage diagnosis."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'local/packet_routing_20261013_e079_v2'
PUBLIC=ROOT/'reports/2026-10-13'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result=json.loads((P/'RESULT.json').read_text())
    audit=json.loads((P/'VERIFICATION.json').read_text())
    assert audit['status']=='PASS' and audit['input_hashes_unchanged']
    for name,h in audit['artifacts'].items():
        assert sha(P/name)==h,name
    assert result['original_e79_max_difference']==0
    windows=json.loads((P/'WINDOWS.json').read_text())
    summary=dict(status='COMPLETE_VERIFIED',epoch=79,reference_event='Previously exposed B FPV/Mavic3Pro 2ms development event',
        reproduction_max_sample_difference=0.,stage_metrics={k:v['metrics'] for k,v in result['stages'].items()},
        waveform_length_samples=200000,fs_hz=100000000,
        within_window_FPV_dominant_slot={str(w['index']):{s:[b['dominant_source_slot'][0] for b in w['stage'][s]['blocks']]
            for s in ('raw','zero','C8')} for w in windows},
        artifact_sha256={name:sha(P/name) for name in ('PLAN.json','RESULT.json','WINDOWS.json','VERIFICATION.json')},
        diagnosis='Within-window source-role changes exist before equal projection, phase averaging and overlap assembly; causal remedy remains untested',
        caveats=['One adaptively examined development event','Joint reference LS coefficients are not additive power fractions',
                 'Unprojected common mode is unidentifiable; pre/post metrics are not an isolated causal ablation',
                 'Low-activity blocks require separate interpretation'])
    (PUBLIC/'PACKET_ROUTING_E079.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    text='''# FPV 출력 배정의 단계별 진단 — epoch79\n\n원래 저장된 2ms 복원을 표본 단위로 **정확히 재현**했다(최대 차이0).\n기존 입력·참조·특징·체크포인트 해시와 종료 후 입력 해시 검산을 통과했다.\n\n|처리|FPV NMSE↓|Mavic3Pro NMSE↓|FPV 복소SI-SDR↑|Mavic3Pro 복소SI-SDR↑|\n|---|---:|---:|---:|---:|\n'''
    for label,key in [('단일위상 + Hann','zero_hann'),('4위상 + Hann','C4_hann'),('8위상 + Hann','C8_hann'),('8위상 + 균일 병합','C8_uniform')]:
        m=result['stages'][key]['metrics'];a,b=m['nmse'];c,d=m['si_sdr']
        text+=f'|{label}|{a:.6f}|{b:.6f}|{c:.3f}dB|{d:.3f}dB|\n'
    text+='''\n## 관찰과 판단\n\n- 네 번째 창(index3)은 균등 합 보정 **이전의 단일위상 출력**에서부터\n  FPV의 지배 출력이2→1로 바뀐다. 창을 합치거나 C8을 적용해서 처음 생긴 현상이 아니다.\n  동일 전환이 보정 후·C8에서도 남는다. 모든 블록에서 전력·회귀조건수도 보존했다.\n- 단일위상보다 C8의 두 성분 NMSE가 모두 낮다. 현재 사례에서 C8을 원인으로 지목할 근거는 없다.\n- 균일 병합의 NMSE가 Hann보다 조금 낮지만 동일 출력 전환은 남는다.\n  한 사례의 작은 차이만으로 기존 전체 추론의 병합 방식을 교체하지 않는다.\n- 합 보정 전 원시 출력에는 공통 성분 자유도가 있다. 제거 전후의 수치만으로\n  보정 방식의 인과효과를 추정하거나 균등 보정을 폐기하지 않는다.\n\n따라서 첫 후속 비교는 연속 세 창의 공통 순열 학습이다. 이는 현재0.64ms PIT를\n1.916ms의 여러 창에 연결하는 비교이며, 파형 입력의 실제 문맥을 늘리는 구조 변경과는 구분한다.\n[동일 시작점·자료·예산의 등록 계획](SEQUENCE_ASSIGNMENT_PLAN_KO.md).\n\n이 결과는 이미 여러 번 확인한 한 B 개발 혼합의 진단이다. 전체 실패의 원인,\n독립 녹음 일반화 또는 기체 식별 개선을 입증한 결과가 아니다.\n파형 회귀의 제곱 계수는 성분 전력의 가산적 분해가 아니며, 비활동 구간과 수신 잡음도 구분해야 한다.\n\n입력 재현 v1은 Python complex/NumPy complex128 계수 승격 차이를 일치 검사가 잡아\n추론 전에 중지했다. v2는 원래 자료형을 보존해 수정했으며 허용 오차를 완화하지 않았다.\n'''
    (PUBLIC/'PACKET_ROUTING_E079_KO.md').write_text(text)
    print('PUBLIC_DIAGNOSIS_VERIFIED')


if __name__=='__main__':main()
