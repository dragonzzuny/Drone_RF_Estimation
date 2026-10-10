"""Additional independent checks requested by the installed Codex plugin."""
import importlib.util
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fresh_review_training',HERE/'train.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


def main():
    torch.set_num_threads(2);root=t.ROOT/'local/fresh_schedule_20261010_v1';public=t.ROOT/'reports/2026-10-10'
    p=t.w.read(root/'PROTOCOL.json');t.verify(root,p);old=p['original_protocol']
    assert t.w.read(public/'FRESH_SCHEDULE_AUDIT.json')['status']=='PASS'
    assert t.w.digest(root/'VALIDATION_000.json')==old['baseline_sha256']
    assert t.w.digest(Path(p['historical_control']))==old['control_validation_sha256']
    parent,ids=t.w.validate(Path(old['baseline']),None)
    actual,_=t.w.validate(root/'VALIDATION_001.json',ids)
    control,_=t.w.validate(Path(p['historical_control']),ids)
    event=t.w.read(root/'EPOCH_001.json')
    expected=min(parent['selection_nmse'],actual['selection_nmse'])
    assert event['best']['metric']==expected
    for name in ('BEST.pt','LAST.pt'):
        saved=torch.load(root/name,map_location='cpu',weights_only=False)
        assert saved['best']['metric']==expected and saved['best']==event['best']
        del saved
    base=t.w.read(public/'CODEX_PLUGIN_REVIEW_FOLLOWUP.json')
    check=t.w.read(public/'FRESH_CHECKPOINT_CHECK.json')
    assert check['status']=='PASS' and check['training_protocol_sha256']==t.w.digest(root/'PROTOCOL.json')
    assert all(g['mean_si_sdr'] is not None for g in actual['by_count'])
    rows=t.core.base.compare(actual,parent,control)
    # This sensitivity display does not rewrite the frozen strict acceptance rule.
    margin=all(r['nmse_delta'] < -1e-5 and r['si_sdr_delta'] > 1e-4 and r['weakest_nmse_delta']<=0
               for r in rows if r['count'] in (2,3))
    result=dict(base,status='FINAL_FOLLOWUP_PASS',best_metric_checked=expected,
        e0_sha256=t.w.digest(root/'VALIDATION_000.json'),checker_sha256=t.w.digest(Path(__file__)),
        no_undefined_si_sdr_in_this_result=True,numerical_margin_display=dict(nmse=1e-5,si_sdr_db=1e-4,passed=margin),
        remaining='Frozen trainer nonfinite-SI failure branch remains a future hardening item; not reached in this run. '
        'All count scores and recording-group errors are in supplementary reports; criterion unchanged.')
    t.w.write(public/'CODEX_PLUGIN_REVIEW_FOLLOWUP.json',result)
    lines=['# 플러그인 검토에 따른 추가 검산','',
        '실행 중인 소스와 규약은 변경하지 않았다. e0 파일의 부모 지문, 과거 대조의 이전 지문, 선택된 best.metric, '
        '실제·선택 가중치의 기존 감사 결과를 대조했다. CPU의 원규모 두 사례에서 재계산 전후 출력·gradient가 bitwise 일치했다. '
        '이는 모든 GPU 업데이트의 완전 동일성이나 동일 계산량을 뜻하지 않는다.','',
        '| 조건 | 성분 수 | 평균 NMSE ↓ | 복소 SI-SDR ↑ dB | 최약 NMSE ↓ | 개수 정확도 |',
        '|---|---:|---:|---:|---:|---:|']
    for label,value in [('부모',parent),('재사용 혼합 대조',control),('새 혼합 일정',actual)]:
        for g in value['by_count']:
            lines.append(f"|{label}|{g['count']}|{g['mean_nmse']:.6f}|{g['mean_si_sdr']:.3f}|{g['weakest_nmse']:.6f}|{100*g['construction_count_accuracy']:.2f}%|")
    lines+=['','개수 정확도는 합성에 사용한 원기록 수의 정답에 대한 값이다. 실제 물리 드론 대수의 정확도가 아니다. '
        '비유한 SI-SDR 예외 경로는 이번 결과에서 발생하지 않았으며 차기 worker의 보강 항목으로 남긴다.',
        '', '[모든 기록·전력순위·기종별 결과](FRESH_SCHEDULE_HISTORY.json) · '
        '[자료 일정 빈도와 지문](FRESH_SCHEDULE_DATA_AUDIT.json) · '
        '[추가 검사](CODEX_PLUGIN_REVIEW_FOLLOWUP.json)']
    (public/'CODEX_PLUGIN_REVIEW_FOLLOWUP_KO.md').write_text('\n'.join(lines)+'\n')
    print(result,flush=True)


if __name__=='__main__':main()
