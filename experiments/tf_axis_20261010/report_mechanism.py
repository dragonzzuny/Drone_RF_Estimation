"""Saved-receipt audit for layer bypass and adapter transfer on fixed TRAIN6."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(public):
    roots=[ROOT/'local'/n for n in ('tf_axis_layer_diagnosis_20261010_v1','tf_axis_transfer_diagnosis_20261010_v1')]
    completed=[];protocols=[]
    for root,name in zip(roots,('TF_AXIS_LAYER_DIAGNOSIS.json','TF_AXIS_TRANSFER_DIAGNOSIS.json')):
        p=read(root/'PROTOCOL.json');r=read(root/'COMPLETE.json')
        assert r==read(public/name) and r['status']=='COMPLETE' and r['protocol_sha256']==digest(root/'PROTOCOL.json')
        assert r['updates']==0 and r['heldout_read'] is False
        for rel,sha in p['source_sha256'].items():assert digest(ROOT/rel)==digest(root/'source_snapshot'/rel)==sha
        assert digest(ROOT/'local/tf_axis_20261010_v1/ACTUAL_001.pt')==r['checkpoint_sha256']
        completed.append(r);protocols.append(p)
    first,transfer=completed
    assert protocols[0]['indices']==protocols[1]['indices'] and len(protocols[0]['indices'])==6
    assert transfer['predecessor_complete_sha256']==digest(roots[0]/'COMPLETE.json')
    assert transfer['rows'][:len(first['rows'])]==first['rows'] and len(first['rows'])==24 and len(transfer['rows'])==30
    assert transfer['parent_weights_preserved'] and transfer['adapter_weights_exact_e1']
    assert sum(r['elements'] for r in first['parameters'])==5263616
    for r in first['parameters']:
        assert 0<=r['changed_elements']<=r['elements'] and all(math.isfinite(r[k]) and r[k]>=0 for k in ('change_norm','initial_norm','trained_norm'))
    assert len(first['internal'])==6
    for r in first['internal']:
        assert r['input_shape']==[1,1024,32,31] and r['count_logits_bitwise_equal_when_bypassed']
        assert math.isclose((r['delta_rms']/r['input_rms'])**2,r['delta_relative_energy'],rel_tol=1e-5)
    models=('parent/e0','retained_unet/e1','tf_axis/e1','tf_axis/e1_bypassed','parent_with_e1_adapter')
    summaries=[]
    for model in models:
        part=[r for r in transfer['rows'] if r['model']==model]
        assert {r['index'] for r in part}==set(protocols[0]['indices']) and len(part)==6
        for r in part:
            assert len(r['nmse'])==len(r['si_sdr'])==len(r['reference_power'])==r['count']
            assert all(math.isfinite(v) and v>=0 for v in r['nmse']) and all(math.isfinite(v) for v in r['si_sdr'])
            assert 0<=r['sum_relative_error']<1e-10
        for count in (1,2,3):
            group=[r for r in part if r['count']==count];assert len(group)==2
            summaries.append(dict(model=model,count=count,cases=2,mean_nmse=statistics.mean(v for r in group for v in r['nmse']),
                mean_si_sdr=statistics.mean(v for r in group for v in r['si_sdr']),
                weakest_nmse=statistics.mean(r['nmse'][r['weakest_index']] for r in group)))
    ratios=[r['delta_rms']/r['input_rms'] for r in first['internal']]
    output=dict(status='PASS',source_hashes_checked=True,all_30_rows_reaggregated=True,
        auditor_sha256=digest(Path(__file__)),completed_sha256=[digest(root/'COMPLETE.json') for root in roots],
        adapter_rms_ratio_range=[min(ratios),max(ratios)],summaries=summaries,
        independently_reran_inference=False,independently_reloaded_all_tensors=False,heldout_read=False,recorded_iq_reads=0)
    (public/'TF_AXIS_MECHANISM_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
    labels={'parent/e0':'기존 부모','retained_unet/e1':'원 손실 대조e1','tf_axis/e1':'두 축 U-Net e1',
        'tf_axis/e1_bypassed':'동일e1 가중치·새 층 우회','parent_with_e1_adapter':'부모 본체·학습된e1 새 층'}
    lines=['# 두 축 U-Net: 새 층의 실제 영향','',
        '개발 점수를 보고 고른 사례가 아니라 기존TRAIN48의 개수별 첫 두 혼합을 사용했다. '
        'e1 체크포인트를 이용한 CPU 추론 진단이며 추가 학습은 없다.','',
        '| 구성 | 성분 수 | 평균 NMSE ↓ | 복소 SI-SDR ↑ dB | 최약 NMSE ↓ |',
        '|---|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"|{labels[r['model']]}|{r['count']}|{r['mean_nmse']:.6f}|{r['mean_si_sdr']:.3f}|{r['weakest_nmse']:.6f}|")
    lines+=['',f"새 층이 bottleneck에 더한 변화의RMS는 기존 입력의 {min(ratios)*100:.3f}–{max(ratios)*100:.3f}%였다. "
        '새 층 파라미터 대부분이 바뀌었지만 실제 출력 영향은 작았고, 우회 비교에서도 지표의 방향이 갈렸다. '
        '이는 e1의 제한된 영향에 관한 관찰이며, 구조가 원천적으로 무효하거나 학습률을 높이면 성공한다는 증거가 아니다.', '',
        '원래 본체에 학습된 새 층만 붙여도 두·세 성분의 공동 개선이 없었다. '
        '학습된 본체와 새 층의 상호 적응을 무시할 수 없다. 본체 고정 학습의 효과는 실제 학습 대조로 확인해야 한다.', '',
        '새 층을 우회해도 개수 logits는 같았다. 현재 개수 head는 긴 평균 전력 문맥만 읽고 새 국소 순환층의 출력을 직접 받지 않는다. '
        '따라서 이 구조의 파형 개선 여부와 자동 개수 추정의 성공을 분리해 보고한다.', '',
        '우회·이식은 동일 가중치의 추론 개입이며 독립 재학습 제거 대조가 아니다. '
        '모든 표는 단6개 학습 사례로 일반화 성과를 대신하지 않는다.', '',
        '[새 층 기록](TF_AXIS_LAYER_DIAGNOSIS.json) · [가중치 이식](TF_AXIS_TRANSFER_DIAGNOSIS.json) · '
        '[수치·출처 검산](TF_AXIS_MECHANISM_AUDIT.json) · [본체 고정 후속 계획](TF_AXIS_ADAPTATION_PLAN_KO.md)']
    (public/'TF_AXIS_MECHANISM_KO.md').write_text('\n'.join(lines)+'\n')
    return {k:v for k,v in output.items() if k!='summaries'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);a=p.parse_args();print(run(a.public))
