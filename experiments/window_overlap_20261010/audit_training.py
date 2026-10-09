"""Audit paired-window full-model epochs, including failed/selected-zero runs."""
import argparse
from pathlib import Path
import torch
import train as training

w=training.w


def run(root,public):
    torch.set_num_threads(2);p=w.read(root/'PROTOCOL.json');training.verify(root,p)
    ph=w.digest(root/'PROTOCOL.json');complete=w.read(root/'COMPLETE.json')
    assert complete['protocol_sha256']==ph and complete['status']=='COMPLETE'
    old=p['original_protocol'];parent,ids=w.validate(Path(old['baseline']),None)
    historical,_=w.validate(Path(old['study'])/'retained_unet/VALIDATION_001.json',ids)
    parent_state=torch.load(old['parent_checkpoint'],map_location='cpu',weights_only=False)['model']
    summaries=[];hashes={}
    for number,arm in enumerate(training.ARMS):
        folder=root/arm;event=w.read(folder/'EPOCH_001.json')
        initial,_=w.validate(folder/'VALIDATION_000.json',ids)
        actual,_=w.validate(folder/'VALIDATION_001.json',ids)
        assert event==complete['events'][number] and event['validation']==actual
        assert event['epoch']==1 and event['updates']==75 and event['count_pairs']==[800,800,800]
        assert sum(event['offset_counts'].values())==2400 and len(event['preclip_gradient_norms'])==75
        assert event['exact_recomputed_second_forward_all_pairs']
        for a,b in zip(parent['by_count'],initial['by_count']):
            for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert abs(a[key]-b[key])<2e-6
        comparator=historical if number==0 else summaries[0]
        comparisons=training.base.compare(actual,parent,comparator)
        for q in comparisons:
            if q['reference']=='retained_control_e1':q['reference']='historical_single_view_e1' if number==0 else 'matched_paired_supervision'
        assert event['comparison']==comparisons
        criterion_rows=[q for q in comparisons if number==1 or q['reference']=='parent']
        assert event['criterion_met']==training.passed(criterion_rows)
        best_epoch=1 if actual['selection_nmse']<initial['selection_nmse'] else 0
        assert event['best']['epoch']==best_epoch
        checkpoint=torch.load(folder/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
        last=torch.load(folder/'LAST.pt',map_location='cpu',weights_only=False)
        best=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False)
        assert checkpoint['protocol_sha256']==last['protocol_sha256']==best['protocol_sha256']==ph
        assert checkpoint['updates']==last['updates']==75 and checkpoint['epoch']==last['epoch']==1
        assert best['best']==last['best']==event['best']
        selected=parent_state if best_epoch==0 else checkpoint['model']
        for key,value in checkpoint['model'].items():
            assert torch.isfinite(value).all() and torch.equal(value,last['model'][key])
            assert torch.equal(best['model'][key],selected[key])
        net=training.worker.make_model('retained_unet');net.load_state_dict(checkpoint['model'])
        params=list(net.parameters());assert sum(q.numel() for q in params)==32142859
        group,=last['optimizer']['param_groups'];states=last['optimizer']['state']
        assert group['lr']==1e-5 and group['weight_decay']==1e-4
        assert len(group['params'])==len(params)==len(states) and {int(q['step']) for q in states.values()}=={75}
        for param,key in zip(params,group['params']):
            for name in ('exp_avg','exp_avg_sq'):
                value=states[key][name];assert value.shape==param.shape and torch.isfinite(value).all()
        for name in ('VALIDATION_000.json','VALIDATION_001.json','ACTUAL_001.pt','LAST.pt','BEST.pt','EPOCH_001.json'):
            hashes[arm+'/'+name]=w.digest(folder/name)
        summaries.append(actual)
        del net,params,checkpoint,last,best,selected
    report=dict(status='PASS',protocol_sha256=ph,complete_sha256=w.digest(root/'COMPLETE.json'),
        auditor_sha256=w.digest(Path(__file__)),all_validation_rows_reaggregated=True,
        all_actual_last_selected_tensors_checked=True,optimizer_moments_and_75_steps_checked=True,
        source_hashes_checked=True,hashes=hashes,heldout_read=False,independent_test=False)
    w.write(public/'PAIRED_WINDOW_AUDIT.json',report)
    lines=['# 두 창 지도학습과 추가 일관성: 같은 예산 비교','',
        'RFUAV 같은 원 RF 대역·native100MS/s·원기록 묶음 분할·seed0·기존 부모 전체32.14M을 유지했다. '
        '각 군은 같은2400혼합의4800창,75업데이트이며 각 쌍당3forward/2backward다. '
        '새 기록4800개나 기존 단일 창 대조와 같은 계산량을 뜻하지 않는다.','',
        '| 군 | 두 성분 NMSE ↓ | 두 성분 복소 SI-SDR ↑ dB | 세 성분 NMSE ↓ | 세 성분 복소 SI-SDR ↑ dB | 선택 epoch |',
        '|---|---:|---:|---:|---:|---:|']
    for name,v,best_epoch in [('parent',parent,0)]+[(a,s,complete['events'][i]['best']['epoch']) for i,(a,s) in enumerate(zip(training.ARMS,summaries))]:
        a,b=[next(q for q in v['by_count'] if q['count']==n) for n in (2,3)]
        lines.append(f"| {name} | {a['mean_nmse']:.6f} | {a['mean_si_sdr']:.3f} | {b['mean_nmse']:.6f} | {b['mean_si_sdr']:.3f} | {best_epoch} |")
    lines+=['','수치는 실제 e1이며, 선택이e0여도 학습 결과를 숨기지 않는다.']
    for event in complete['events']:
        lines.append(f"- {event['arm']}의 사전 공동 개선 기준: {'충족' if event['criterion_met'] else '미충족'}")
    lines+=['','추가 일관성 항은 새 정답 정보가 아니라 창 의존성에 대한 가중치 변경이다. '
        '원래 개발630혼합 전체를 평가했고 Autel·예약 확인6파일은 읽지 않았다. '
        '반복 사용한 DEV와 한seed의 결과여서 독립 일반화 확인을 대신하지 않는다.',
        '', '[계획](PAIRED_WINDOW_PLAN_KO.md) · [수식·선행](PAIRED_WINDOW_METHOD_KO.md) · '
        '[모든 수치](PAIRED_WINDOW_RESULT.json) · [최종 검산](PAIRED_WINDOW_AUDIT.json)']
    (public/'PAIRED_WINDOW_KO.md').write_text('\n'.join(lines)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True);a=parser.parse_args()
    run(a.run.resolve(),a.public.resolve())
