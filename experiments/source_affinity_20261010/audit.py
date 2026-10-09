"""Reaggregate every DEV row and verify full states/optimizer receipts."""
import argparse
from pathlib import Path
import torch
import train as training
w=training.w


def run(root,public):
    torch.set_num_threads(2);p=w.read(root/'PROTOCOL.json');training.verify(root,p)
    ph=w.digest(root/'PROTOCOL.json');complete=w.read(root/'COMPLETE.json')
    assert complete['protocol_sha256']==ph and complete['status']=='COMPLETE' and complete['total_updates']==225
    old=p['original_protocol'];parent,ids=w.validate(Path(old['baseline']),None)
    historical,_=w.validate(Path(old['study'])/'retained_unet/VALIDATION_001.json',ids)
    parent_net,parent_capture=training.make_model(Path(old['parent_checkpoint']))
    parent_state=parent_net.state_dict();summaries=[];hashes={}
    for number,arm in enumerate(training.ARMS):
        folder=root/arm;event=w.read(folder/'EPOCH_001.json')
        initial,_=w.validate(folder/'VALIDATION_000.json',ids)
        actual,_=w.validate(folder/'VALIDATION_001.json',ids)
        assert event==complete['events'][number] and event['validation']==actual
        assert event['epoch']==1 and event['updates']==75 and event['count_examples']==[800,800,800]
        assert len(event['preclip_gradient_norms'])==75
        for a,b in zip(parent['by_count'],initial['by_count']):
            for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert abs(a[key]-b[key])<2e-6
        comparator=historical if number==0 else summaries[0]
        comparisons=training.base.compare(actual,parent,comparator)
        for q in comparisons:
            if q['reference']=='retained_control_e1':q['reference']='historical_e1' if number==0 else 'matched_affinity_control'
        assert event['comparison']==comparisons
        assert event['criterion_met']==training.passed([q for q in comparisons if number>0 or q['reference']=='parent'])
        if number==2:
            rows=training.base.compare(actual,parent,summaries[1])[3:]
            for q in rows:q['reference']='matched_hard_affinity'
            assert rows==event['hard_comparison'] and event['soft_beats_hard']==training.passed(rows)
        best_epoch=1 if actual['selection_nmse']<initial['selection_nmse'] else 0
        assert event['best']['epoch']==best_epoch
        checkpoint=torch.load(folder/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
        last=torch.load(folder/'LAST.pt',map_location='cpu',weights_only=False)
        best=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False)
        assert checkpoint['protocol_sha256']==last['protocol_sha256']==best['protocol_sha256']==ph
        assert checkpoint['updates']==last['updates']==75 and checkpoint['epoch']==last['epoch']==1
        assert best['best']==last['best']==event['best']
        selected=parent_state if best_epoch==0 else checkpoint['model']
        assert set(checkpoint['model'])==set(selected)==set(best['model'])==set(last['model'])
        for key,value in checkpoint['model'].items():
            assert torch.isfinite(value).all() and torch.equal(value,last['model'][key])
            assert torch.equal(best['model'][key],selected[key])
        net,capture=training.make_model();net.load_state_dict(checkpoint['model'])
        params=list(net.parameters());assert sum(q.numel() for q in params)==32143899
        group,=last['optimizer']['param_groups'];states=last['optimizer']['state']
        assert group['lr']==1e-5 and group['weight_decay']==1e-4
        assert len(group['params'])==len(params)==len(states) and {int(q['step']) for q in states.values()}=={75}
        for param,key in zip(params,group['params']):
            for name in ('exp_avg','exp_avg_sq'):
                value=states[key][name];assert value.shape==param.shape and torch.isfinite(value).all()
        for name in ('VALIDATION_000.json','VALIDATION_001.json','ACTUAL_001.pt','LAST.pt','BEST.pt','EPOCH_001.json'):
            hashes[arm+'/'+name]=w.digest(folder/name)
        summaries.append(actual);capture.close()
        del net,capture,params,checkpoint,last,best,selected
    parent_capture.close()
    report=dict(status='PASS',protocol_sha256=ph,complete_sha256=w.digest(root/'COMPLETE.json'),
        auditor_sha256=w.digest(Path(__file__)),all_validation_rows_reaggregated=True,
        all_actual_last_selected_tensors_checked=True,optimizer_moments_and_75_steps_checked=True,
        source_hashes_checked=True,hashes=hashes,heldout_read=False,independent_test=False,
        recorded_iq_reads=0,independently_reran_inference=False)
    w.write(public/'SOURCE_AFFINITY_AUDIT.json',report)
    lines=['# 성분 친화도 보조 학습: 세 군 완료 비교','',
        'RFUAV 같은 원RF 대역·수신 중심 차이·기록 묶음 분할·native100MS/s·seed0다. '
        '같은 부모 원규모32.14M에 보조1,040개를 추가했고 각2400혼합/75업데이트다. '
        '두·세 성분에만 추가0.1 친화도 손실을 사용한다. 추론에는 보조 층·정답·정답 개수를 쓰지 않는다.','',
        '| 군 | NMSE2/3 ↓ | 복소SI-SDR2/3 ↑ dB | 최약NMSE2/3 ↓ | 선택epoch |',
        '|---|---|---|---|---|']
    for name,v,best_epoch in [('parent',parent,0)]+[(a,s,complete['events'][i]['best']['epoch']) for i,(a,s) in enumerate(zip(training.ARMS,summaries))]:
        a,b=[next(q for q in v['by_count'] if q['count']==n) for n in (2,3)]
        lines.append(f"|{name}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{best_epoch}|")
    lines+=['','각 군 실제e1을 보고했다. 선택e0도 학습한e1의 실패를 숨기지 않는다.']
    for event in complete['events']:
        lines.append(f"- {event['arm']}의 사전 채택 기준: {'충족' if event['criterion_met'] else '미충족'}")
    lines += [f"- soft가hard보다 공동 개선됐는지: {complete['events'][2]['soft_beats_hard']}",
        '', '동일630개 DEV를 반복 사용한 한seed 개발 실험이다. 독립 확인·범용 드론 분리·새 군집 이론의 입증으로 해석하지 않는다. '
        '부모·0가중치 대조 대비 개선과 hard/soft 라벨 차이의 효과를 구분한다. Autel·예약 확인6파일은 읽지 않았다.',
        '', '[사전 계획](SOURCE_AFFINITY_PLAN_KO.md) · [수식·선행](SOURCE_AFFINITY_METHOD_KO.md) · '
        '[전체 수치](SOURCE_AFFINITY_RESULT.json) · [검산](SOURCE_AFFINITY_AUDIT.json)']
    (public/'SOURCE_AFFINITY_KO.md').write_text('\n'.join(lines)+'\n');print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True);a=parser.parse_args();run(a.run.resolve(),a.public.resolve())
