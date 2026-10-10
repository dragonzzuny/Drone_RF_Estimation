"""Verify both full recursive trial states and every saved DEV row."""
import argparse
from pathlib import Path
import torch
import train as training
w=training.w

def run(root,public):
    torch.set_num_threads(2);p=w.read(root/'PROTOCOL.json');training.verify(root,p)
    ph=w.digest(root/'PROTOCOL.json');complete=w.read(root/'COMPLETE.json')
    assert complete['protocol_sha256']==ph and complete['total_updates']==150 and complete['status']=='COMPLETE'
    old=p['original_protocol'];parent,ids=w.validate(Path(old['baseline']),None)
    initial_model=training.worker.make_model('retained_unet',Path(old['parent_checkpoint']))
    initial_state=initial_model.state_dict();summaries=[];hashes={}
    for number,arm in enumerate(training.ARMS):
        folder=root/arm;event=w.read(folder/'EPOCH_001.json')
        initial,_=w.validate(folder/'VALIDATION_000.json',ids);actual,_=w.validate(folder/'VALIDATION_001.json',ids)
        assert complete['events'][number]==event and event['initial']==initial and event['validation']==actual
        assert event['epoch']==1 and event['updates']==75 and event['count_examples']==[800,800,800]
        assert len(event['preclip_gradient_norms'])==75
        if number==0:
            for a,b in zip(parent['by_count'],initial['by_count']):
                for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert abs(a[key]-b[key])<2e-6
        comparisons=training.core.base.compare(actual,parent,parent if number==0 else summaries[0])
        if number==0:comparisons=[r for r in comparisons if r['reference']=='parent']
        else:
            for r in comparisons:
                if r['reference']=='retained_control_e1':r['reference']='matched_two_pass_control'
        assert event['comparison']==comparisons and event['criterion_met']==training.passed(comparisons)
        best_epoch=1 if actual['selection_nmse']<initial['selection_nmse'] else 0
        assert event['best']['epoch']==best_epoch
        actual_state=torch.load(folder/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
        last=torch.load(folder/'LAST.pt',map_location='cpu',weights_only=False)
        best=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False)
        assert actual_state['protocol_sha256']==last['protocol_sha256']==best['protocol_sha256']==ph
        assert actual_state['updates']==last['updates']==75 and actual_state['epoch']==last['epoch']==1
        assert best['best']==last['best']==event['best']
        selected=initial_state if best_epoch==0 else actual_state['model']
        assert set(actual_state['model'])==set(initial_state)==set(best['model'])==set(last['model'])
        for key,value in actual_state['model'].items():
            assert torch.isfinite(value).all() and torch.equal(value,last['model'][key])
            assert torch.equal(best['model'][key],selected[key])
        params=list(initial_model.parameters());group,=last['optimizer']['param_groups'];states=last['optimizer']['state']
        assert group['lr']==1e-5 and group['weight_decay']==1e-4
        assert len(params)==len(group['params'])==len(states) and {int(s['step']) for s in states.values()}=={75}
        for param,key in zip(params,group['params']):
            for name in ('exp_avg','exp_avg_sq'):
                assert states[key][name].shape==param.shape and torch.isfinite(states[key][name]).all()
        for name in ('VALIDATION_000.json','VALIDATION_001.json','ACTUAL_001.pt','LAST.pt','BEST.pt','EPOCH_001.json'):
            hashes[arm+'/'+name]=w.digest(folder/name)
        summaries.append(actual);del actual_state,last,best,selected,states
    report=dict(status='PASS',protocol_sha256=ph,complete_sha256=w.digest(root/'COMPLETE.json'),
        auditor_sha256=w.digest(Path(__file__)),all_validation_rows_reaggregated=True,
        all_actual_last_selected_tensors_checked=True,optimizer_moments_and_75_steps_checked=True,
        source_hashes_checked=True,hashes=hashes,heldout_read=False,independent_test=False,
        recorded_iq_reads=0,independently_reran_inference=False)
    w.write(public/'SUCCESSIVE_AUDIT.json',report)
    lines=['# 공유 U-Net 순차 추출: 두 군 학습 비교','',
        '같은 RFUAV native 자료·기록 묶음 분할·부모 가중치·seed0·각2400혼합/75업데이트다. '
        '두 군 모두32,142,859파라미터와 예제당 두 번의 원 규모 U-Net 처리를 사용한다. '
        '대조는 같은 입력에 반씩 손실을 주고, 후보는 두 단계를 연결해 최종3출력 PIT로 함께 학습한다. '
        '후보 추론은 두 번 실행하고 정답·실제 개수·기종을 받지 않는다.','',
        '| 조건 | NMSE2/3 ↓ | 복소SI-SDR2/3 ↑ dB | 최약NMSE2/3 ↓ | 선택epoch |',
        '|---|---|---|---|---:|']
    values=[('parent',parent,0)]
    for event in complete['events']:
        values.append((event['arm']+' e0',event['initial'],0))
        values.append((event['arm']+' e1',event['validation'],event['best']['epoch']))
    for name,value,best_epoch in values:
        a,b=[next(q for q in value['by_count'] if q['count']==n) for n in (2,3)]
        lines.append(f"|{name}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{best_epoch}|")
    lines+=['','초기 가중치는 같지만 초기 함수·출력은 다르다. 두 번째 단계는 원래 혼합의 긴 문맥을 계속 사용한다. '
        '후보의 마지막 파형에 잔차가 집중될 수 있어 약한 성분과비활성 출력을 함께 확인한다. '
        '다섯 DEV 기록 묶음의630창을 반복 사용한 한seed 실험이며 독립 확인 결과가 아니다.']
    for event in complete['events']:lines.append(f"- {event['arm']} 공동 채택 기준: {event['criterion_met']}")
    lines+=['','[전체 수치](SUCCESSIVE_RESULT.json) · [검산](SUCCESSIVE_AUDIT.json) · [규약](SUCCESSIVE_PLAN_KO.md)']
    (public/'SUCCESSIVE_KO.md').write_text('\n'.join(lines)+'\n');print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.public.resolve())
