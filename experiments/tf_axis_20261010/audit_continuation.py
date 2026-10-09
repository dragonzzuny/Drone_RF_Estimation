"""Verify two resumed epochs and report every actual epoch, including failures."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
import continue_training as training

w=training.w;ROOT=training.ROOT


def audit(root,public):
    torch.set_num_threads(2);p=w.read(root/'PROTOCOL.json');training.verify(root,p)
    ph=w.digest(root/'PROTOCOL.json');result=w.read(root/'COMPLETE.json')
    assert result['status']=='COMPLETE' and result['protocol_sha256']==ph
    assert result['additional_updates']==150 and result['total_updates']==225
    predecessor=Path(p['predecessor']);old=p['original_protocol']
    resume=w.read(root/'RESUME_CHECK.json')
    assert resume['status']=='PASS' and resume['source_checkpoint_sha256']==p['start_checkpoint_sha256']
    parent,ids=w.validate(Path(old['baseline']),None)
    initial,_=w.validate(predecessor/'VALIDATION_000.json',ids)
    first,_=w.validate(predecessor/'VALIDATION_001.json',ids)
    best=dict(epoch=0,metric=initial['selection_nmse'])
    if first['selection_nmse']<best['metric']:best=dict(epoch=1,metric=first['selection_nmse'])
    events=[]
    for epoch in (2,3):
        actual,_=w.validate(root/f'VALIDATION_{epoch:03d}.json',ids)
        control,_=w.validate(Path(old['study'])/f'retained_unet/VALIDATION_{epoch:03d}.json',ids)
        event=w.read(root/f'EPOCH_{epoch:03d}.json')
        assert event['validation']==actual and event['protocol_sha256']==ph and event['updates']==epoch*75
        comparison=training.base.compare(actual,parent,control)
        for row in comparison:
            if row['reference']=='retained_control_e1':row['reference']=f'retained_control_e{epoch}'
        assert comparison==event['comparison']
        criterion=all(r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0 for r in comparison if r['count'] in (2,3))
        assert criterion==event['criterion_met']
        if actual['selection_nmse']<best['metric']:best=dict(epoch=epoch,metric=actual['selection_nmse'])
        assert event['best']==best
        receipts=root/f'GRADIENT_UPDATES_{epoch:03d}.json'
        assert event['gradient_receipts_sha256']==w.digest(receipts)
        records=w.read(receipts);assert records['protocol_sha256']==ph and len(records['updates'])==75
        for i,row in enumerate(records['updates'],1):
            assert row['update']==(epoch-1)*75+i and row['examples']==i*32
            gram=np.asarray(row['gram']);one=np.ones(3)
            assert np.allclose(gram,gram.T) and np.linalg.eigvalsh(gram).min()>-1e-5*max(float(np.trace(gram)),1.)
            norm=math.sqrt(max(float(one@gram@one),0.))
            assert math.isclose(norm,row['ordinary_norm'],rel_tol=2e-4,abs_tol=2e-5)
            assert row['ordinary_norm']==row['projected_norm'] and row['change_norm']==0
            assert math.isclose(row['preclip_norm'],row['ordinary_norm'],rel_tol=2e-4,abs_tol=2e-5)
            assert np.allclose(one@gram,row['projected_direction_dot_original_tasks'],rtol=2e-4,atol=2e-5)
        current=torch.load(root/f'ACTUAL_{epoch:03d}.pt',map_location='cpu',weights_only=False)
        assert current['epoch']==epoch and current['updates']==epoch*75 and current['protocol_sha256']==ph
        assert all(torch.isfinite(v).all() for v in current['model'].values())
        del current;events.append(event)
    assert events==result['events'] and best==result['best']
    saved=torch.load(root/'LAST.pt',map_location='cpu',weights_only=False)
    actual=torch.load(root/'ACTUAL_003.pt',map_location='cpu',weights_only=False)
    chosen=torch.load(root/'BEST.pt',map_location='cpu',weights_only=False)
    assert saved['protocol_sha256']==chosen['protocol_sha256']==ph
    assert saved['epoch']==3 and saved['updates']==225 and saved['best']==chosen['best']==best
    expected=(torch.load(predecessor/'BEST.pt',map_location='cpu',weights_only=False)['model'] if best['epoch']<2
        else torch.load(root/f"ACTUAL_{best['epoch']:03d}.pt",map_location='cpu',weights_only=False)['model'])
    assert actual['model'].keys()==saved['model'].keys()==chosen['model'].keys()==expected.keys()
    for key,value in actual['model'].items():
        assert torch.equal(value,saved['model'][key]) and torch.equal(chosen['model'][key],expected[key])
    net=training.first.make_model('retained_unet');net.load_state_dict(actual['model'])
    params=list(net.parameters());assert sum(q.numel() for q in params)==37406475
    group,=saved['optimizer']['param_groups']
    assert group['lr']==p['lr'] and group['weight_decay']==p['weight_decay']
    states=saved['optimizer']['state'];assert len(group['params'])==len(params)==len(states)
    assert {int(v['step']) for v in states.values()}=={225}
    for param,key in zip(params,group['params']):
        for name in ('exp_avg','exp_avg_sq'):
            value=states[key][name];assert value.shape==param.shape and torch.isfinite(value).all()
    output=dict(status='PASS',protocol_sha256=ph,complete_sha256=w.digest(root/'COMPLETE.json'),
        auditor_sha256=w.digest(Path(__file__)),all_epoch_630_rows_reaggregated=True,
        continuation_model_optimizer_rng_receipt_checked=True,optimizer_225_steps_checked=True,
        selected_epoch=best['epoch'],selected_and_actual_tensors_checked=True,
        independently_recomputed_gradients=False,independently_reran_inference=False,
        heldout_read=False,gpu_use=False,recorded_iq_reads=0,
        hashes={name:w.digest(root/name) for name in ('ACTUAL_002.pt','ACTUAL_003.pt','LAST.pt','BEST.pt')})
    w.write(public/'TF_AXIS_CONTINUATION_AUDIT.json',output)
    lines=['# 시간·주파수 두 축 U-Net: 총 3epoch 결과','',
        '같은 부모·RFUAV native 혼합·seed0·원 손실·각225업데이트의 비교다. '
        '후보는37,406,475개, 대조는32,142,859개 파라미터다. e1의 실제 모델과 AdamW·RNG를 이어 e2/e3를 학습했다. '
        '각 시점의 실제 결과이며 선택e0로 실패를 숨기지 않는다.','',
        '| 모델 | 추가 epoch | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB | 최약 NMSE 2/3 ↓ |',
        '|---|---:|---|---|---|']
    for epoch in (0,1,2,3):
        variants=[('부모',initial)] if epoch==0 else [
            ('기존 U-Net',w.validate(Path(old['study'])/f'retained_unet/VALIDATION_{epoch:03d}.json',ids)[0]),
            ('두 축 U-Net',first if epoch==1 else events[epoch-2]['validation'])]
        for name,record in variants:
            a,b=record['by_count'][1:]
            lines.append(f"|{name}|{epoch}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|"
                f"{a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|")
    lines+=['',f"후보의 선택은 e{best['epoch']}다. e2/e3의 공동 개선 기준: "+str([e['criterion_met'] for e in events])+'.',
        '', '5개 개발 기록 묶음의630창을 반복 사용한 한 seed의 개발 비교다. 독립 시험·충분한 수렴·같은 비용 우월성의 증거가 아니다. '
        '학습 시 FP32 기울기 덧셈 순서도 기존 대조와 다르다. Autel/예약 확인 파일은 미개봉이다.', '',
        '[사전 계획](TF_AXIS_CONTINUATION_PLAN_KO.md) · [실제 epoch 수치](TF_AXIS_CONTINUATION_RESULT.json) · '
        '[검산](TF_AXIS_CONTINUATION_AUDIT.json)']
    w.write(public/'TF_AXIS_CONTINUATION_KO.md','\n'.join(lines)+'\n');return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True);args=parser.parse_args()
    print(audit(args.run.resolve(),args.public.resolve()),flush=True)
