"""Audit matched adaptation arms, including exact frozen-parent preservation."""
import argparse
import math
from pathlib import Path
import torch
import train_adaptation as training

w=training.w


def audit(root,public):
    torch.set_num_threads(2);p=w.read(root/'PROTOCOL.json');training.verify(root,p)
    result=w.read(root/'COMPLETE.json');ph=w.digest(root/'PROTOCOL.json')
    assert result['status']=='COMPLETE' and result['protocol_sha256']==ph and result['total_updates']==150
    old=p['original_protocol'];parent,ids=w.validate(Path(old['baseline']),None)
    control,_=w.validate(Path(old['study'])/'retained_unet/VALIDATION_001.json',ids)
    expected_parent=training.first.make_model('retained_unet',Path(old['parent_checkpoint'])).state_dict()
    events=[];hashes={}
    for arm in training.ARMS:
        folder=root/arm;initial,_=w.validate(folder/'VALIDATION_000.json',ids);actual,_=w.validate(folder/'VALIDATION_001.json',ids)
        for a,b in zip(initial['by_count'],parent['by_count']):
            for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert abs(a[key]-b[key])<2e-6
        event=w.read(folder/'EPOCH_001.json')
        assert event['arm']==arm and event['epoch']==1 and event['updates']==75 and event['protocol_sha256']==ph
        assert event['validation']==actual and event['count_examples']==[800,800,800]
        assert len(event['preclip_gradient_norms'])==75 and all(math.isfinite(v) and v>=0 for v in event['preclip_gradient_norms'])
        comparison=training.base.compare(actual,parent,control)
        assert event['comparison']==comparison and event['criterion_met']==training.passed(comparison)
        selected=initial if initial['selection_nmse']<=actual['selection_nmse'] else actual
        best=dict(epoch=selected['epoch'],metric=selected['selection_nmse']);assert event['best']==best
        current=torch.load(folder/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
        last=torch.load(folder/'LAST.pt',map_location='cpu',weights_only=False)
        chosen=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False)
        assert current['protocol_sha256']==last['protocol_sha256']==chosen['protocol_sha256']==ph
        assert current['arm']==last['arm']==chosen['arm']==arm
        assert current['epoch']==last['epoch']==1 and current['updates']==last['updates']==75
        assert last['best']==chosen['best']==best
        expected=expected_parent if best['epoch']==0 else current['model']
        assert current['model'].keys()==last['model'].keys()==chosen['model'].keys()==expected.keys()
        for key,value in current['model'].items():
            assert torch.isfinite(value).all() and torch.equal(value,last['model'][key])
            assert torch.equal(chosen['model'][key],expected[key])
            if arm=='frozen_backbone' and not key.startswith('tf_axes.'):assert torch.equal(value,expected_parent[key])
        if arm=='frozen_backbone':
            before=w.read(folder/'VALIDATION_000.json')['rows'];after=w.read(folder/'VALIDATION_001.json')['rows']
            assert all(a['predicted_count']==b['predicted_count'] for a,b in zip(before,after))
        net=training.first.make_model('retained_unet');net.load_state_dict(current['model'])
        opt,params=training.configure(net,arm)
        assert sum(q.numel() for q in net.parameters())==p['full_parameters']
        assert sum(q.numel() for q in params)==p['trainable_parameters'][arm]
        stored=last['optimizer'];assert len(stored['param_groups'])==len(opt.param_groups)
        assert len(stored['state'])==len(params) and {int(v['step']) for v in stored['state'].values()}=={75}
        for actual_group,expected_group in zip(stored['param_groups'],opt.param_groups):
            for key in ('name','lr','weight_decay'):assert actual_group[key]==expected_group[key]
            assert len(actual_group['params'])==len(expected_group['params'])
            for key,param in zip(actual_group['params'],expected_group['params']):
                for moment in ('exp_avg','exp_avg_sq'):
                    value=stored['state'][key][moment];assert value.shape==param.shape and torch.isfinite(value).all()
        hashes[arm]={name:w.digest(folder/name) for name in ('ACTUAL_001.pt','LAST.pt','BEST.pt','VALIDATION_001.json')}
        events.append(event);del current,last,chosen,net,opt,params,stored,expected
    assert events==result['events']
    output=dict(status='PASS',protocol_sha256=ph,complete_sha256=w.digest(root/'COMPLETE.json'),
        auditor_sha256=w.digest(Path(__file__)),all_630_rows_each_arm_reaggregated=True,
        all_actual_selected_tensors_checked=True,optimizer_groups_learning_rates_and_steps_checked=True,
        frozen_parent_tensors_exact=True,frozen_predicted_counts_all_630_unchanged=True,
        source_hashes_checked=True,hashes=hashes,gpu_use=False,heldout_read=False,recorded_iq_reads=0,
        independently_reran_inference=False)
    w.write(public/'TF_AXIS_ADAPTATION_AUDIT.json',output)
    rows=[('부모',parent),('원 손실 대조e1',control)]+[(e['arm'],e['validation']) for e in events]
    lines=['# 새 층 학습률·본체 고정: 두 군 비교','',
        'RFUAV 같은 native 자료·초기 부모·새 층 seed0·각1epoch/75업데이트·원 손실의 비교다. '
        '새 층lr1e-4는 양쪽에서 같고, joint의 본체lr은1e-5다. '
        '추론 구조는 둘 다37,406,475개 파라미터이며 고정군은5,263,616개만 학습한다.','',
        '| 군 | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB | 최약 NMSE 2/3 ↓ |',
        '|---|---|---|---|']
    for name,row in rows:
        a,b=row['by_count'][1:]
        lines.append(f"|{name}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|"
            f"{a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|")
    for e in events:lines+=['',f"{e['arm']}: 실제e1을 위 표에 보고했다. 선택e{e['best']['epoch']}, 부모/원 대조에 대한 공동 개선 기준 {e['criterion_met']}."]
    joint,frozen=[e['validation'] for e in events]
    direct=[]
    for a,b in zip(joint['by_count'][1:],frozen['by_count'][1:]):
        direct.append(dict(count=a['count'],nmse_delta=b['mean_nmse']-a['mean_nmse'],
            si_sdr_delta=b['mean_si_sdr']-a['mean_si_sdr'],weakest_nmse_delta=b['weakest_nmse']-a['weakest_nmse']))
    w.write(public/'TF_AXIS_ADAPTATION_PAIRED.json',dict(frozen_minus_joint=direct,
        all_three_metrics_each_count_improve=training.passed(direct),heldout_read=False))
    lines+=['','고정군의 기존 본체·문맥·개수 head 전체가 부모와 정확히 같고630개 개수 예측도 그대로임을 검산했다. '
        '따라서 고정군을 개수 추정 개선으로 해석하지 않는다. '
        '한 seed·반복 개발검증이며 두 군의 학습 가능 파라미터 수와 gradient clip 대상이 다르다. '
        '이전 모든 층lr1e-5 비교와는 새 층 학습률·FP32 누적 순서가 다르다.','',
        '[사전 계획](TF_AXIS_ADAPTATION_PLAN_KO.md) · [전체 수치](TF_AXIS_ADAPTATION_RESULT.json) · '
        '[고정/공동 직접 비교](TF_AXIS_ADAPTATION_PAIRED.json) · [검산](TF_AXIS_ADAPTATION_AUDIT.json)']
    w.write(public/'TF_AXIS_ADAPTATION_KO.md','\n'.join(lines)+'\n');return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();print(audit(a.run.resolve(),a.public.resolve()),flush=True)
