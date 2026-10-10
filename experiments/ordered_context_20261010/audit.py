"""Independent row, checkpoint, optimizer, gate and selection checks."""
import argparse
from pathlib import Path
import torch
import train as t
w=t.w


def audit(root):
    torch.set_num_threads(2);p=w.read(root/'PROTOCOL.json');t.verify(root,p)
    ph=w.digest(root/'PROTOCOL.json');old=p['original_protocol']
    assert w.digest(root/'VALIDATION_000.json')==old['baseline_sha256']
    parent,ids=w.validate(Path(old['baseline']),None)
    control,_=w.validate(Path(p['comparison_control']),ids)
    actual,_=w.validate(root/'VALIDATION_001.json',ids)
    e=w.read(root/'EPOCH_001.json');complete=w.read(root/'COMPLETE.json')
    assert e==complete['event'] and e['validation']==actual and e['protocol_sha256']==ph
    assert e['updates']==75 and e['count_examples']==[800,800,800] and len(e['gate_history'])==len(e['preclip_gradient_norms'])==75
    assert e['comparison']==t.compare(actual,parent,control) and e['criterion_met']==t.passed(e['comparison'])
    selected=1 if actual['selection_nmse']<parent['selection_nmse'] else 0
    assert e['best']==dict(epoch=selected,metric=min(actual['selection_nmse'],parent['selection_nmse']))
    a=torch.load(root/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
    last=torch.load(root/'LAST.pt',map_location='cpu',weights_only=False)
    best=torch.load(root/'BEST.pt',map_location='cpu',weights_only=False)
    initial=torch.load(old['parent_checkpoint'],map_location='cpu',weights_only=False)
    assert a['updates']==last['updates']==75 and a['epoch']==last['epoch']==1
    assert a['protocol_sha256']==last['protocol_sha256']==best['protocol_sha256']==ph
    assert last['best']==best['best']==e['best']
    assert set(a['model'])==set(last['model'])==set(best['model'])
    for key,value in a['model'].items():
        assert torch.isfinite(value).all() and torch.equal(value,last['model'][key])
        pk=t.parent_key(key)
        target=value if selected else initial['model'][pk] if pk is not None else torch.zeros_like(value)
        assert torch.equal(best['model'][key],target)
    gate=a['model']['context_encoder.order_gate']
    assert torch.allclose(gate.tanh(),torch.tensor(e['gate_history'][-1]),atol=1e-7,rtol=1e-7)
    assert torch.count_nonzero(gate)>0
    net=t.make_net();net.load_state_dict(a['model']);named=list(net.named_parameters())
    assert sum(q.numel() for _,q in named)==p['parameters']
    groups=last['optimizer']['param_groups'];states=last['optimizer']['state']
    assert len(groups)==2 and len(states)==len(named) and {int(s['step']) for s in states.values()}=={75}
    partitions=[[q for k,q in named if k!='context_encoder.order_gate'],[net.context_encoder.order_gate]]
    for group,params,lr,wd in zip(groups,partitions,[p['parent_lr'],p['gate_lr']],[1e-4,0.]):
        assert group['lr']==lr and group['weight_decay']==wd and len(group['params'])==len(params)
        for param,key in zip(params,group['params']):
            for name in ('exp_avg','exp_avg_sq'):
                value=states[key][name];assert value.shape==param.shape and torch.isfinite(value).all()
    report=dict(status='PASS',protocol_sha256=ph,complete_sha256=w.digest(root/'COMPLETE.json'),
        hashes={name:w.digest(root/name) for name in ('ACTUAL_001.pt','LAST.pt','BEST.pt','EPOCH_001.json','VALIDATION_001.json')},
        validation_rows_reaggregated=630,optimizer_steps_checked=75,selected_tensors_checked=True,
        gate_nonzero=True,independently_reran_inference=False,heldout_read=False)
    w.write(t.PUBLIC/'ORDERED_CONTEXT_AUDIT.json',report)
    lines=['# 時間 순서 문맥의 동일 예산 결과'.replace('時間','시간'),'',
        'RFUAV 같은 원 RF 대역, native 100 MS/s, TRAIN 8/DEV 5 원기록 묶음, seed 0. '
        '부모 전체와 게이트 65개를 새 schedule 3의 2,400혼합·75업데이트로 학습했다. '
        '부모 lr=1e-5, 게이트 lr=1e-2이며 대조는 같은 일정·원 손실·부모 lr=1e-5다. '
        '자료·업데이트 예산을 맞췄으며 FLOP 또는 새 파라미터 학습률이 동일한 비교는 아니다.','',
        '| 조건 | 성분 수 | NMSE ↓ | 복소 SI-SDR ↑ dB | 최약 NMSE ↓ | 합성 개수 정확도 |',
        '|---|---:|---:|---:|---:|---:|']
    for label,v in [('부모',parent),('같은 새 일정 대조',control),('시간 순서 게이트',actual)]:
        for g in v['by_count']:
            si='undefined' if g['mean_si_sdr'] is None else f"{g['mean_si_sdr']:.3f}"
            lines.append(f"|{label}|{g['count']}|{g['mean_nmse']:.6f}|{si}|{g['weakest_nmse']:.6f}|{100*g['construction_count_accuracy']:.2f}%|")
    lines+=['',f"선택 epoch={selected}; 부모와 대조에 대한 공동 채택 기준={e['criterion_met']}.",
        '', '모든 실패 조건과 실제 e1을 보존했다. 같은 DEV의 반복 선택이며 독립 시험이 아니다. '
        '게이트가 변한 사실은 입력 경로 학습의 확인이고 호핑 주기 이해의 증거가 아니다. '
        '긴 전력 경로의 순서는 유지하지만 긴 구간의 복소 위상을 제공한 실험은 아니다. '
        '합성 성분 수 정확도를 물리 드론 대수 정확도로 부르지 않는다.']
    (t.PUBLIC/'ORDERED_CONTEXT_KO.md').write_text('\n'.join(lines)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);a=parser.parse_args();audit(a.run.resolve())
