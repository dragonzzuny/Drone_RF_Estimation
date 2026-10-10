"""Reaggregate all DEV rows; audit parameters, optimizer, selection and recipes."""
import gc
import argparse
import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('fresh_schedule_training',Path(__file__).with_name('train.py'))
t=importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
w=t.w


def audit(root,public):
    torch.set_num_threads(2); p=w.read(root/'PROTOCOL.json'); t.verify(root,p)
    ph=w.digest(root/'PROTOCOL.json'); old=p['original_protocol']
    assert w.read(root/'SCHEDULE_AUDIT.json')==t.schedule_audit(old['preparation'])
    parent,ids=w.validate(Path(old['baseline']),None)
    control,_=w.validate(Path(p['historical_control']),ids)
    actual,_=w.validate(root/'VALIDATION_001.json',ids)
    e=w.read(root/'EPOCH_001.json'); complete=w.read(root/'COMPLETE.json')
    assert e==complete['event'] and e['validation']==actual and e['protocol_sha256']==ph
    assert e['updates']==75 and e['count_examples']==[800,800,800] and len(e['preclip_gradient_norms'])==75
    assert e['comparison']==t.core.base.compare(actual,parent,control)
    assert e['criterion_met']==t.passed(e['comparison'])
    selected=1 if actual['selection_nmse']<parent['selection_nmse'] else 0
    assert e['best']['epoch']==selected
    a=torch.load(root/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
    last=torch.load(root/'LAST.pt',map_location='cpu',weights_only=False)
    best=torch.load(root/'BEST.pt',map_location='cpu',weights_only=False)
    initial=torch.load(old['parent_checkpoint'],map_location='cpu',weights_only=False)
    assert a['updates']==last['updates']==75 and a['epoch']==last['epoch']==1
    assert a['protocol_sha256']==last['protocol_sha256']==best['protocol_sha256']==ph
    assert last['best']==best['best']==e['best']
    target=a['model'] if selected else initial['model']
    assert set(a['model'])==set(last['model'])==set(best['model'])==set(target)
    for key,value in a['model'].items():
        assert torch.isfinite(value).all() and torch.equal(value,last['model'][key])
        assert torch.equal(best['model'][key],target[key])
    net=t.worker.make_model('retained_unet'); net.load_state_dict(a['model']); params=list(net.parameters())
    assert sum(q.numel() for q in params)==32142859
    group,=last['optimizer']['param_groups']; states=last['optimizer']['state']
    assert group['lr']==1e-5 and group['weight_decay']==1e-4
    assert len(group['params'])==len(params)==len(states) and {int(s['step']) for s in states.values()}=={75}
    for param,key in zip(params,group['params']):
        for name in ('exp_avg','exp_avg_sq'):
            value=states[key][name]; assert value.shape==param.shape and torch.isfinite(value).all()
    hashes={name:w.digest(root/name) for name in ('ACTUAL_001.pt','LAST.pt','BEST.pt','EPOCH_001.json','VALIDATION_001.json')}
    report=dict(status='PASS',protocol_sha256=ph,complete_sha256=w.digest(root/'COMPLETE.json'),hashes=hashes,
        validation_rows_reaggregated=630,source_and_schedule_hashes_checked=True,optimizer_steps_checked=75,
        selected_tensors_checked=True,independently_reran_inference=False,heldout_read=False)
    w.write(public/'FRESH_SCHEDULE_AUDIT.json',report)
    lines=['# 새 혼합 일정의 동일 예산 비교','',
        'RFUAV 같은 원RF대역·native100MS/s·기록 묶음 분할·seed0다. 부모32.14M 전체를 원 손실과 새AdamW1e-5로75업데이트했다. '
        '기존schedule1 대조의 완료결과를 재사용한다. 새schedule3도 같은 TRAIN 원기록에서 만든 혼합이다.','',
        '| 조건 | NMSE2/3 ↓ | 복소SI-SDR2/3 ↑ dB | 최약NMSE2/3 ↓ |','|---|---|---|---|']
    for label,v in [('부모',parent),('재사용 혼합 schedule1 대조',control),('부모에 새 혼합 schedule3',actual)]:
        x,y=v['by_count'][1:]
        lines.append(f"|{label}|{x['mean_nmse']:.6f}/{y['mean_nmse']:.6f}|{x['mean_si_sdr']:.3f}/{y['mean_si_sdr']:.3f}|{x['weakest_nmse']:.6f}/{y['weakest_nmse']:.6f}|")
    lines += ['',f"선택epoch={selected}; 두·세 성분 공동 채택 기준={e['criterion_met']}.",
        '', '현재 체크포인트의 native 학습에서 보지 않은 합성 예제이며 새 녹음이 아니다. '
        '무작위 자료 구성도 달라지므로 암기/과적합의 단독 인과효과로 해석하지 않는다. '
        'schedule3는 이전 개발 비교에서 사용됐고 동일DEV630을 반복 사용했다. 독립확인은 아직 수행하지 않았다.']
    (public/'FRESH_SCHEDULE_KO.md').write_text('\n'.join(lines)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True); args=parser.parse_args()
    audit(args.run.resolve(),args.public.resolve())
