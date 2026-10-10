"""Bounded CPU audit of saved integrated epochs, never raw I/Q or CUDA."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'reports/2026-10-11'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):result.update(block)
    return result.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def grouped(rows):
    result=defaultdict(lambda:dict(cases=0,nmse=[],si=[],weak=[]))
    for r in rows:
        if r['count']==1:continue
        key=' + '.join(sorted(r['categories']))
        cell=result[key];cell['cases']+=1
        cell['nmse'].extend(r['nmse']);cell['si'].extend(r['si_sdr'])
        cell['weak'].append(r['nmse'][r['weakest_index']])
    mean=lambda xs:sum(xs)/len(xs) if all(x is not None for x in xs) else None
    return {k:dict(cases=v['cases'],nmse=mean(v['nmse']),si_sdr=mean(v['si']),weakest_nmse=mean(v['weak']))
            for k,v in sorted(result.items())}


def checkpoint_audit(folder,event,p):
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    path=folder/'LAST.pt'
    assert sha(path)==event['last_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert saved['epoch']==event['epoch'] and saved['updates']==75*event['epoch']
    assert saved['protocol_sha256']==sha(folder/'PROTOCOL.json')
    assert saved['best']==event['best'] and saved['monitor']==event['monitor']
    parent=torch.load(p['parent_checkpoint'],map_location='cpu',weights_only=False,mmap=True)['model']
    state=saved['model'];checked=0
    for name,value in parent.items():
        mapped=('context_encoder.base.'+name[len('context_encoder.'):] if name.startswith('context_encoder.')
                else 'output.base.'+name[len('output.'):] if name.startswith('output.') else name)
        assert torch.equal(state[mapped],value),mapped
        checked+=1
    groups={'order':[],'axes':[],'head':[]}
    for name in state:
        if name=='context_encoder.order_gate':groups['order'].append(name)
        elif name.startswith('tf_axes.'):groups['axes'].append(name)
        elif name.startswith('output.') and not name.startswith('output.base.'):groups['head'].append(name)
    assert sum(state[n].numel() for names in groups.values() for n in names)==p['trainable']
    opt=saved['optimizer'];optimizer_params=[i for g in opt['param_groups'] for i in g['params']]
    assert set(optimizer_params)==set(opt['state']) and len(optimizer_params)==len(set(optimizer_params))
    gradient_history={}
    for group in opt['param_groups']:
        names=groups[group['name']]
        assert len(names)==len(group['params'])
        for name,index in zip(names,group['params']):
            value=opt['state'][index]
            assert int(value['step'])==event['updates']
            assert value['exp_avg'].shape==value['exp_avg_sq'].shape==state[name].shape
            assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
            gradient_history[name]=float(value['exp_avg_sq'].sum())
    probes=['context_encoder.order_gate','tf_axes.blocks.0.time.weight_ih_l0',
            'tf_axes.blocks.0.frequency.weight_ih_l0','output.input.weight',
            'output.qkv.weight','output.readout.weight','tf_axes.output.weight']
    assert all(gradient_history[n]>0 for n in probes)
    nonzero_readouts={n:float(state[n].norm()) for n in
        ['context_encoder.order_gate','output.readout.weight','tf_axes.output.weight']}
    assert all(value>0 for value in nonzero_readouts.values())
    assert all(torch.isfinite(value).all() for value in state.values())
    assert sha(path)==event['last_sha256'],'Concurrent replacement; retry with next receipt'
    return dict(status='PASS',frozen_tensors_bitwise_equal=checked,
                optimizer_parameter_tensors=len(optimizer_params),optimizer_steps=event['updates'],
                nonzero_gradient_history={n:gradient_history[n] for n in probes},
                initially_zero_parameters_norm=nonzero_readouts,
                gpu_initialized=torch.cuda.is_initialized(),checkpoint_sha256=event['last_sha256'])


def audit(folder,output,epoch):
    p=read(folder/'PROTOCOL.json');event=read(folder/f'EPOCH_{epoch:03d}.json')
    vp=folder/f'VALIDATION_{epoch:03d}.json'
    assert sha(vp)==event['validation_sha256']
    data=read(vp);baseline=read(p['baseline'])
    assert len(data['rows'])==len(baseline['rows'])==630
    for a,b in zip(data['rows'],baseline['rows']):
        for key in ('index','count','categories','pack_ids','reference_power'):
            assert a[key]==b[key],(key,a['index'])
    for group in event['validation']['by_count']:
        rows=[r for r in data['rows'] if r['count']==group['count']]
        assert len(rows)==210
        values=[v for r in rows for v in r['nmse']]
        assert abs(sum(values)/len(values)-group['mean_nmse'])<1e-12
        si=[v for r in rows for v in r['si_sdr']]
        if all(v is not None for v in si):assert abs(sum(si)/len(si)-group['mean_si_sdr'])<1e-10
        assert abs(sum(r['nmse'][r['weakest_index']] for r in rows)/210-group['weakest_nmse'])<1e-12
        assert all(r['sum_relative_error']<1e-9 for r in rows)
    guards=read(folder/f'GUARD_{epoch:03d}.json')['updates'];assert len(guards)==75
    for guard in guards:
        assert all(v<=t for v,t in zip(guard['protected_dot_after_fp32'],guard['fp32_constraint_tolerance']))
    checkpoint=checkpoint_audit(folder,event,p)
    before,after=grouped(baseline['rows']),grouped(data['rows'])
    assert before.keys()==after.keys()
    conditions=[dict(combination=k,parent=before[k],integrated=after[k]) for k in before]
    result=dict(status='PASS',epoch=epoch,all_rows=630,groups=event['validation']['by_count'],
        selected=event['best'],joint_success=event['joint_success'],checkpoint=checkpoint,
        all75_guard_constraints_pass=True,combinations=conditions,
        protocol_sha256=sha(folder/'PROTOCOL.json'),event_sha256=sha(folder/f'EPOCH_{epoch:03d}.json'),
        auditor_sha256=sha(__file__),heldout_read=False,raw_iq_read=False,time=time.time())
    write(output/f'AUDIT_{epoch:03d}.json',result)
    write(PUBLIC/f'INTEGRATED_AUDIT_E{epoch:03d}.json',result)
    lines=[f'# 통합 모델 e{epoch}: 저장 결과 CPU 검산','',
        '630개 개발 검증 행·참조 정체성·요약·optimizer 단계·고정 본체 불변·75개 파형 보호 제약을 확인했다. '
        '새 모듈 내부 LSTM/attention도 누적 기울기가0이 아니었다. 구현 검산이며 독립 일반화나 원인별 기여의 증거는 아니다.','',
        '|혼합|사례 수|기준→통합 NMSE ↓|기준→통합 SI-SDR ↑ dB|기준→통합 최약NMSE ↓|',
        '|---|---:|---|---|---|']
    fmt=lambda value:'미정의' if value is None else f'{value:.4f}'
    for cell in conditions:
        a,b=cell['parent'],cell['integrated']
        lines.append(f"|{cell['combination']}|{a['cases']}|{fmt(a['nmse'])}→{fmt(b['nmse'])}|"
                     f"{fmt(a['si_sdr'])}→{fmt(b['si_sdr'])}|{fmt(a['weakest_nmse'])}→{fmt(b['weakest_nmse'])}|")
    lines+=['',f"선택: e{event['best']['epoch']}. 두/세 성분 공동 개선 기준 충족: {event['joint_success']}.",
            '', '한 seed의 반복 개발 자료이며 각 구성의 독립 기여나 같은 예산 대조의 우월성을 주장하지 않는다.','']
    write(PUBLIC/f'INTEGRATED_AUDIT_E{epoch:03d}_KO.md','\n'.join(lines))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True);folder=args.run.resolve()
    write(output/'STATE.json',dict(status='WAITING_FOR_EPOCH',pid=os.getpid(),time=time.time()))
    while True:
        paths=sorted(folder.glob('EPOCH_*.json'))
        if paths:
            epoch=int(paths[-1].stem.split('_')[-1])
            if not (output/f'AUDIT_{epoch:03d}.json').exists():
                try:
                    result=audit(folder,output,epoch)
                    write(output/'STATE.json',dict(status='AUDITED',last_epoch=epoch,pid=os.getpid(),time=time.time()))
                    print(dict(status=result['status'],epoch=epoch),flush=True)
                except Exception:
                    write(output/'FAILURE.json',dict(epoch=epoch,traceback=traceback.format_exc(),time=time.time()))
                    raise
        if (folder/'COMPLETE.json').exists() or (folder/'FAILURE.json').exists():return
        time.sleep(10)


if __name__=='__main__':main()
