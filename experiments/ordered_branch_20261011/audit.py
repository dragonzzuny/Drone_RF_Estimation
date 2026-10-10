"""CPU weights/optimizer/paired DEV receipts, without waveform reinference."""
from pathlib import Path
import numpy as np
import torch
import architecture as model
import evaluation

ROOT,w,worker=model.ROOT,model.w,model.worker
OUT=ROOT/'local/ordered_branch_20261011_v1'
PUBLIC=ROOT/'reports/2026-10-11'


def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    p=w.read(OUT/'PROTOCOL.json');r=w.read(OUT/'COMPLETE.json')
    assert r==w.read(PUBLIC/'ORDERED_BRANCH_RESULT.json')
    assert r['protocol_sha256']==w.digest(OUT/'PROTOCOL.json')
    for rel,h in p['source_sha256'].items():
        assert w.digest(ROOT/rel)==h and w.digest(OUT/'source_snapshot'/rel)==h
    for f,h in p['pinned_files'].items():assert w.digest(Path(f))==h
    initial=torch.load(OUT/'INITIAL.pt',map_location='cpu',weights_only=False,mmap=True)
    final=torch.load(OUT/'ACTUAL_001.pt',map_location='cpu',weights_only=False,mmap=True)
    parent=torch.load(p['parent_checkpoint'],map_location='cpu',weights_only=False,mmap=True)['model']
    assert final['updates']==75 and final['epoch']==1
    assert initial['model'].keys()==final['model'].keys()
    frozen=[];changed=[]
    for key,value in final['model'].items():
        assert torch.isfinite(value).all(),key
        if key.startswith('parent.'):
            assert torch.equal(value,parent[key[7:]]) and torch.equal(value,initial['model'][key])
            frozen.append(key)
        elif not torch.equal(value,initial['model'][key]):changed.append(key)
    assert len(frozen)==131 and 'gate' in changed
    assert any(k.startswith('ordered_encoder.') for k in changed)
    assert any(k.startswith('ordered_projection.') for k in changed)
    assert {int(v['step']) for v in final['optimizer']['state'].values()}=={75}
    history=r['training']['gradient_history'];assert len(history)==75
    assert history[0]['ordered_encoder']==0 and history[0]['ordered_projection']==0 and history[0]['gate']>0
    assert all(all(v>0 and np.isfinite(v) for v in row.values()) for row in history[1:])
    assert r['training']['counts']==[800,800,800]
    paired={};checks={}
    for name,key in [('single','baseline'),('four_phase','four_phase_baseline')]:
        data=r[name];baseline=w.read(Path(p[key]));calc=evaluation.summarize(data['rows'])
        assert calc==data
        checks[name]=evaluation.compare(data,baseline)
        sources=[]
        for a,b in zip(baseline['rows'],data['rows']):
            for field in ('index','count','categories','pack_ids','nominal_levels_db','reference_power','input_si_sdr','predicted_count'):
                assert a[field]==b[field],(a['index'],field)
            for i in range(a['count']):
                dn=b['nmse'][i]-a['nmse'][i];ds=b['si_sdr'][i]-a['si_sdr'][i]
                sources.append(dict(index=a['index'],source=i,count=a['count'],category=a['categories'][i],
                    nmse_delta=dn,si_sdr_delta=ds,jointly_better=dn<0 and ds>0,jointly_worse=dn>0 and ds<0))
        assert len(sources)==1260
        paired[name]=dict(sources=1260,jointly_better=sum(x['jointly_better'] for x in sources),
            jointly_worse=sum(x['jointly_worse'] for x in sources),rows=sources)
    assert checks==r['checks']
    candidate=all(all(v for k,v in c.items() if k!='count') for rows in checks.values() for c in rows)
    assert candidate==r['candidate'] and r['selected_epoch']==int(candidate)
    selected=OUT/('ACTUAL_001.pt' if candidate else 'INITIAL.pt')
    assert w.digest(selected)==w.digest(OUT/'SELECTED.pt')==r['selected_checkpoint_sha256']
    assert w.digest(OUT/'ACTUAL_001.pt')==r['actual_checkpoint_sha256']
    audit=dict(status='PASS',frozen_parent_tensors=len(frozen),changed_branch_tensors=changed,
        optimizer_steps=75,train_examples=2400,dev_cases_per_inference=630,paired=paired,
        checks=checks,candidate=candidate,selected_epoch=r['selected_epoch'],
        independent_cpu_waveform_reinference=False,heldout_read=False,
        result_sha256=w.digest(OUT/'COMPLETE.json'),auditor_sha256=w.digest(Path(__file__)))
    w.write(PUBLIC/'ORDERED_BRANCH_AUDIT.json',audit)
    lines=['# 보존 U-Net + 별도 시간 변화 경로: 추가1epoch 결과','',
        f"전체2400혼합·75업데이트 완료. 기존131개 tensor 보존과 optimizer·630행 CPU 감사 통과. 선택e{r['selected_epoch']}.",
        '', '|추론|성분 수|기존 NMSE|새 NMSE|기존 복소SI-SDR dB|새 복소SI-SDR dB|최약 NMSE 기존→새|',
        '|---|---|---:|---:|---:|---:|---|']
    for name,key in [('single','baseline'),('four_phase','four_phase_baseline')]:
        for a,b in zip(w.read(Path(p[key]))['by_count'],r[name]['by_count']):
            lines.append(f"|{name}|{a['count']}|{a['mean_nmse']:.6f}|{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}|{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}→{b['weakest_nmse']:.6f}|")
    lines+=['', '사전 공동 개선 기준을 '+('통과했다. 후속 독립 확인 전이다.' if candidate else '실패해 첫epoch에서 중지했다. 기존e0를 유지한다.'),
        f"학습{r['training']['train_seconds']:.1f}초, 단일/4위상 공동평가{r['evaluation_seconds']:.1f}초.",
        '', '원 모델과 별도 시간 인코더·투영·gate를 합쳐32,878,091파라미터 중735,232개를 학습했다. '
        '원 모델의 전체 용량은 유지했다. 기존 대조를 재학습하지 않았으며 학습 대상·학습률 차이가 있다.',
        '', '한 seed, 반복 DEV, 같은 TRAIN 원기록·다른 합성 recipe. DEV VTSBW20은 TRAIN에 없다. '
        'Autel/별도 확인 자료 미개봉. 시간 구조 자체가 일반적으로 유효/무효하다는 결론은 아니다. '
        '긴 전력 문맥에 복소 위상은 없고, 수집된 원기록의 대역 제한 기여 파형을 목표로 한다.',
        '', 'CPU는 가중치·optimizer·수치 행과 선택을 검산했으며 전체 파형 독립 재추론은 하지 않았다. '
        '개수 경로를 고정했으므로 개수 추정 개선은 없다. 실제 물리 드론 대수나 드론 식별 성능을 평가한 결과가 아니다.','']
    (PUBLIC/'ORDERED_BRANCH_RESULT_KO.md').write_text('\n'.join(lines))
    print(dict(status='PASS',candidate=candidate,selected_epoch=r['selected_epoch'],frozen_tensors=131),flush=True)


if __name__=='__main__':main()
