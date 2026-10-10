"""First-estimate error transfer with fixed original long-context conditioning.

This is a TRAIN diagnostic, not a trained recursive architecture. The second
forward receives the local residual and ORIGINAL mixture context explicitly.
Ground truth selects no output in the prediction-removal branch. It is used
only for scoring and the separately labeled ideal-removal comparison.
"""
import argparse
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback
import numpy as np
import torch
import diagnose as core

ROOT=core.ROOT;w=core.w

def run(root,public):
    root.mkdir(parents=True,exist_ok=True);assert not (root/'PROTOCOL.json').exists()
    oracle_path=ROOT/'local/oracle_removal_20261010_v1/COMPLETE.json'
    oracle=w.read(oracle_path);old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    assert w.read(oracle_path.with_name('AUDIT.json'))['status']=='PASS'
    expected={r['index']:r for r in oracle['original']};assert len(expected)==14
    sources=dict(old['source_sha256'])
    for name in ('diagnose.py','diagnose_predicted_removal.py'):
        path=Path(__file__).with_name(name);sources[str(path.relative_to(ROOT))]=w.digest(path)
    plan=ROOT/'reports/2026-10-10/PREDICTED_REMOVAL_PLAN_KO.md'
    protocol=dict(status='REGISTERED_CPU_PREDICTED_REMOVAL',indices=list(expected),
        sources=sources,plan_sha256=w.digest(plan),oracle_sha256=w.digest(oracle_path),
        parent_sha256=old['parent_checkpoint_sha256'],preparation_sha256=old['preparation_sha256'],
        selection='Highest mean power among three predicted source slots; references are not consulted',
        context='Original long mixture features retained for BOTH local-residual conditions; not residual-long features',
        updates=0,inferences=42,gpu_use=False,heldout_read=False,registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',protocol)
    assert w.digest(Path(old['parent_checkpoint']))==protocol['parent_sha256']
    assert w.digest(Path(old['preparation'])/'PREPARATION.json')==protocol['preparation_sha256']
    torch.set_num_threads(2)
    data=core.base.worker.NativeMixtures(old['preparation'],'train_pack',1)
    net=core.base.worker.make_model('retained_unet',Path(old['parent_checkpoint'])).eval()
    assert sum(p.numel() for p in net.parameters())==32142859
    rows=[];started=time.time()
    with torch.inference_mode():
        for position,(index,prior) in enumerate(expected.items()):
            raw=data[index];item=core.batch(raw)
            prediction,original_logits=core.base.worker.predict(net,item)
            original,assignment=core.metrics(prediction,item)
            for key in ('nmse','si_sdr','reference_power'):
                assert np.max(np.abs(np.array(original[key])-np.array(prior[key])))<2e-5
            # The deployable choice is made before inspecting any assignment.
            selected=int(prediction[0,:3].abs().square().mean(-1).argmax())
            estimate=prediction[0,selected].cpu().numpy()
            removed=original['assignment'].index(selected) # evaluation/oracle bookkeeping ONLY
            kept=[i for i in range(3) if i!=removed]
            refs=np.zeros_like(raw['references']);refs[:2]=raw['references'][kept]
            base=dict(raw,references=refs,active=np.array([True,True,False]),construction_count=2)
            predicted_residual=(raw['mixture']-estimate).astype(np.complex64)
            ideal_residual=(raw['mixture']-raw['references'][removed]).astype(np.complex64)
            error=estimate-raw['references'][removed]
            identity_error=float(np.mean(np.abs((predicted_residual-ideal_residual)+error).astype(np.float64)**2))
            identity_error/=max(float(np.mean(np.abs(raw['mixture']).astype(np.float64)**2)),1e-30)
            assert identity_error<1e-12
            conditions={}
            for mode,mixture in [('oracle_original_context',ideal_residual),('predicted_original_context',predicted_residual)]:
                subset=dict(base,mixture=mixture)
                assert np.array_equal(subset['context_features'],raw['context_features'])
                batch=core.batch(subset);second,logits=core.base.worker.predict(net,batch)
                metric,_=core.metrics(second,batch)
                metric['predicted_count']=int(logits.argmax(-1))+1
                assert torch.equal(logits,original_logits),'Count head should see identical original context'
                assert np.max(np.abs(np.array(metric['reference_power'])-np.array(original['reference_power'])[kept]))<1e-12
                conditions[mode]=metric
            long_oracle=next(r for r in oracle['rows'] if r['index']==index and r['removed_slot']==removed)
            row=dict(index=index,selected_prediction_slot=selected,matched_reference_slot=removed,kept_slots=kept,
                selected_reference_power_rank=sorted(range(3),key=lambda i:-original['reference_power'][i]).index(removed)+1,
                selected_reference_category=long_oracle['removed_category'],kept_categories=long_oracle['kept_categories'],
                first_estimate_nmse=original['nmse'][removed],first_estimate_si_sdr=original['si_sdr'][removed],
                first_error_over_remaining_energy=original['absolute_error_power'][removed]/sum(original['reference_power'][j] for j in kept),
                residual_error_identity_roundoff=identity_error,
                original_remaining_nmse=[original['nmse'][j] for j in kept],
                original_remaining_si_sdr=[original['si_sdr'][j] for j in kept],
                oracle_recomputed_context_nmse=long_oracle['nmse'],oracle_recomputed_context_si_sdr=long_oracle['si_sdr'],
                conditions=conditions)
            rows.append(row)
            w.write(root/'PARTIAL.json',dict(rows=rows))
            w.write(root/'STATE.json',dict(status='CPU_PREDICTED_REMOVAL',cases=position+1,total_cases=14,
                inferences=3*(position+1),total_inferences=42,pid=os.getpid(),time=time.time()))
    summaries=[]
    for mode in ('original_remaining','oracle_recomputed_context','oracle_original_context','predicted_original_context'):
        nmse=[];si=[];both=0
        for row in rows:
            n=row[mode+'_nmse'] if mode not in row['conditions'] else row['conditions'][mode]['nmse']
            s=row[mode+'_si_sdr'] if mode not in row['conditions'] else row['conditions'][mode]['si_sdr']
            nmse.extend(n);si.extend(s)
            both+=sum(a<b and c>d for a,b,c,d in zip(n,row['original_remaining_nmse'],s,row['original_remaining_si_sdr']))
        summaries.append(dict(mode=mode,cases=14,source_contributions=28,nmse=statistics.mean(nmse),
            si_sdr=statistics.mean(si),both_improved_sources_vs_original=both))
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    assert w.digest(plan)==protocol['plan_sha256'] and w.digest(oracle_path)==protocol['oracle_sha256']
    result=dict(status='COMPLETE_CHECKED',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,summary=summaries,
        selected_reference_rank_counts={str(i):sum(r['selected_reference_power_rank']==i for r in rows) for i in (1,2,3)},
        first_error_over_remaining_energy_mean=statistics.mean(r['first_error_over_remaining_energy'] for r in rows),
        seconds=time.time()-started,updates=0,gpu_use=False,heldout_read=False,
        limitation='Fixed TRAIN14. Original long context retained. Second-pass PIT assesses remaining references, not a complete variable-count recursive system. No recursive training or full-long predicted residual inference.')
    w.write(root/'COMPLETE.json',result);w.write(public/'PREDICTED_REMOVAL_RESULT.json',result)
    lines=['# 첫 추출 오차가 두 번째 분리에 미치는 영향','',
        '고정TRAIN14의 세 성분 혼합에서 예측3슬롯 중 평균 전력이 가장 큰 하나를 먼저 뺐다. '
        '출력 선택에 정답을 사용하지 않았다. 정답 대응은 평가 및 별도 oracle 조건에만 사용했다. '
        '원본14회+잔여혼합28회로 총42회 CPU 추론이며 학습·DEV·보류자료 접근은 없다.','',
        '| 조건 | 남은 성분 NMSE ↓ | 복소 SI-SDR ↑ dB | 원래 대비 두 지표 개선 |',
        '|---|---:|---:|---:|']
    names={'original_remaining':'원래 동시 분리의 남은 두 성분','oracle_recomputed_context':'정답 제거·전체 잔여 문맥 재계산',
           'oracle_original_context':'정답 제거·원래 혼합 문맥 유지','predicted_original_context':'예측 제거·원래 혼합 문맥 유지'}
    for row in summaries:lines.append(f"|{names[row['mode']]}|{row['nmse']:.6f}|{row['si_sdr']:.3f}|{row['both_improved_sources_vs_original']}/28|")
    lines+=['','**원래 혼합 문맥을 유지한 조건**임을 명시한다. 두 잔여 조건의 문맥을 같게 하여 첫 추출 오류를 비교했다. '
        '실제 잔여20.8896ms를 전부 추론해 문맥을 다시 계산한 결과가 아니다. '
        '두 번째 U-Net의 출력은 세 슬롯과 배경이며, 남은 두 정답에 대한 PIT는 평가용이다. '
        '가변 개수 순차 모델의 완성된 성능으로 보고하지 않는다.', '',
        f"첫 예측이 대응한 성분의 전력 순위 집계: {result['selected_reference_rank_counts']}. "
        f"첫 오차/남은 두 정답 에너지 비 평균: {result['first_error_over_remaining_energy_mean']:.6f}.",
        '', '[모든14행과 조건](PREDICTED_REMOVAL_RESULT.json)']
    (public/'PREDICTED_REMOVAL_KO.md').write_text('\n'.join(lines)+'\n')
    w.write(root/'STATE.json',dict(status='COMPLETE_CHECKED',cases=14,inferences=42,pid=os.getpid(),time=time.time()))
    print(dict(summary=summaries,seconds=result['seconds']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args()
    try:run(a.run.resolve(),a.public.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
