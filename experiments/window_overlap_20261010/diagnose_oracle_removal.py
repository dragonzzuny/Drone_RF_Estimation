"""Can the retained separator recover two sources after ideal first removal?

TRAIN-only counterfactual diagnostic; removing a true source is an oracle,
not a deployable recursive separator. Preserve original complex gains and
rebuild the entire observed long-context mixture, not stale input features.
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
from drone_rf.context_data import component_gains,mixture_context_features
ROOT=core.ROOT;w=core.w


def run(root,public):
    root.mkdir(parents=True,exist_ok=True);assert not (root/'PROTOCOL.json').exists()
    fit_path=ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json'
    fixed=[q for q in w.read(fit_path)['rows'] if q['model']=='parent/e0' and q['count']==3]
    assert len(fixed)==14
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    sources=dict(old['source_sha256'])
    for name in ('diagnose.py','diagnose_oracle_removal.py'):
        q=Path(__file__).with_name(name);sources[str(q.relative_to(ROOT))]=w.digest(q)
    p=dict(status='REGISTERED_CPU_ORACLE_REMOVAL',indices=[q['index'] for q in fixed],
        source_sha256=sources,parent_sha256=old['parent_checkpoint_sha256'],preparation_sha256=old['preparation_sha256'],
        fit_sha256=w.digest(fit_path),selection='All14 three-source cases of existing fixed TRAIN48; remove EACH of3 sources',
        original_gains_preserved=True,long_context_recomputed=True,original_inferences=14,oracle_inferences=42,
        updates=0,gpu_use=False,heldout_read=False,oracle=True,registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',p)
    torch.set_num_threads(2);data=core.base.worker.NativeMixtures(old['preparation'],'train_pack',1)
    net=core.base.worker.make_model('retained_unet',Path(old['parent_checkpoint'])).eval()
    rows=[];original=[];started=time.time()
    with torch.inference_mode():
        for position,prior in enumerate(fixed):
            index=prior['index'];scheduled=data.rows[index];raw=data[index]
            ids=scheduled['indices'][:3];start=int(raw['crop_start']);length=raw['mixture'].shape[-1]
            gains=component_gains([data.library.clips[int(i)]['mean_power'] for i in ids],scheduled['levels'][:3],scheduled['phases'][:3])
            components=[(data.library._array(int(i))*gain).astype(np.complex64) for i,gain in zip(ids,gains)]
            long_mix=np.zeros_like(components[0])
            for component in components:long_mix+=component
            recomputed=mixture_context_features(long_mix)['context_features']
            assert np.array_equal(recomputed,raw['context_features'])
            assert np.array_equal(long_mix[start:start+length],raw['mixture'])
            item=core.batch(raw);prediction,_=core.base.worker.predict(net,item)
            current,_=core.metrics(prediction,item)
            for key in ('nmse','si_sdr','reference_power'):assert np.max(np.abs(np.array(current[key])-np.array(prior[key])))<2e-5
            original.append(dict(index=index,**current))
            ranks=np.argsort(-np.asarray(current['reference_power']))
            for removed in range(3):
                kept=[j for j in range(3) if j!=removed]
                remaining=(components[kept[0]]+components[kept[1]]).astype(np.complex64)
                refs=np.zeros_like(raw['references'])
                refs[:2]=raw['references'][kept]
                subset=dict(mixture=remaining[start:start+length].copy(),references=refs,
                    active=np.any(refs!=0,axis=1),context_features=mixture_context_features(remaining)['context_features'],
                    crop_start=start,construction_count=2)
                assert np.array_equal(subset['mixture'],refs.sum(0))
                residual_roundoff=np.mean(np.abs((raw['mixture']-raw['references'][removed])-subset['mixture']).astype(np.float64)**2)
                residual_roundoff/=max(float(np.mean(np.abs(raw['mixture']).astype(np.float64)**2)),1e-30)
                assert residual_roundoff<1e-12
                batch=core.batch(subset);estimate,logits=core.base.worker.predict(net,batch);metric,_=core.metrics(estimate,batch)
                row=dict(index=index,removed_slot=removed,removed_category=prior['categories'][removed],
                    removed_power_rank=int(np.flatnonzero(ranks==removed)[0])+1,
                    kept_slots=kept,kept_categories=[prior['categories'][j] for j in kept],
                    parent_kept_nmse=[current['nmse'][j] for j in kept],
                    parent_kept_si_sdr=[current['si_sdr'][j] for j in kept],
                    predicted_count=int(logits.argmax(-1))+1,subtraction_roundoff_relative=residual_roundoff,**metric)
                assert np.max(np.abs(np.array(row['reference_power'])-np.array(current['reference_power'])[kept]))<1e-12
                rows.append(row)
            w.write(root/'STATE.json',dict(status='CPU_ORACLE_REMOVAL',cases=position+1,total_cases=14,
                inferences=4*(position+1),total_inferences=56,pid=os.getpid(),time=time.time()))
    assert len(rows)==42 and len(original)==14
    summary=[]
    for rank in (1,2,3):
        selected=[r for r in rows if r['removed_power_rank']==rank]
        assert len(selected)==14
        summary.append(dict(removed_power_rank=rank,cases=14,source_contributions=28,
            parent_kept_nmse=statistics.mean(x for r in selected for x in r['parent_kept_nmse']),
            oracle_remaining_nmse=statistics.mean(x for r in selected for x in r['nmse']),
            parent_kept_si_sdr=statistics.mean(x for r in selected for x in r['parent_kept_si_sdr']),
            oracle_remaining_si_sdr=statistics.mean(x for r in selected for x in r['si_sdr']),
            both_improved_sources=sum(a<b and c>d for r in selected for a,b,c,d in zip(r['nmse'],r['parent_kept_nmse'],r['si_sdr'],r['parent_kept_si_sdr']))))
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    result=dict(status='COMPLETE_CHECKED',protocol_sha256=w.digest(root/'PROTOCOL.json'),original=original,rows=rows,summary=summary,
        seconds=time.time()-started,original_parent_metrics_reproduced=True,original_contexts_bitwise_reproduced=True,
        all_remaining_gains_preserved=True,oracle=True,heldout_read=False,updates=0,gpu_use=False,
        limitation='Ideal removal with ground truth, fixed TRAIN14; not deployable recursive separation, not a guarantee/bound for training OR-PIT')
    w.write(root/'COMPLETE.json',result);w.write(public/'ORACLE_REMOVAL_RESULT.json',result)
    lines=['# 한 성분이 정확히 제거됐을 때: 순차 분리의 사전 진단','',
        '기존 TRAIN48 중 세 성분14혼합 모두에서 각각의 정답 성분을 하나씩 제거했다. '
        '남은 성분의 복소 배율은 유지하고20.8896ms 전체 혼합과 문맥 특징을 다시 만들었다. '
        '기존 U-Net을14회 원본+42회 잔여혼합에 적용했다. 원본 문맥은 bitwise 일치하고 원본 지표도 재현됐다. '
        '정답 제거를 사용하는 oracle 진단이며 실제 복원 성과로 보고하지 않는다.','',
        '| 제거한 성분 전력 순위 | 남은 성분 수 | 원래 NMSE ↓ | 이상적 제거 후 NMSE ↓ | 원래 복소SI-SDR ↑ dB | 제거 후 복소SI-SDR ↑ dB | 두 지표 모두 개선 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"|{s['removed_power_rank']}|28|{s['parent_kept_nmse']:.6f}|{s['oracle_remaining_nmse']:.6f}|{s['parent_kept_si_sdr']:.3f}|{s['oracle_remaining_si_sdr']:.3f}|{s['both_improved_sources']}/28|")
    lines+=['','순위1이 가장 강한 성분이다.각14혼합의 남은28성분을 같은 파형끼리 비교했다. '
        '실제 순차 분리에서는 첫 추출 오차가 잔여 입력에 들어간다. '
        '이 실험은 그 오차를0으로 둔 가정의 진단이며 순차 구조를 학습한 결과가 아니다. '
        '정답 개수는 평가와 합성 기록에만 쓰고, U-Net forward는 잔여 혼합과 그 문맥만 받는다. '
        '추가 학습이나 DEV·보류 자료 접근은 없었다.',
        '', '[42가지 제거와원본14행](ORACLE_REMOVAL_RESULT.json)']
    (public/'ORACLE_REMOVAL_KO.md').write_text('\n'.join(lines)+'\n')
    w.write(root/'STATE.json',dict(status='COMPLETE_CHECKED',cases=14,inferences=56,pid=os.getpid(),time=time.time()))
    print(dict(status=result['status'],summary=summary,seconds=result['seconds']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args()
    try:run(a.run.resolve(),a.public.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
