"""TRAIN48 target-frequency pooling diagnosis; no U-Net modification."""
import argparse
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback
import torch
from torch.nn import functional as F
import diagnose as core
from drone_rf.waveform import analyze
ROOT=core.ROOT;w=core.w


def run(root,public):
    root.mkdir(parents=True,exist_ok=True);assert not (root/'PROTOCOL.json').exists()
    fit_path=ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json'
    old_path=ROOT/'reports/2026-10-10/SOURCE_DOMINANCE_RESULT.json'
    fit={r['index']:r for r in w.read(fit_path)['rows'] if r['model']=='parent/e0'}
    old={(r['index'],r['slot']):r for r in w.read(old_path)['rows']}
    prior=w.read(ROOT/'local/source_dominance_20261010_v1/PROTOCOL.json')
    old_p=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    sources=dict(prior['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    p=dict(status='REGISTERED_CPU_REFERENCE_FREQUENCY_RESOLUTION',indices=prior['indices'],
        pooling_factors=[1,2,4,8,16],pooling_axis='Frequency only; time500 frames unchanged',
        source_sha256=sources,prior_result_sha256=w.digest(old_path),train_fit_sha256=w.digest(fit_path),
        preparation_sha256=old_p['preparation_sha256'],updates=0,gpu_use=False,heldout_read=False,registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',p)
    torch.set_num_threads(2);data=core.base.worker.NativeMixtures(old_p['preparation'],'train_pack',1);rows=[];started=time.time()
    with torch.inference_mode():
        for position,index in enumerate(p['indices']):
            raw=data[index];n=int(raw['construction_count']);ref=torch.as_tensor(raw['references'][:n])
            power=analyze(ref).to(torch.complex128).abs().square()
            for factor in p['pooling_factors']:
                pooled=F.avg_pool2d(power[None],(factor,1))[0];winner=pooled.argmax(0)
                for j in range(n):
                    energy=float(pooled[j].sum());coverage=float(pooled[j][winner==j].sum())/energy
                    previous=old[(index,j)]
                    if factor==1:assert abs(previous['source_energy_in_winning_bins']-coverage)<1e-12
                    rows.append(dict(index=index,count=n,slot=j,category=previous['category'],weakest=previous['weakest'],
                        frequency_pool=factor,frequency_bin_hz=100e6/512*factor,
                        source_energy_in_winning_bins=coverage,
                        parent_nmse=fit[index]['nmse'][j],parent_si_sdr=fit[index]['si_sdr'][j]))
            w.write(root/'STATE.json',dict(status='CPU_TARGET_RESOLUTION',cases=position+1,total=48,pid=os.getpid(),time=time.time()))
    assert len(rows)==500
    summaries=[]
    for n in (1,2,3):
        for factor in p['pooling_factors']:
            values=[r['source_energy_in_winning_bins'] for r in rows if r['count']==n and r['frequency_pool']==factor and r['weakest']]
            summaries.append(dict(count=n,frequency_pool=factor,cases=len(values),
                mean_weak_coverage=statistics.mean(values),median_weak_coverage=statistics.median(values)))
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    result=dict(status='COMPLETE_CHECKED',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,summary=summaries,
        original_full_resolution_100_rows_reproduced=True,updates=0,gpu_use=False,heldout_read=False,
        seconds=time.time()-started,
        limitation='Reference power pooling, not an actual learned-feature intervention or a proof that U-Net skips lost source information. TRAIN48 descriptive analysis only.')
    w.write(root/'COMPLETE.json',result);w.write(public/'SOURCE_RESOLUTION_RESULT.json',result)
    lines=['# 주파수 해상도와 약한 성분: 정답 전력 진단','',
        '기존 TRAIN48의 정답100성분을 같은 STFT512/hop128로 계산하고 주파수 칸만1/2/4/8/16개씩 평균했다. '
        '시간500프레임은 모두 유지했다. 비풀링100행은 이전 우세도 결과와1e-12이내 일치했다. '
        'U-Net 특징 자체를 바꾸거나 모델을 학습한 결과가 아니다.','',
        '| 성분 수 | 주파수 풀링 | 칸 폭 kHz | 최약 성분 사례 수 | 우세 칸의 자기 에너지 평균 | 중앙값 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        if s['count']>1:
            lines.append(f"|{s['count']}|{s['frequency_pool']}|{195.3125*s['frequency_pool']:.3f}|{s['cases']}|{100*s['mean_weak_coverage']:.2f}%|{100*s['median_weak_coverage']:.2f}%|")
    lines+=['','고정 TRAIN의 index37에서는 약한FPV 성분이 우세한 원래 칸에 자기 에너지 약97.95%가 있었지만 부모 NMSE는약0.9596이었다. '
        '모든 실패가 시간·주파수 완전 중첩 때문이라고 설명할 수 없다는 사례다. '
        '이 특정 사례로 빈도나 기종 전체의 성능을 추정하지 않는다.',
        '', '풀링한 정답 전력은 학습한 복소 특징과 다르며, 현재 U-Net에는 고해상도 skip 경로도 있다. '
        '따라서 이 표만으로 인코더가 분리 정보를 지운다고 결론내리지 않는다. '
        '보조 친화도 층은2×2풀링이므로 여기의주파수축만 바꾼2배 결과와도 같지 않다.',
        '', '[500행·모든조건](SOURCE_RESOLUTION_RESULT.json) · [우세도 원 분석](SOURCE_DOMINANCE_KO.md)']
    (public/'SOURCE_RESOLUTION_KO.md').write_text('\n'.join(lines)+'\n')
    w.write(root/'STATE.json',dict(status='COMPLETE_CHECKED',cases=48,pid=os.getpid(),time=time.time()))
    print(dict(status=result['status'],seconds=result['seconds'],summary=summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args()
    try:run(a.run.resolve(),a.public.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
