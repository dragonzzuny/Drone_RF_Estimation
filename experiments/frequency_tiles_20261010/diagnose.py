"""Fixed TRAIN6 full-capacity frequency tiling diagnosis; no optimization."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback
import numpy as np
import torch
from tiles import predict

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'experiments/window_overlap_20261010/diagnose.py'
spec=importlib.util.spec_from_file_location('window_core_for_tiles',path)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
from drone_rf.waveform import analyze,synthesize
w=core.w


def run(root):
    root.mkdir(parents=True,exist_ok=True);assert not (root/'PROTOCOL.json').exists()
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    fixed=[r for r in w.read(ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json')['rows'] if r['model']=='parent/e0']
    chosen=[r for n in (1,2,3) for r in [v for v in fixed if v['count']==n][:2]]
    check=ROOT/'reports/2026-10-10/FREQUENCY_TILES_ALGEBRA_CHECK.json'
    assert w.read(check)['status']=='PASS'
    sources=dict(old['source_sha256']);sources.update(w.read(check)['sources'])
    for path in (Path(__file__),ROOT/'experiments/window_overlap_20261010/diagnose.py'):
        sources[str(path.relative_to(ROOT))]=w.digest(path)
    protocol=dict(status='REGISTERED_CPU_FREQUENCY_TILES_TRAIN6',indices=[r['index'] for r in chosen],
        sources=sources,check_sha256=w.digest(check),parent_sha256=old['parent_checkpoint_sha256'],
        preparation_sha256=old['preparation_sha256'],width=128,step=64,total_bins=512,
        samples=63872,parameters=32142859,cases=6,inferences=54,optimizer_steps=0,gpu_use=False,heldout_read=False,
        selection='First two cases per count of pre-existing fixed TRAIN48',
        modes=['parent','tiles','blend50'],alignment='Complex weighted distance to original full-band predictions, six source permutations; background fixed',
        limitation='Inference distribution intervention, including per-band normalization and lost across-band context; not trained frequency modeling or causal pooling ablation',
        registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',protocol);torch.set_num_threads(2)
    data=core.base.worker.NativeMixtures(old['preparation'],'train_pack',1)
    net=core.base.worker.make_model('retained_unet',Path(old['parent_checkpoint'])).eval()
    assert sum(p.numel() for p in net.parameters())==32142859
    rows=[];started=time.time()
    with torch.inference_mode():
        for prior in chosen:
            item=core.batch(data[prior['index']]);z=analyze(item['mixture'])
            spectra,logits,trace=predict(net,z,item['context_features'],item['crop_start'])
            metric={}
            for mode,spectrum in spectra.items():
                waveform=synthesize(spectrum,item['mixture'].shape[-1])
                metric[mode],_=core.metrics(waveform,item)
            for key in ('nmse','si_sdr','reference_power'):
                assert np.max(np.abs(np.array(metric['parent'][key])-np.array(prior[key])))<2e-5
            rows.append(dict(index=prior['index'],count=prior['count'],categories=prior['categories'],
                modes=metric,tile_alignment=trace,predicted_count=int(logits.argmax(-1))+1))
            w.write(root/'ROWS.json',rows)
            w.write(root/'STATE.json',dict(status='CPU_FREQUENCY_TILES',cases=len(rows),total_cases=6,
                inferences=len(rows)*9,total_inferences=54,pid=os.getpid(),time=time.time()))
    summary=[]
    for mode in protocol['modes']:
        for n in (1,2,3):
            selected=[r['modes'][mode] for r in rows if r['count']==n]
            summary.append(dict(mode=mode,count=n,cases=len(selected),
                nmse=statistics.mean(v for r in selected for v in r['nmse']),
                si_sdr=statistics.mean(v for r in selected for v in r['si_sdr']),
                weakest_nmse=statistics.mean(r['nmse'][r['weakest_index']] for r in selected)))
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    assert w.digest(Path(old['parent_checkpoint']))==protocol['parent_sha256']
    result=dict(status='COMPLETE_CHECKED',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,summary=summary,
        seconds=time.time()-started,optimizer_steps=0,gpu_use=False,heldout_read=False,dev_read=False,
        limitation=protocol['limitation'])
    public=ROOT/'reports/2026-10-10';w.write(root/'COMPLETE.json',result);w.write(public/'FREQUENCY_TILES_RESULT.json',result)
    lines=['# 전체 모델의 주파수 구간 처리: TRAIN6 초기 진단','',
        '원32.14M모델과전체638.72μs를 유지하고512개 주파수칸을 모두복원했다. '
        '128칸씩64칸 간격으로8개 구간을 처리하고, 정답 없이 원래 전체대역 출력에 순서를 맞춰 양의 가중합으로 병합했다. '
        '원래모델1회+구간8회씩총54회CPU추론이며 학습0회다.','',
        '| 처리 | 성분 수 | NMSE ↓ | 복소SI-SDR ↑ dB | 최약NMSE ↓ |',
        '|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['mode']}|{r['count']}|{r['nmse']:.6f}|{r['si_sdr']:.3f}|{r['weakest_nmse']:.6f}|")
    lines+=['','각개수2사례뿐이다. 구간별 정규화·문맥범위·경계·예측정렬이 함께 바뀌므로 주파수pooling만의 인과효과가 아니다. '
        '대역별 처리에 맞게 새로 학습한 모델도 아니며BSRNN/BS-RoFormer재현이 아니다. '
        'blend50은 원래출력과구간출력을0.5씩 섞는 사전고정조건이다. 보류자료나개발630개는열지않았다.',
        '', '[전체6행](FREQUENCY_TILES_RESULT.json)']
    (public/'FREQUENCY_TILES_KO.md').write_text('\n'.join(lines)+'\n')
    w.write(root/'STATE.json',dict(status='COMPLETE_CHECKED',cases=6,inferences=54,pid=os.getpid(),time=time.time()))
    print(dict(summary=summary,seconds=result['seconds']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
    try:run(a.run.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
