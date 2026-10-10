"""All fixed TRAIN48 initial successive outputs, no optimization or DEV."""
import argparse
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback
import numpy as np
import torch
from successive import predict

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/window_overlap_20261010'))
import diagnose as core
from drone_rf.waveform import waveform_metrics
w=core.w

def run(root):
    root.mkdir(parents=True,exist_ok=True);assert not (root/'PROTOCOL.json').exists()
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    fixed=[r for r in w.read(ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json')['rows'] if r['model']=='parent/e0']
    assert len(fixed)==48
    sources=dict(old['source_sha256'])
    for path in (Path(__file__),Path(__file__).with_name('successive.py'),ROOT/'experiments/window_overlap_20261010/diagnose.py'):
        sources[str(path.relative_to(ROOT))]=w.digest(path)
    p=dict(status='REGISTERED_CPU_SUCCESSIVE_INITIAL48',indices=[r['index'] for r in fixed],sources=sources,
        parent_sha256=old['parent_checkpoint_sha256'],preparation_sha256=old['preparation_sha256'],
        inferences=96,updates=0,heldout_read=False,gpu_use=False,
        selection='All original fixed TRAIN48; include1/2/3 and every failed case',registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',p)
    torch.set_num_threads(2);data=core.base.worker.NativeMixtures(old['preparation'],'train_pack',1)
    net=core.base.worker.make_model('retained_unet',Path(old['parent_checkpoint'])).eval()
    rows=[];started=time.time()
    with torch.inference_mode():
        for number,prior in enumerate(fixed):
            item=core.batch(data[prior['index']]);output,logits,trace=predict(net,item,core.base.worker.predict)
            parent,_=core.metrics(trace['first_predictions'],item)
            candidate,_=core.metrics(output,item)
            for key in ('nmse','si_sdr','reference_power'):
                assert np.max(np.abs(np.array(parent[key])-np.array(prior[key])))<2e-5
            for estimates,metric in ((trace['first_predictions'],parent),(output,candidate)):
                measured=waveform_metrics(estimates,item['references'],item['active'],item['mixture'])
                metric['inactive_leak']=float(measured['inactive_leak'][0].sum())
                metric['background_nmse']=float(measured['background_nmse'][0])
            rows.append(dict(index=prior['index'],count=prior['count'],categories=prior['categories'],
                predicted_count=int(logits.argmax(-1))+1,first_slot=int(trace['first_slot']),second_slot=int(trace['second_slot']),
                parent=parent,successive=candidate))
            w.write(root/'STATE.json',dict(status='CPU_SUCCESSIVE_INITIAL',cases=number+1,total_cases=48,
                inferences=2*(number+1),total_inferences=96,pid=os.getpid(),time=time.time()))
    summaries=[]
    for mode in ('parent','successive'):
        for count in (1,2,3):
            selected=[r[mode] for r in rows if r['count']==count]
            summaries.append(dict(mode=mode,count=count,cases=len(selected),
                nmse=statistics.mean(v for r in selected for v in r['nmse']),
                si_sdr=statistics.mean(v for r in selected for v in r['si_sdr']),
                weakest_nmse=statistics.mean(r['nmse'][r['weakest_index']] for r in selected),
                inactive_leak=statistics.mean(r['inactive_leak'] for r in selected)))
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    result=dict(status='COMPLETE_CHECKED',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,summary=summaries,
        seconds=time.time()-started,optimizer_steps=0,heldout_read=False,gpu_use=False,
        limitation='Initial architecture only on reused TRAIN48; not post-training DEV or new-recording generalization')
    w.write(root/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-10/SUCCESSIVE_INITIAL48_RESULT.json',result)
    lines=['# 두 단계 전체 출력의 초기 TRAIN48 진단','',
        '부모를 수정하지 않고 같은 고정TRAIN48 모두를 두 단계 구조에 적용했다. '
        '정답·실제 개수 없이 세 파형을 생성했고, 원본 부모의 출력은 첫 단계에서 그대로 함께 검산했다. '
        '96회 CPU 추론, optimizer0회. 이 값은 새 구조를 학습한 DEV 성과가 아니다.','',
        '| 구조 | 성분 수 | 혼합 수 | NMSE ↓ | 복소SI-SDR ↑ dB | 최약NMSE ↓ | 비활성 출력/혼합 에너지 ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"|{s['mode']}|{s['count']}|{s['cases']}|{s['nmse']:.6f}|{s['si_sdr']:.3f}|{s['weakest_nmse']:.6f}|{s['inactive_leak']:.6f}|")
    lines+=['','두 번째 단계는 원래 혼합의 긴 문맥을 유지한다. 세 번째 출력이 남은 잔차를 모두 받는 구조이며, '
        '한·두 성분에서는 비활성 슬롯을 포함한다. 유리한 사례만 골라 보고하지 않는다. '
        '기존의 남은 두 성분만 평가한 제거 진단과 전체3출력의 지표는 서로 다른 비교다.',
        '', '[모든48행](SUCCESSIVE_INITIAL48_RESULT.json)']
    (ROOT/'reports/2026-10-10/SUCCESSIVE_INITIAL48_KO.md').write_text('\n'.join(lines)+'\n')
    w.write(root/'STATE.json',dict(status='COMPLETE_CHECKED',cases=48,inferences=96,pid=os.getpid(),time=time.time()))
    print(dict(summary=summaries,seconds=result['seconds']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
    try:run(a.run.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
