"""Four training-case phase probe on the existing full STFT U-Net incumbent."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from audit_decoder_context import align_predictions

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/rfuav_dense_gated_20261008'))
from models import build,predict
from study import admitted_dataset,batch
from drone_rf.waveform import waveform_metrics

SOURCE=Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/rfuav_architecture_20261008')
CHECKPOINT=SOURCE/'dense_gpu_run/unet_mean/SELECTED_005.pt'
EXPECTED='86470af0de460d6b71072e08fb19ec77bfb9d3545a41e7da728fad084870a356'


@torch.no_grad()
def run():
    torch.set_num_threads(2)
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()!=EXPECTED:
        raise ValueError('Previously audited incumbent checkpoint changed')
    net=build('unet_mean').eval()
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    net.load_state_dict(saved['model'])
    data=admitted_dataset(SOURCE/'dense_preparation','train_pack',1)
    indices=[int(i) for k in (2,3) for i in np.flatnonzero(data.rows['count']==k)[:2]]
    rows=[]
    for index in indices:
        item=batch(data[index],'cpu')
        full,_=predict(net,item)
        def score(estimate):
            m=waveform_metrics(estimate,item['references'],item['active'],item['mixture'])
            active=item['active'][0]
            return dict(nmse=m['nmse'][0][active].tolist(),si_sdr=m['si_sdr'][0][active].tolist(),
                output_change_relative_energy=float((estimate-full).abs().square().sum()/full.abs().square().sum()),
                sum_relative_error=float(m['sum_relative_error'][0]))
        row=dict(index=index,count=int(item['construction_count'][0]),modes={'full':score(full)})
        views=[full]
        for degree,factor in ((90,1j),(180,-1+0j),(270,-1j)):
            changed=dict(item,mixture=item['mixture']*factor)
            estimate,_=predict(net,changed)
            estimate,order=align_predictions(estimate/factor,full)
            row['modes'][f'phase_{degree}']=dict(score(estimate),prediction_only_order=order)
            views.append(estimate)
        row['modes']['four_phase_average']=score(torch.stack(views).mean(0))
        rows.append(row)
    return dict(status='TRAIN_ONLY_CPU_COMPLETE',model='existing full STFT U-Net',
        checkpoint=str(CHECKPOINT),checkpoint_sha256=EXPECTED,selected_epoch=saved['best']['epoch'],
        rows=rows,model_updates=0,heldout_or_validation_iq_read=False,inference_labels_used=False,
        interpretation='train-only candidate diagnostic; not generalization evidence')


if __name__=='__main__':
    os.nice(10);cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[-4:-2] or cpus))
    start=time.time();result=run();result['seconds']=time.time()-start
    destination=ROOT/'local/decoder_fusion_20261009_v1/monitor/STFT_PHASE_TRAIN_PROBE.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
