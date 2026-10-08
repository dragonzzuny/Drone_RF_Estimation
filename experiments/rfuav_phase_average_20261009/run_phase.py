"""Queued full development check of fixed four-phase averaging on three models.

Waits for the frozen late-fusion comparison and obtains the shared GPU lock.
Checkpoints are selected by the PREVIOUS base-prediction rules, never by phase
averaging scores. No training, new recording admission or heldout access.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/rfuav_decoder_fusion_20261009'))
from late_models import build as late_build,predict as late_predict
from run_late import PREPARATION,LOCK,gpu_check
from long_data import admitted_dataset,batch
from models import build as stft_build,predict as stft_predict
from drone_rf.data import sha256
from drone_rf.waveform import waveform_metrics
from drone_rf.context_training_data import write_json
sys.path.insert(0,str(ROOT/'scripts'))
from summarize_waveform_context import audit_validation
from phase_core import phase_predictions
from study import finite_values, value_status

STFT=Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/rfuav_architecture_20261008/dense_gpu_run/unet_mean')
STFT_SHA='86470af0de460d6b71072e08fb19ec77bfb9d3545a41e7da728fad084870a356'


def specifications(parent):
    return [('stft_incumbent',STFT/'SELECTED_005.pt',lambda:stft_build('unet_mean'),stft_predict,STFT),
        *[(f'late_{a}',parent/a/'SELECTED_005.pt',lambda a=a:late_build(a),late_predict,parent/a)
          for a in ('local_only','local_global')]]


def freeze(root,parent):
    if sha256(STFT/'SELECTED_005.pt')!=STFT_SHA:
        raise RuntimeError('STFT incumbent changed')
    base=json.loads((parent/'PROTOCOL.json').read_text())
    sources=dict(base['source_sha256'])
    for p in [*HERE.glob('*.py'),ROOT/'scripts/summarize_waveform_context.py']:
        sources[str(p.relative_to(ROOT))]=sha256(p)
    for p,h in base['source_sha256'].items():
        if sha256(ROOT/p)!=h:
            raise RuntimeError('Parent experiment source changed')
    for p,h in base['data_sha256'].items():
        if sha256(Path(p))!=h:
            raise RuntimeError('Parent data/checkpoint changed')
    checkpoints={name:dict(path=str(p),sha256=sha256(p)) for name,p,*_ in specifications(parent)}
    plan=dict(status='FROZEN_FOUR_PHASE_DEVELOPMENT_EVALUATION',parent_run=str(parent),
        source_sha256=sources,data_sha256=base['data_sha256'],checkpoints=checkpoints,
        angles_degrees=[0,90,180,270],aggregation='equal mean after exact inverse phase and prediction-only whole-window slot alignment',
        checkpoint_selection='existing base NMSE rule in each completed experiment, not phase scores',
        models_fixed_before_evaluation=['stft_incumbent','late_local_only','late_local_global'],
        forward_passes_baseline=1,forward_passes_average=4,model_updates=0,
        validation_cases=630,same_dataset=True,same_native_band=True,native_center_offsets_preserved=False,
        inference_reference_access=False,heldout_iq_access=False,physical_aircraft_count=False,
        acceptance='NMSE lower and complex SI-SDR higher for each count2/3, no posthoc exclusions',
        status_scope='development evaluation of a data-informed candidate; no independent test or claim of exact set equivariance')
    path=root/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=plan:
            raise RuntimeError('Frozen phase evaluation changed')
    else:
        for name in sources:
            target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        write_json(path,plan)
    return sha256(path)


def row_metrics(estimate,logits,item,source_row,clips,index,details):
    m=waveform_metrics(estimate,item['references'],item['active'],item['mixture'])
    active=item['active'][0];count=int(item['construction_count'][0]);power=m['reference_power'][0][active]
    si=m['si_sdr'][0][active];gain=si-m['input_si_sdr'][0][active]
    row=dict(index=index,count=count,categories=[c['category'] for c in clips],pack_ids=[c['pack_id'] for c in clips],
        nominal_levels_db=source_row['levels'][:count].tolist(),reference_power=power.tolist(),
        nmse=m['nmse'][0][active].tolist(),si_sdr=si.tolist(),si_status=['finite']*count,
        input_si_sdr=finite_values(m['input_si_sdr'][0][active]),si_sdr_gain=finite_values(gain),
        input_si_status=value_status(m['input_si_sdr'][0][active]),gain_status=value_status(gain),
        weakest_index=int(power.argmin()),
        assignment=m['assignment'][0].tolist(),predicted_count=int(logits.argmax(-1)[0])+1,
        inactive_leak=float(m['inactive_leak'][0].sum()),background_nmse=float(m['background_nmse'][0]),
        sum_relative_error=float(m['sum_relative_error'][0]),phase_details=details)
    if not all(math.isfinite(v) for key in ('nmse','si_sdr') for v in row[key]):
        raise RuntimeError('Undefined waveform score; never discard silently')
    # A one-source mixture can equal its reference: its input SI-SDR is +inf,
    # and gain is then -inf. Preserve null+status, as in the original evaluator;
    # absolute SI-SDR and NMSE remain primary and no case is dropped.
    if count>1 and any(v is None for v in row['si_sdr_gain']):
        raise RuntimeError('Undefined multi-source SI-SDR improvement')
    return row


def aggregate(rows,epoch):
    groups=[]
    for count in (1,2,3):
        subset=[r for r in rows if r['count']==count]
        avg=lambda key:float(np.mean([v for r in subset for v in r[key]]))
        groups.append(dict(count=count,cases=len(subset),mean_nmse=avg('nmse'),mean_si_sdr=avg('si_sdr'),
            mean_si_sdr_gain=avg('si_sdr_gain') if count>1 else None,nonfinite_si_sdr=0,
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in subset])),
            construction_count_accuracy=float(np.mean([r['predicted_count']==count for r in subset]))))
    return dict(epoch=epoch,by_count=groups,rows=rows,selection_nmse=float(np.mean([g['mean_nmse'] for g in groups if g['count']>1])),
        independent_test=False,physical_aircraft_count=False,whole_record_tracking=False)


@torch.no_grad()
def evaluate(name,checkpoint,builder,predict,original_folder,data,root,frozen):
    folder=root/name;folder.mkdir(exist_ok=True)
    if (folder/'COMPLETE.json').exists():
        raise RuntimeError('Refuse silent reuse; report saved results explicitly')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    epoch=saved['best']['epoch'];net=builder().cuda().eval();net.load_state_dict(saved['model']);del saved
    original=json.loads((original_folder/f'VALIDATION_{epoch:03d}.json').read_text())['rows']
    rows={mode:[] for mode in ('baseline','four_phase')};start=time.time()
    for index in range(len(data)):
        item=batch(data[index],'cuda');source=data.rows[index]
        clips=[data.library.clips[int(i)] for i in source['indices'][:int(source['count'])]]
        full,mean,logits,details=phase_predictions(net,predict,item)
        for mode,estimate in (('baseline',full),('four_phase',mean)):
            row=row_metrics(estimate,logits,item,source,clips,index,details)
            if mode=='baseline':
                prior=original[index]
                for key in ('index','count','categories','pack_ids','nominal_levels_db'):
                    if row[key]!=prior[key]:
                        raise RuntimeError('Previously evaluated case identity differs')
                for key in ('reference_power','nmse','si_sdr'):
                    if not np.allclose(row[key],prior[key],rtol=1e-5,atol=1e-7):
                        raise RuntimeError(f'Baseline reproduction differs: {name} {index} {key}')
            rows[mode].append(row)
        if index%50==0:
            progress=dict(stage='PHASE_EVALUATION',model=name,cases=index+1,total=len(data),seconds=time.time()-start,time=time.time())
            write_json(root/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    result={}
    for mode in rows:
        grouped=aggregate(rows[mode],epoch)
        diagnostic=audit_validation(grouped)
        write_json(folder/f'{mode.upper()}.json',grouped)
        result[mode]=dict(by_count=grouped['by_count'],diagnostics=diagnostic)
    a,b=({g['count']:g for g in result[mode]['by_count']} for mode in ('baseline','four_phase'))
    result.update(model=name,selected_epoch=epoch,protocol_sha256=frozen,seconds=time.time()-start,
        baseline_reproduced_all_630=True,
        directional_criterion=all(b[k]['mean_nmse']<a[k]['mean_nmse'] and b[k]['mean_si_sdr']>a[k]['mean_si_sdr'] for k in (2,3)))
    write_json(folder/'COMPLETE.json',result);print(json.dumps(dict(stage='MODEL_COMPLETE',**result)),flush=True)
    del net;gc.collect();torch.cuda.empty_cache()
    return result


def main(root,parent):
    write_json(root/'STATE.json',dict(status='WAITING_FOR_LATE_FUSION',parent=str(parent),pid=os.getpid(),time=time.time()))
    while not (parent/'COMPLETE.json').exists():
        if (parent/'FAILURE.json').exists():
            raise RuntimeError('Parent training failed; stop dependent evaluation')
        time.sleep(15)
    with LOCK.open('r') as lock:
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                time.sleep(15)
        # COMPLETE is written immediately before the parent releases its lock.
        # CUDA teardown can lag that release briefly; wait only for that PID.
        state=json.loads((parent/'RUN_STATE.json').read_text())
        progress=json.loads((parent/'PROGRESS.json').read_text())
        parent_pid=state.get('pid',progress.get('pid'))
        for _ in range(10):
            if parent_pid is None or not Path('/proc',str(parent_pid)).exists():
                break
            time.sleep(1)
        gpu_check()
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        frozen=freeze(root,parent)
        write_json(root/'STATE.json',dict(status='GPU_EVALUATION_RUNNING',pid=os.getpid(),time=time.time()))
        data=admitted_dataset(PREPARATION,'validation_pack',1)
        results=[evaluate(*spec,data,root,frozen) for spec in specifications(parent)]
        if freeze(root,parent)!=frozen:
            raise RuntimeError('Evaluation source/data/checkpoint changed during execution')
        write_json(root/'COMPLETE.json',dict(status='COMPLETE',protocol_sha256=frozen,models=results,time=time.time()))
        write_json(root/'STATE.json',dict(status='COMPLETED',time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    args=parser.parse_args();args.run.mkdir(parents=True,exist_ok=True)
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[-2:]));os.nice(10)
    try:
        main(args.run,args.parent)
    except Exception:
        write_json(args.run/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
