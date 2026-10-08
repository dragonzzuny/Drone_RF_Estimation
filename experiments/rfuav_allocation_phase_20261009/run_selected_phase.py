"""Queued identical four-phase inference on both completed allocation arms.

Model identities and inference are fixed while the parent is still training.
Parent checkpoint selection is unchanged; no phase-based epoch reselection.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/rfuav_phase_average_20261009'))
import run_phase as engine
from drone_rf.context_training_data import write_json
from drone_rf.data import sha256
from summarize_waveform_context import summarize,audit_completion,audit_validation

ARMS=('waveform_only','waveform_allocation')


def read(path):
    return json.loads(path.read_text())


def verify_files(plan):
    for name,digest in plan['source_sha256'].items():
        if sha256(ROOT/name)!=digest:
            raise RuntimeError(f'Frozen evaluation source changed: {name}')
    for name,digest in plan['data_sha256'].items():
        if sha256(Path(name))!=digest:
            raise RuntimeError(f'Frozen evaluation metadata changed: {name}')


def register(root,parent,prior_phase):
    base=read(parent/'PROTOCOL.json');prior=read(prior_phase/'PROTOCOL.json')
    if base['arms']!=list(ARMS) or base['epochs']!=5 or base['total_updates_per_arm']!=375:
        raise RuntimeError('Unexpected parent comparison')
    if not (prior_phase/'COMPLETE.json').exists():
        raise RuntimeError('Incumbent phase evaluation must be complete')
    sources=dict(prior['source_sha256'])
    for name,digest in base['source_sha256'].items():
        if name in sources and sources[name]!=digest:
            raise RuntimeError('Parent source versions conflict')
        sources[name]=digest
    sources.update({str(p.relative_to(ROOT)):sha256(p) for p in HERE.glob('*.py')})
    data=dict(base['data_sha256'])
    for path in [parent/'PROTOCOL.json',prior_phase/'PROTOCOL.json',
                 prior_phase/'stft_incumbent/FOUR_PHASE.json',prior_phase/'stft_incumbent/COMPLETE.json']:
        data[str(path)]=sha256(path)
    plan=dict(status='REGISTERED_BEFORE_PARENT_COMPLETION',parent_run=str(parent),prior_phase=str(prior_phase),
        source_sha256=sources,data_sha256=data,models_fixed_before_evaluation=list(ARMS),
        checkpoint_selection='parent SELECTED_005 minimum base-prediction count2/3 NMSE including e0; no reselection',
        angles_degrees=[0,90,180,270],forward_passes_baseline=1,forward_passes_average=4,model_updates=0,
        aggregation='equal complex mean; inverse rotation; prediction-only whole-window source matching; background fixed',
        validation_cases=630,same_dataset=True,same_native_band=True,native_center_offsets_preserved=False,
        inference_reference_access=False,heldout_iq_access=False,physical_aircraft_count=False,
        acceptance='candidate four-phase NMSE lower AND complex SI-SDR higher for both counts2/3 vs control four-phase; separately compare saved incumbent four-phase',
        scope='repeated development validation; no independent generalization or exact set-equivariance claim')
    verify_files(plan)
    path=root/'PROTOCOL.json'
    if path.exists():
        if read(path)!=plan:
            raise RuntimeError('Registered evaluation changed')
    else:
        for name in sources:
            target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        write_json(path,plan)
    return plan,sha256(path)


def compare(rows):
    control,candidate,incumbent=(rows[k] for k in ('waveform_only','waveform_allocation','incumbent'))
    identities=lambda r:[{k:x[k] for k in ('index','count','categories','pack_ids','reference_power','input_si_sdr')} for x in r['rows']]
    if identities(control)!=identities(candidate) or identities(control)!=identities(incumbent):
        raise RuntimeError('Final inference evaluation inputs differ')
    groups={name:{g['count']:g for g in saved['by_count']} for name,saved in rows.items()}
    beats=lambda a,b:all(groups[a][k]['mean_nmse']<groups[b][k]['mean_nmse'] and
        groups[a][k]['mean_si_sdr']>groups[b][k]['mean_si_sdr'] for k in (2,3))
    for saved in rows.values():
        audit_validation(saved)
    return dict(candidate_vs_matched_control=beats('waveform_allocation','waveform_only'),
        candidate_vs_incumbent=beats('waveform_allocation','incumbent'),
        control_vs_incumbent=beats('waveform_only','incumbent'),
        four_phase_by_count={name:saved['by_count'] for name,saved in rows.items()},
        all_rows_reaggregated=True,same_inputs=True,independent_test=False,
        criterion='strict directional improvement of both metrics at both counts2/3; not significance')


def main(root,parent,prior_phase):
    if (root/'COMPLETE.json').exists():
        raise RuntimeError('Evaluation already complete')
    plan,frozen=register(root,parent,prior_phase)
    parent_pid=read(parent/'STATE.json').get('pid')
    write_json(root/'STATE.json',dict(status='WAITING_FOR_ALLOCATION',pid=os.getpid(),time=time.time()))
    deadline=time.monotonic()+6*3600
    while not (parent/'COMPLETE.json').exists():
        if (parent/'FAILURE.json').exists() or time.monotonic()>deadline:
            raise RuntimeError('Parent failed or six-hour queue limit reached')
        time.sleep(15)
    with engine.LOCK.open('r') as lock:
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>deadline:
                    raise RuntimeError('GPU queue limit reached')
                time.sleep(15)
        # The completion marker precedes CUDA process teardown.
        for _ in range(10):
            if parent_pid is None or not Path('/proc',str(parent_pid)).exists():
                break
            time.sleep(1)
        time.sleep(2)
        engine.gpu_check()
        engine.torch.set_num_threads(2)
        engine.torch.backends.cudnn.benchmark=False
        engine.torch.backends.cuda.matmul.allow_tf32=False
        engine.torch.backends.cudnn.allow_tf32=False
        verify_files(plan)
        result=summarize(parent,ARMS)
        write_json(root/'PARENT_AUDIT.json',audit_completion(parent,result))
        checkpoints={a:dict(path=str(parent/a/'SELECTED_005.pt'),sha256=sha256(parent/a/'SELECTED_005.pt'),
            selected_epoch=result['selected_at_common_budget'][a]['epoch']) for a in ARMS}
        write_json(root/'CHECKPOINTS.json',checkpoints)
        write_json(root/'STATE.json',dict(status='GPU_EVALUATION_RUNNING',pid=os.getpid(),time=time.time()))
        data=engine.admitted_dataset(engine.PREPARATION,'validation_pack',1)
        results=[engine.evaluate(a,Path(checkpoints[a]['path']),lambda:engine.stft_build('unet_mean'),
            engine.stft_predict,parent/a,data,root,frozen) for a in ARMS]
        scored={a:read(root/a/'FOUR_PHASE.json') for a in ARMS}
        scored['incumbent']=read(prior_phase/'stft_incumbent/FOUR_PHASE.json')
        comparison=compare(scored);write_json(root/'COMPARISON.json',comparison)
        verify_files(plan)
        for a,checkpoint in checkpoints.items():
            if sha256(Path(checkpoint['path']))!=checkpoint['sha256']:
                raise RuntimeError('Selected checkpoint changed during evaluation')
        write_json(root/'COMPLETE.json',dict(status='COMPLETE',protocol_sha256=frozen,models=results,time=time.time()))
        write_json(root/'STATE.json',dict(status='COMPLETED',time=time.time()))
        print(json.dumps(dict(stage='FINAL_COMPARISON',**comparison)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--parent',required=True,type=Path)
    p.add_argument('--prior-phase',required=True,type=Path);args=p.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-2:]));os.nice(10)
    try:
        main(args.run,args.parent,args.prior_phase)
    except Exception:
        write_json(args.run/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        write_json(args.run/'STATE.json',dict(status='FAILED',time=time.time()))
        raise
