"""Fixed 50% prediction-space correction; no gain fit or optimizer updates."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/integrated_20261011'))
import model as integrated
worker = integrated.worker
w = worker.watch
validation = integrated.module('half_validation', 'experiments/recursive_20261010/validation.py')
phase = integrated.module('half_alignment', 'experiments/rfuav_phase_average_20261009/phase_core.py')
audit = integrated.module('half_groups', 'experiments/integrated_audit_20261011/audit.py')
from drone_rf.waveform import waveform_metrics

PUBLIC = ROOT/'reports/2026-10-11'
PREVIOUS = ROOT/'local/integrated_20261011_v2'


def state(root, status, **values):
    w.write(root/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**values))


@torch.no_grad()
def predictions(models, item):
    # Ground truth, actual count, class and recording identity are not read here.
    inputs = {key:item[key] for key in ('mixture','context_features','crop_start')}
    base, counts = worker.predict(models[0],inputs)
    changed, _ = worker.predict(models[1],inputs)
    aligned, order = phase.align_predictions(changed,base)
    return (base+aligned)*.5, counts, base, changed, order


def register(root):
    assert not (root/'PROTOCOL.json').exists()
    old = w.read(PREVIOUS/'PROTOCOL.json')
    stop = w.read(PREVIOUS/'COMPLETE.json')
    event = w.read(PREVIOUS/'EPOCH_001.json')
    assert stop['status']=='STOPPED_AT_USER_EPOCH_REVIEW' and stop['epochs']==1
    assert w.digest(PREVIOUS/'LAST.pt')==event['last_sha256']
    sources = dict(old['source_sha256'])
    for relative in ('experiments/half_correction_20261011/run.py',
        'experiments/integrated_audit_20261011/audit.py',
        'experiments/rfuav_phase_average_20261009/phase_core.py'):
        sources[relative] = w.digest(ROOT/relative)
    for relative,digest in sources.items():
        assert w.digest(ROOT/relative)==digest,relative
        dest=root/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,dest)
    pins=dict(old['pinned_files'])
    for path in (PREVIOUS/'LAST.pt',PREVIOUS/'VALIDATION_001.json',PREVIOUS/'COMPLETE.json',
        PREVIOUS/'EPOCH_001.json',PUBLIC/'HALF_CORRECTION_PLAN_KO.md',
        Path(old['baseline']).with_name('FOUR_PHASE.json')):
        pins[str(path)]=w.digest(path)
    protocol=dict(status='REGISTERED_FIXED_HALF_CORRECTION',source_sha256=sources,pinned_files=pins,
        parent_checkpoint=old['parent_checkpoint'],integrated_checkpoint=str(PREVIOUS/'LAST.pt'),
        parent_baseline=old['baseline'],integrated_baseline=str(PREVIOUS/'VALIDATION_001.json'),
        preparation=old['preparation'],coefficient=.5,validation_cases=630,forward_passes=2,
        count_output='parent logits, unchanged',optimizer_updates=0,inference_reference_access=False,
        alignment='prediction-only complex squared error over whole window; three slots, background fixed',
        criterion='count2/3 NMSE lower and complex SI-SDR higher and weakest NMSE nonworse versus parent single pass',
        reproduction_tolerances=dict(reference_power='exact',nmse='rtol1e-5 atol1e-7',
            si_sdr='rtol1e-5 plus absolute1e-5 dB; preserves previous relative bound and handles near0dB'),
        heldout_read=False,independent_test=False,registered_at=time.time())
    w.write(root/'PROTOCOL.json',protocol)
    return protocol


def verify(root,p):
    for relative,digest in p['source_sha256'].items():
        assert w.digest(ROOT/relative)==w.digest(root/'source_snapshot'/relative)==digest,relative
    for path,digest in p['pinned_files'].items():assert w.digest(Path(path))==digest,path


def algebra_check():
    torch.manual_seed(0)
    anchor=torch.complex(torch.randn(1,4,19),torch.randn(1,4,19))
    permuted=anchor[:,[2,0,1,3]]
    aligned,order=phase.align_predictions(permuted,anchor)
    assert order==[1,2,0] and torch.equal(aligned,anchor)
    assert torch.equal((anchor+aligned)*.5,anchor)
    return dict(status='PASS',known_permutation_restored=True,identity_average_exact=True)


def run(root):
    root.mkdir(parents=True,exist_ok=True)
    p=register(root);check=algebra_check();verify(root,p)
    w.write(root/'PREFLIGHT.json',check)
    state(root,'WAITING_GPU_LOCK')
    with worker.fit.base.LOCK.open('r') as gpu:
        fcntl.flock(gpu,fcntl.LOCK_EX)
        assert torch.cuda.is_available()
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        models=[]
        for name in ('parent','integrated'):
            net=worker.make_model('retained_unet')
            if name=='integrated':integrated.augment(net)
            saved=torch.load(p[name+'_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
            net.load_state_dict(saved['model']);del saved
            net.requires_grad_(False);net.cuda().eval();models.append(net)
        models=torch.nn.ModuleList(models)
        data=worker.NativeMixtures(p['preparation'],'validation_pack',1)
        originals=[w.read(Path(p[name+'_baseline'])) for name in ('parent','integrated')]
        counter=0;orders=[];started=time.time();reproduction=[]
        def predict_and_verify(models,item):
            nonlocal counter
            mean,logits,base,changed,order=predictions(models,item)
            # Scoring is outside predictions; references never choose the blend.
            for name,output,original in zip(('parent','integrated'),(base,changed),originals):
                metric=waveform_metrics(output,item['references'],item['active'],item['mixture'])
                active=item['active'][0];old=original['rows'][counter]
                for key in ('reference_power','nmse','si_sdr'):
                    actual=metric[key][0][active].cpu().numpy()
                    absolute=float(np.max(np.abs(actual-np.array(old[key]))))
                    reproduction.append(dict(index=counter,model=name,metric=key,max_abs_error=absolute))
                    rtol,atol=((0,0) if key=='reference_power' else (1e-5,1e-5) if key=='si_sdr' else (1e-5,1e-7))
                    np.testing.assert_allclose(actual,old[key],rtol=rtol,atol=atol,
                        err_msg=f'{name} case{counter} metric{key}')
            orders.append(order);counter+=1
            if counter%25==0:state(root,'EVALUATING',cases=counter,total=630,seconds=time.time()-started)
            return mean,logits
        state(root,'EVALUATING',cases=0,total=630)
        result=validation.validate(models,data,root/'HALF.json',1,predict_and_verify,worker)
        assert counter==630
        for row,base,changed in zip(result['rows'],originals[0]['rows'],originals[1]['rows']):
            for key in ('index','count','categories','pack_ids','reference_power'):
                assert row[key]==base[key]==changed[key],(row['index'],key)
            assert row['predicted_count']==base['predicted_count']
        checks=[]
        for a,b in zip(result['by_count'][1:],originals[0]['by_count'][1:]):
            checks.append(dict(count=a['count'],nmse=a['mean_nmse']<b['mean_nmse'],
                si=a['mean_si_sdr']>b['mean_si_sdr'],weak=a['weakest_nmse']<=b['weakest_nmse']))
        accepted=all(c[k] for c in checks for k in ('nmse','si','weak'))
        conditions={name:audit.grouped(value['rows']) for name,value in
            [('parent',originals[0]),('integrated',originals[1]),('half',result)]}
        verify(root,p)
        receipt=dict(status='COMPLETED',directional_criterion=accepted,checks=checks,
            groups=result['by_count'],parent=originals[0]['by_count'],integrated=originals[1]['by_count'],
            four_phase=w.read(Path(p['parent_baseline']).with_name('FOUR_PHASE.json'))['by_count'],
            combinations=conditions,baseline_and_integrated_reproduced_all630=True,
            reproduction_max_abs={name:{key:max(r['max_abs_error'] for r in reproduction
                if r['model']==name and r['metric']==key) for key in ('reference_power','nmse','si_sdr')}
                for name in ('parent','integrated')},
            optimizer_updates=0,forward_passes=2,coefficient=.5,heldout_read=False,
            independent_test=False,seconds=time.time()-started,time=time.time(),
            protocol_sha256=w.digest(root/'PROTOCOL.json'),result_sha256=w.digest(root/'HALF.json'))
        # Preserve exact prediction-only correspondence, not just a histogram.
        w.write(root/'ORDERS.json',orders)
        receipt['permutation_histogram']={str(order):sum(tuple(o)==order for o in orders)
            for order in sorted(set(map(tuple,orders)))}
        w.write(root/'COMPLETE.json',receipt);w.write(PUBLIC/'HALF_CORRECTION_RESULT.json',receipt)
        state(root,'COMPLETED',cases=630,directional_criterion=accepted)
        print(receipt,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();root=args.run.resolve()
    try:run(root)
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        state(root,'FAILED');raise
