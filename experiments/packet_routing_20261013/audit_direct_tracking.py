"""Independent final-checkpoint CPU inference and direct I/Q score replay.

No references enter the embedding or assignment. Reference loading is deferred
until after CPU routing has been compared with the saved GPU choices.
"""
import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha
from tracking_cache import feature_input,observed_features
from spectral_routing_constrained import fit
from spectral_routing import PERMS
from numpy_waveform_score import score
from diagnose_count_routing import verify_metrics,summary

HERE=Path(__file__).resolve().parent
DEV=ROOT/'local/native_tracking_cache_20261013_v1'


def apply_orders(output,bounds,orders):
    result=np.empty_like(output)
    assert len(bounds)==len(orders)
    for (a,b),p in zip(bounds,orders):
        assert sorted(p)==[0,1,2]
        result[:,a:b]=output[[*p,3],a:b]
    assert np.array_equal(result[3],output[3])
    return result


def main(run,out):
    def state(status,**kw):write(out/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))
    out.mkdir(exist_ok=True)
    while True:
        current=read(run/'STATE.json')
        if current['status']=='COMPLETE_AUDITED':break
        if current['status']=='FAILED':raise RuntimeError('Training failed')
        state('WAITING_TRAINING',previous=current['status'],epoch=current.get('epoch'));time.sleep(10)
    protocol=read(run/'PROTOCOL.json')
    assert protocol['epochs']==50 and protocol['final_epoch_fixed']==50
    assert sha(run/'RESULT.json')==read(run/'AUDIT.json')['result_sha256']
    sources=dict(protocol['source_sha256'])
    assert sources[str(Path(__file__).resolve().relative_to(ROOT))]==sha(__file__)
    files={str(p):sha(p) for p in (run/'PROTOCOL.json',run/'RESULT.json',run/'AUDIT.json',
        run/'ACTUAL_050.pt',run/'C_050.json',DEV/'RESULT.json')}
    plan=dict(registered=time.time(),source_sha256=sources,input_sha256=files,cases=24,
        no_training_updates=True,references_only_after_routing=True,direct_waveform_metrics=True)
    assert not (out/'PLAN.json').exists();write(out/'PLAN.json',plan)
    for name,h in sources.items():
        assert sha(ROOT/name)==h
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest);assert sha(dest)==h
    import torch
    from direct_tracking_model import DirectTracking as TrackingEmbedding,decode
    from sequence_data import Sequences,ROLES
    from native_full_context import components
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
    from native_data import NativeMixtures
    net=TrackingEmbedding().eval()
    saved=torch.load(run/'ACTUAL_050.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['epoch']==50 and saved['updates']==600
    assert saved['protocol_sha256']==sha(run/'PROTOCOL.json')
    assert sha(run/'ACTUAL_050.pt')==read(run/'EPOCH_050.json')['checkpoint_sha256']
    net.load_state_dict(saved['model']);del saved
    development=read(DEV/'RESULT.json');groups={}
    for group,name in (('C24','C_050.json'),):
        original=read(run/name);expected={r['index']:r for r in original['rows']}
        receipts={r['index']:r for r in original['routing_receipts']};rows=[]
        data=Sequences(ROLES[1]) if group=='B24' else NativeMixtures(
            Path(read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')['preparation']),'validation_pack',1)
        for j,item in enumerate(development['groups'][group]):
            index=item['index'];state('CPU_DIRECT_AUDIT',group=group,case=j+1,total=24)
            observed,powers,bounds=feature_input(item)
            path=Path(item['metadata']['parent_output']);assert sha(path)==item['metadata']['parent_output_sha256']
            output=np.load(path,mmap_mode='r',allow_pickle=False)
            rebuilt=observed_features(output)
            with np.load(item['features'],allow_pickle=False) as cache:
                for key,value in rebuilt.items():np.testing.assert_array_equal(value,cache[key])
            with torch.no_grad():
                logits=net(torch.from_numpy(observed)[None])[0]
            decoded,objective=decode(logits);orders=decoded.tolist()
            wanted=receipts[index]['orders']
            disagreement=np.flatnonzero(np.any(np.asarray(orders)!=np.asarray(wanted),axis=1)).tolist()
            if disagreement:
                write(out/f'DIFFERENCE_{group}_{index:03d}.json',dict(blocks=disagreement,cpu_orders=orders,gpu_orders=wanted))
            assert not disagreement,('CPU/GPU routing differs',group,index,disagreement)
            routed=apply_orders(output,bounds,orders)
            # Only now is source-reference I/Q loaded for direct scoring.
            refs=data.components(index,True) if group=='B24' else components(data,index)
            metrics=score(routed,refs,refs.sum(0),item['count'],item['predicted_count'])
            verify_metrics(metrics,expected[index]);verify_metrics(score(output,refs,refs.sum(0),item['count'],
                item['predicted_count']),item['baseline_metrics'])
            row=dict(index=index,**metrics);rows.append(row)
            write(out/f'CASE_{group}_{index:03d}.json',dict(status='PASS',metrics=row,
                features_exact=True,cpu_gpu_orders_exact=True,background_unchanged=True,
                routing_confidence=objective))
            del observed,output,rebuilt,logits,routed,refs;gc.collect()
        groups[group]={str(k):summary([r for r in rows if r['count']==k]) for k in (1,2,3)}
        for k in (1,2,3):
            for metric in ('mean_nmse','mean_si_sdr','weakest_nmse','original_count_accuracy'):
                np.testing.assert_allclose(groups[group][str(k)][metric],original['by_count'][str(k)][metric],rtol=1e-6,atol=1e-7)
        del data;gc.collect()
    assert all(sha(ROOT/p)==h for p,h in sources.items())
    assert all(sha(p)==h for p,h in files.items())
    result=dict(status='PASS',epoch=50,cases=24,groups=groups,plan_sha256=sha(out/'PLAN.json'),
        direct_iq_metrics_reproduced=True,observed_features_reproduced_exactly=True,
        cpu_gpu_orders_reproduced_exactly=True,background_unchanged=True,
        references_not_used_in_forward_or_routing=True,training_updates=0,
        parent_frozen_checkpoint_sha256=read(Path(protocol['teacher'])/'PLAN.json')['input_sha256'][read(Path(protocol['teacher'])/'PLAN.json')['checkpoint']])
    write(out/'RESULT.json',result);write(out/'AUDIT.json',dict(status='PASS',result_sha256=sha(out/'RESULT.json')))
    write(ROOT/'reports/2026-10-13/DIRECT_TRACKING_DIRECT_AUDIT.json',result)
    state('COMPLETE_AUDITED',cases=24)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    try:main(a.run,a.output)
    except BaseException:
        a.output.mkdir(exist_ok=True);write(a.output/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write(a.output/'STATE.json',dict(status='FAILED',time=time.time(),pid=os.getpid()));raise
