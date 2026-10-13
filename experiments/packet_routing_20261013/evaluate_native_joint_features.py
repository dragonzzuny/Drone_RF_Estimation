"""One fixed PSD/learned-feature blend; endpoints and direct I/Q are audited."""
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha
from tracking_cache import feature_input
from spectral_routing import profiles,PERMS
from spectral_routing_joint_features import fit
from routing_statistics import score as cached_score
from diagnose_count_routing import verify_metrics,summary
from audit_learned_tracking import apply_orders
from numpy_waveform_score import score

OUT=ROOT/'local/native_joint_features_20261013_v1'
RUN=ROOT/'local/native_tracking_20261013_v1'
CACHE=ROOT/'local/native_tracking_cache_20261013_v1'
BASE=ROOT/'local/incumbent_tracking_c24_20261013_v1'


def state(status,**kw):
    write(OUT/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**kw))


def main():
    assert read(ROOT/'local/native_tracking_direct_audit_20261013_v1/STATE.json')['status']=='COMPLETE_AUDITED'
    protocol=read(RUN/'PROTOCOL.json');sources=dict(protocol['source_sha256'])
    for path in (Path(__file__),Path(__file__).with_name('spectral_routing_joint_features.py'),
                 ROOT/'reports/2026-10-13/NATIVE_JOINT_FEATURES_PLAN_KO.md'):
        sources[str(path.relative_to(ROOT))]=sha(path)
    files={str(p):sha(p) for p in (RUN/'ACTUAL_050.pt',RUN/'RESULT.json',RUN/'C_050.json',
        CACHE/'RESULT.json',BASE/'RESULT.json')}
    assert not (OUT/'PLAN.json').exists()
    write(OUT/'PLAN.json',dict(registered=time.time(),source_sha256=sources,input_sha256=files,
        cases=24,learned_feature_energy_fraction=.5,optimizer_updates=0,heldout_access=False,
        true_count_in_routing=False,reference_only_in_scoring=True,scope='One additional fixed development candidate'))
    for name,h in sources.items():
        assert sha(ROOT/name)==h
        path=OUT/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,path);assert sha(path)==h
    import torch
    from learned_tracking_model import TrackingEmbedding
    from native_full_context import components
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
    from native_data import NativeMixtures
    data=NativeMixtures(Path(read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')['preparation']),'validation_pack',1)
    net=TrackingEmbedding().eval().requires_grad_(False)
    saved=torch.load(RUN/'ACTUAL_050.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['epoch']==50 and saved['updates']==600
    net.load_state_dict(saved['model']);del saved
    base={r['index']:r for r in read(BASE/'RESULT.json')['cases']}
    prior=read(RUN/'C_050.json');learned={r['index']:r for r in prior['rows']}
    learned_orders={r['index']:r['orders'] for r in prior['routing_receipts']}
    rows=[]
    for j,item in enumerate(read(CACHE/'RESULT.json')['groups']['C24']):
        i=item['index'];state('CPU_JOINT_FEATURE_EVALUATION',case=j+1,total=24)
        observed,powers,bounds=feature_input(item)
        path=Path(item['metadata']['parent_output']);assert sha(path)==item['metadata']['parent_output_sha256']
        output=np.load(path,mmap_mode='r',allow_pickle=False)
        psd,rebuilt_power,rebuilt_bounds=profiles(output)
        np.testing.assert_array_equal(powers,rebuilt_power);np.testing.assert_array_equal(bounds,rebuilt_bounds)
        with torch.no_grad():embedding=net(torch.from_numpy(observed)[None])[0].numpy().astype(np.float64)
        orders={}
        for name,alpha in [('psd',0.),('learned',1.),('joint',.5)]:
            vector=np.concatenate([np.sqrt(1-alpha)*psd,np.sqrt(alpha)*embedding],axis=-1)
            _,choice,receipt=fit(vector,powers,item['predicted_count'])
            orders[name]=np.array([PERMS[int(q)] for q in choice])
        np.testing.assert_array_equal(orders['psd'],base[i]['routing']['psd']['orders'])
        np.testing.assert_array_equal(orders['learned'],learned_orders[i])
        # All candidate routing is fixed before any reference I/Q is loaded.
        assert sha(item['scoring'])==item['scoring_sha256']
        with np.load(item['scoring'],allow_pickle=False) as z:stats={k:z[k].copy() for k in z.files}
        for name,wanted in [('psd',base[i]['metrics']['psd']),('learned',learned[i])]:
            verify_metrics(cached_score(stats,orders[name],item['count'],item['predicted_count']),wanted)
        refs=components(data,i);routed=apply_orders(output,bounds,orders['joint'])
        direct=score(routed,refs,refs.sum(0),item['count'],item['predicted_count'])
        verify_metrics(direct,cached_score(stats,orders['joint'],item['count'],item['predicted_count']))
        row=dict(index=i,count=item['count'],categories=item['metadata']['categories'],
            nominal_levels_db=item['metadata']['nominal_levels_db'],metrics=direct,
            nmse_change_vs_psd=float(np.mean(direct['nmse'])-np.mean(base[i]['metrics']['psd']['nmse'])),
            orders=orders['joint'].tolist(),endpoints_orders_exact=True,direct_iq_verified=True)
        write(OUT/f'CASE_{i:03d}.json',row);rows.append(row)
        del output,refs,routed,psd,embedding,stats;gc.collect()
    assert all(sha(ROOT/p)==h for p,h in sources.items()) and all(sha(p)==h for p,h in files.items())
    groups={str(k):summary([r['metrics'] for r in rows if r['count']==k]) for k in (1,2,3)}
    result=dict(status='PASS',cases=rows,by_count=groups,plan_sha256=sha(OUT/'PLAN.json'),
        optimizer_updates=0,heldout_access=False,inference_reference_free=True,
        all24_endpoint_orders_exact=True,all24_direct_iq_verified=True)
    write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json')))
    write(ROOT/'reports/2026-10-13/NATIVE_JOINT_FEATURES_RESULT.json',result)
    state('COMPLETE_AUDITED',cases=24)
    print(groups,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
