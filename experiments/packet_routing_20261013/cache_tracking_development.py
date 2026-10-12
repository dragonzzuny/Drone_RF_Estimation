"""CPU preparation only; no tracker training or neural experiment queued."""
import gc
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha
from tracking_cache import save_case
from routing_statistics import score
from diagnose_count_routing import verify_metrics
from sequence_data import Sequences,ROLES
from native_full_context import components

OUT=ROOT/'local/learned_tracking_preparation_20261013_v1'
HERE=Path(__file__).resolve().parent


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def run():
    check=read(ROOT/'reports/2026-10-13/ROUTING_STATISTICS_CPU_CHECK.json');assert check['status']=='PASS'
    assert check['source_sha256']==sha(HERE/'routing_statistics.py')
    sources={str((HERE/name).relative_to(ROOT)):sha(HERE/name) for name in (
        'tracking_cache.py','routing_statistics.py','cache_tracking_development.py','numpy_waveform_score.py')}
    previous=ROOT/'local/constrained_routing_20261013_v1'
    dependencies={str(previous/'RESULT.json'):sha(previous/'RESULT.json')}
    old=read(previous/'RESULT.json');assert dependencies[str(previous/'RESULT.json')]==read(previous/'AUDIT.json')['result_sha256']
    assert not (OUT/'PREPARATION.json').exists()
    write(OUT/'PREPARATION.json',dict(created=time.time(),source_sha256=sources,input_sha256=dependencies,
        scope='Already exposed B24/C24 development features and separate reference scoring caches',
        training_started=False,neural_training_queued=False,heldout_access=False))
    sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
    from native_data import NativeMixtures
    groups={}
    for group,name in (('B24','spectral_routing_24_20261013_v1'),('C24','spectral_routing_c24_20261013_v1')):
        directory=ROOT/'local'/name;result=read(directory/'RESULT.json');dest=OUT/group;dest.mkdir()
        assert sha(directory/'RESULT.json')==read(directory/'AUDIT.json')['result_sha256']
        if group=='B24':data=Sequences(ROLES[1])
        else:data=NativeMixtures(Path(read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')['preparation']),'validation_pack',1)
        comparator={r['index']:r for r in old['groups'][group]['cases']};items=[]
        for position,case in enumerate(result['cases']):
            i=case['index'];state('CPU_PREPARATION',group=group,case=position+1,total=24)
            path=directory/f'OUTPUT_{i:03d}.npy'
            if case.get('reused_parent_output'):path=ROOT/'local/full_context_parent_20261013_v1'/path.name
            assert sha(path)==case['parent_output_sha256']
            output=np.load(path,mmap_mode='r',allow_pickle=False)
            refs=data.components(i,True) if group=='B24' else components(data,i)
            metadata=dict(categories=case['categories'],nominal_levels_db=case['nominal_levels_db'],
                parent_output_sha256=sha(path),parent_output=str(path))
            item=save_case(dest,i,output,refs,case['parent']['count'],case['parent']['predicted_count'],metadata)
            verify_metrics(item['baseline_metrics'],case['parent'])
            order_path=previous/group/f'ORDERS_{i:03d}.json';orders=read(order_path)['orders']
            assert sha(order_path)==comparator[i]['orders_sha256']
            with np.load(item['scoring'],allow_pickle=False) as cache:
                alternative=score(cache,np.asarray([r['permutation'] for r in orders]),item['count'],item['predicted_count'])
            verify_metrics(alternative,comparator[i]['candidate']);item['constrained_baseline']=alternative
            items.append(item);del output,refs;gc.collect()
        groups[group]=items;del data;gc.collect()
    assert all(sha(ROOT/p)==h for p,h in sources.items()) and all(sha(p)==h for p,h in dependencies.items())
    result=dict(status='VERIFIED',groups=groups,source_sha256=sources,preparation_sha256=sha(OUT/'PREPARATION.json'),
        cases=48,reference_caches_separate_from_observed_features=True,neural_training_updates=0,heldout_access=False)
    write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),
        all48_parent_and_constrained_active_assignments_and_metrics_reproduced=True))
    write(ROOT/'reports/2026-10-13/TRACKING_CACHE_CPU_CHECK.json',dict(status='PASS',cases=48,
        source_sha256=sources,result_sha256=sha(OUT/'RESULT.json'),
        all48_parent_and_constrained_metrics_reproduced=True,training_or_gpu_work_started=False))
    state('COMPLETE_AUDITED',cases=48)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        state('FAILED');raise
