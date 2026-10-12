"""CPU final50 diagnostic: did affinity fitting improve FIT waveform routing?

No optimizer or model selection. FIT labels are read for diagnostic loss only;
the same observed-only encoder and predicted-count assignment are retained.
This waits for the independent B/C direct-waveform audit before running.
"""
import gc
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha
from tracking_cache import feature_input
from routing_statistics import score
from spectral_routing import PERMS
from spectral_routing_constrained import fit
from diagnose_count_routing import summary

RUN=ROOT/'local/learned_tracking_20261013_v1'
CHECK=ROOT/'local/tracking_direct_audit_20261013_v1'
OUT=ROOT/'local/fitted_tracking_diagnosis_20261013_v1'


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def run():
    OUT.mkdir(exist_ok=True)
    while True:
        status=read(CHECK/'STATE.json')
        if status['status']=='COMPLETE_AUDITED':break
        if status['status']=='FAILED':raise RuntimeError('Inspect independent audit failure first')
        state('WAITING_DIRECT_AUDIT',previous=status['status']);time.sleep(10)
    protocol=read(RUN/'PROTOCOL.json');teacher=Path(protocol['teacher'])
    assert sha(CHECK/'RESULT.json')==read(CHECK/'AUDIT.json')['result_sha256']
    assert sha(teacher/'RESULT.json')==read(teacher/'AUDIT.json')['result_sha256']
    sources=dict(protocol['source_sha256']);sources[str(Path(__file__).resolve().relative_to(ROOT))]=sha(__file__)
    plan=ROOT/'reports/2026-10-13/FITTED_TRACKER_DIAGNOSIS_PLAN_KO.md';sources[str(plan.relative_to(ROOT))]=sha(plan)
    files={str(p):sha(p) for p in (RUN/'PROTOCOL.json',RUN/'ACTUAL_050.pt',teacher/'RESULT.json',CHECK/'RESULT.json')}
    assert not (OUT/'PLAN.json').exists()
    write(OUT/'PLAN.json',dict(registered=time.time(),source_sha256=sources,input_sha256=files,
        final_epoch=50,cases=48,role='fit_block',optimizer_updates=0))
    for name,h in sources.items():
        assert sha(ROOT/name)==h;p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,p);assert sha(p)==h
    import torch
    from learned_tracking_model import TrackingEmbedding,affinity_loss
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
    initial=TrackingEmbedding().eval();net=TrackingEmbedding().eval()
    saved=torch.load(RUN/'ACTUAL_050.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['epoch']==50 and saved['updates']==600;net.load_state_dict(saved['model']);del saved
    rows=[]
    for position,item in enumerate(read(teacher/'RESULT.json')['cases']):
        assert item['training'];state('FIT_DIAGNOSIS',case=position+1,total=48)
        observed,powers,bounds=feature_input(item)
        assert sha(item['labels'])==item['labels_sha256'] and sha(item['scoring'])==item['scoring_sha256']
        with np.load(item['labels'],allow_pickle=False) as z:
            labels=torch.from_numpy(z['labels'])[None];weights=torch.from_numpy(z['weights'])[None]
        with np.load(item['scoring'],allow_pickle=False) as z:stats={name:z[name].copy() for name in z.files}
        inputs=torch.from_numpy(observed)[None]
        with torch.no_grad():
            initial_embedding=initial(inputs);embedding=net(inputs)
            losses={'initial':float(affinity_loss(initial_embedding,labels,weights)),
                    'final50':float(affinity_loss(embedding,labels,weights))}
        metrics={}
        for name,v in (('psd',observed[...,:512].astype(np.float64)),
                       ('initial',np.pad(initial_embedding[0].numpy().astype(np.float64),((0,0),(0,0),(0,384)))),
                       ('final50',np.pad(embedding[0].numpy().astype(np.float64),((0,0),(0,0),(0,384))))):
            _,choices,receipt=fit(v,powers,item['predicted_count']);orders=np.array([PERMS[int(q)] for q in choices])
            metrics[name]=score(stats,orders,item['count'],item['predicted_count'])
        row=dict(index=item['index'],count=item['count'],categories=item['metadata']['categories'],
            nominal_levels_db=item['metadata']['nominal_levels_db'],affinity_loss=losses,
            parent=item['baseline_metrics'],**metrics)
        write(OUT/f'CASE_{item["index"]:03d}.json',row);rows.append(row)
        del observed,inputs,stats,initial_embedding,embedding,labels,weights;gc.collect()
    summaries={name:{str(k):summary([r[name] for r in rows if r['count']==k]) for k in (1,2,3)}
               for name in ('parent','psd','initial','final50')}
    assert all(sha(ROOT/name)==h for name,h in sources.items()) and all(sha(name)==h for name,h in files.items())
    result=dict(status='PASS',cases=rows,by_count=summaries,
        affinity_loss={name:float(np.mean([r['affinity_loss'][name] for r in rows])) for name in ('initial','final50')},
        role='fit_block',new_recording_validation=False,optimizer_updates=0,epoch_fixed=50,
        plan_sha256=sha(OUT/'PLAN.json'),no_model_selection=True,
        caveat='FIT performance and affinity-to-waveform diagnostic only; not independent efficacy or a control training trajectory')
    write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json')))
    write(ROOT/'reports/2026-10-13/FITTED_TRACKER_DIAGNOSIS_RESULT.json',result)
    state('COMPLETE_AUDITED',cases=48)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        state('FAILED');raise
