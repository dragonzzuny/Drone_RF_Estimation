"""Fixed saved packet, both specified weights and both registered routing rules."""
import gc
import os
from pathlib import Path
import shutil
import time
import traceback
from evaluate_parent_full_context import ROOT,read,write,sha

HERE=Path(__file__).resolve().parent
OUT=ROOT/'local/routing_packet_20261013_v1'
BASE=ROOT.parent/'uav_analysis/rf_detection/separator_detector_20261012'
INPUTS={'parent99':ROOT/'local/packet_final100_20261013_v1',
        'independent3':ROOT/'local/sequence_packets_20261013_v1/independent'}


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def run():
    sources=dict(read(ROOT/'local/constrained_routing_20261013_v1/PLAN.json')['source_sha256'])
    for p in (Path(__file__),HERE/'check_packets.py',ROOT/'reports/2026-10-13/ROUTING_PACKET_PLAN_KO.md'):
        sources[str(p.relative_to(ROOT))]=sha(p)
    reference_path=BASE/'retest_best_e079/SEPARATED_EVENT.npz'
    files={str(p):sha(p) for p in (reference_path,BASE/'receiver_pilot_v1/PLAN.json')}
    for directory in INPUTS.values():
        verification=read(directory/'VERIFICATION.json');assert verification['status']=='PASS'
        for name in ('RESULT.json','ASSEMBLED.npz'):
            assert sha(directory/name)==verification['artifacts'][name]
            files[str(directory/name)]=sha(directory/name)
        files[str(directory/'VERIFICATION.json')]=sha(directory/'VERIFICATION.json')
    plan=dict(registered=time.time(),source_sha256=sources,input_sha256=files,arms=list(INPUTS),
        methods=['slot_initialized_clusters','constrained_slot_clusters'],samples=200000,
        baseline_reproduction_rtol=1e-5,baseline_reproduction_atol=1e-5,
        cpu_only=True,optimizer_updates=0,heldout_access=False,known_event=True,
        crc_guided_tuning=False,references_in_routing=False,no_payload_fields_stored=True)
    OUT.mkdir(exist_ok=True);assert not (OUT/'PLAN.json').exists();write(OUT/'PLAN.json',plan)
    for name,digest in sources.items():
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p);assert sha(p)==digest
    def verify():
        assert all(sha(ROOT/p)==h and sha(OUT/'source_snapshot'/p)==h for p,h in sources.items())
        assert all(sha(p)==h for p,h in files.items())
    verify()
    import numpy as np
    import torch
    from sequence_eval import score
    from spectral_routing_v2 import route as route_previous
    from spectral_routing_constrained import route as route_new
    from check_packets import run as receiver_check
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    original=np.load(reference_path,allow_pickle=False);refs=original['references'];mixture=original['mixture']
    assert mixture.shape==(plan['samples'],);ref=torch.from_numpy(refs)[None];mix=torch.from_numpy(mixture)[None]
    comparisons=[]
    for arm,directory in INPUTS.items():
        y=np.load(directory/'ASSEMBLED.npz',allow_pickle=False)['C8_hann']
        old=read(directory/'RESULT.json');expected=old['stages']['C8_hann']['metrics']
        predicted=expected['predicted_count'];logits=torch.zeros(1,3);logits[0,predicted-1]=1
        def scoring(value):return score(torch.from_numpy(value)[None],ref,mix,2,logits)
        with torch.no_grad():
            baseline=scoring(y)
            for key in ('nmse','si_sdr','reference_power'):
                np.testing.assert_allclose(baseline[key],expected[key],rtol=1e-5,atol=1e-5)
            assert baseline['assignment']==expected['assignment']
            for method in plan['methods']:
                state('CPU_ROUTING',arm=arm,method=method)
                if method=='slot_initialized_clusters':routed,receipt=route_previous(y,{},method,predicted,mixture)
                else:routed,receipt=route_new(y,predicted)
                metrics=scoring(routed);output=OUT/f'{arm}_{method}';output.mkdir(exist_ok=True)
                np.savez_compressed(output/'ASSEMBLED.npz',C8_hann=routed)
                result=dict(status='COMPLETE',epoch=99 if arm=='parent99' else 3,arm=arm,method=method,
                    stages={'C8_hann':dict(metrics=metrics,routing=receipt)})
                write(output/'RESULT.json',result)
                write(output/'VERIFICATION.json',dict(status='PASS',input_hashes_unchanged=True,
                    artifacts={n:sha(output/n) for n in ('RESULT.json','ASSEMBLED.npz')}))
                state('CPU_RECEIVER',arm=arm,method=method);receiver_check(output)
                audit=read(output/'PACKET_AUDIT.json');assert audit['status']=='PASS'
                comparisons.append(dict(arm=arm,method=method,baseline=baseline,metrics=metrics,
                    packet=read(output/'PACKET_RESULTS.json'),packet_audit=audit))
                del routed;gc.collect()
    verify();result=dict(status='VERIFIED',comparisons=comparisons,plan_sha256=sha(OUT/'PLAN.json'),
        references_in_routing=False,cpu_only=True,optimizer_updates=0,heldout_access=False,
        caveat='Known2ms event, mixture already decodes, no detector-gain or exclusive-separation claim')
    write(OUT/'RESULT.json',result);write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),
        original_waveform_metrics_reproduced=True,all4conditions_and7inputs_preserved=True))
    write(ROOT/'reports/2026-10-13/ROUTING_PACKET_RESULT.json',result);state('COMPLETE_AUDITED',conditions=4)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
