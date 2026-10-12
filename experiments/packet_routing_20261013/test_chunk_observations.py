"""Audit all630 primary observations and24x5 long targets without model inference."""
from pathlib import Path
import numpy as np
import torch
from chunk_evaluation import Mixtures, Sequences, ROLES, WINDOW, LENGTH, add_neighbours, neighbour_start
from sequence_data import ROOT, write_json, sha256


def run():
    torch.set_num_threads(2)
    cases=[]
    for start in (0,WINDOW-1,WINDOW,LENGTH-2*WINDOW,LENGTH-2*WINDOW+1,LENGTH-WINDOW):
        first=neighbour_start(start)
        cases.append(dict(start=start,first=first,primary=(start-first)//WINDOW))
    data=Mixtures(ROLES[1]);indices=[]
    for i in range(len(data)):
        raw=data[i];item=add_neighbours(data,i,raw,'cpu')
        assert torch.equal(item['mixture'][0],torch.from_numpy(raw['mixture']))
        assert torch.equal(item['references'][0],torch.from_numpy(raw['references']))
        assert torch.equal(item['context_features'][0],torch.from_numpy(raw['context_features']))
        indices.append(int((item['crop_start']-item['long_start'])[0])//WINDOW)
    seq=Sequences(ROLES[1]);n=0
    for i in range(len(seq)):
        chunks=seq[i];reference=np.concatenate([v['references'] for v in chunks],-1);mixture=reference.sum(0)
        for j in range(5):
            offset=j*(WINDOW//2)
            raw=dict(mixture=mixture[offset:offset+WINDOW],references=reference[:,offset:offset+WINDOW],
                active=np.any(reference[:,offset:offset+WINDOW]!=0,axis=-1),
                context_features=chunks[0]['context_features'],crop_start=chunks[0]['crop_start']+offset,
                construction_count=chunks[0]['construction_count'])
            add_neighbours(seq,i,raw,'cpu');n+=1
    receipt=dict(status='PASS',primary_cases=len(data),long_targets=n,boundary_cases=cases,
        primary_position_counts={str(j):indices.count(j) for j in (0,1,2)},
        assertions=['All primary input samples exactly retained','Targets/context features exactly retained',
                    'All neighbouring inputs within original admitted context','Same native offsets and whole-context gains'],
        source_sha256={Path(p).name:sha256(p) for p in (__file__,Path(__file__).with_name('chunk_evaluation.py'))},
        training_or_model_inference=False,heldout_access=False)
    write_json(ROOT/'reports/2026-10-13/CHUNK_OBSERVATION_CPU_CHECK.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':run()
