"""Input-preservation checks before any candidate follow-up is registered."""
from pathlib import Path
import sys
import numpy as np
import torch
from evaluate_context_candidate import ROOT,attach,add_native_context,read,sha,write
from sequence_data import WINDOW,LENGTH
from study import batch


def main():
    torch.set_num_threads(2)
    rng=np.random.default_rng(0)
    mixture=(rng.normal(size=LENGTH)+1j*rng.normal(size=LENGTH)).astype(np.complex64)
    starts=list(range(0,LENGTH-WINDOW+1,WINDOW//2))+[LENGTH-WINDOW,WINDOW-1,LENGTH-2*WINDOW+1]
    for start in starts:
        item=batch(dict(mixture=mixture[start:start+WINDOW].copy(),references=np.zeros((3,WINDOW),np.complex64),
            active=np.zeros(3,bool),context_features=np.zeros((65,255),np.float32),crop_start=start,construction_count=1),'cpu')
        result=attach(item,mixture,start,'cpu');first=int(result['long_start'][0])
        assert torch.equal(result['long_mixture'][0,start-first:start-first+WINDOW],item['mixture'][0])
    sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
    from native_data import NativeMixtures
    p=read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')
    data=NativeMixtures(Path(p['preparation']),'validation_pack',1);checks=[]
    for count in (1,2,3):
        index=next(i for i,r in enumerate(data.rows) if r['count']==count)
        raw=data[index];result=add_native_context(data,index,raw,'cpu')
        assert torch.equal(result['references'][0],torch.from_numpy(raw['references']))
        assert torch.equal(result['mixture'][0],torch.from_numpy(raw['mixture']))
        assert torch.equal(result['context_features'][0],torch.from_numpy(raw['context_features']))
        checks.append(dict(index=index,count=count,primary_observation_and_target_exact=True))
    result=dict(status='PASS',synthetic_full_context_positions=len(starts),native_c_checks=checks,
        model_inference=False,training=False,heldout_access=False,
        source_sha256={Path(p).name:sha(p) for p in (__file__,Path(__file__).with_name('evaluate_context_candidate.py'))})
    write(ROOT/'reports/2026-10-13/FOLLOWUP_CONTEXT_CPU_CHECK.json',result);print(result,flush=True)


if __name__=='__main__':main()
