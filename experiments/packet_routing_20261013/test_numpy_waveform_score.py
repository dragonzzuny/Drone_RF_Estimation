"""Compare bounded CPU metrics to the frozen Torch definition on small signals."""
import numpy as np
import torch
from sequence_eval import score as torch_score
from numpy_waveform_score import score
from sequence_data import ROOT,write_json,sha256
from pathlib import Path


def run():
    rng=np.random.default_rng(1013);rows=[]
    for count in (1,2,3):
        for trial in range(8):
            ref=np.zeros((3,4096),np.complex64)
            ref[:count]=(rng.normal(size=(count,4096))+1j*rng.normal(size=(count,4096)))
            ref[:count]*=np.geomspace(.03,1,count)[:,None]
            mix=ref.sum(0)
            y=np.zeros((4,4096),np.complex64)
            y[:3]=ref+(rng.normal(size=ref.shape)+1j*rng.normal(size=ref.shape))*.02
            y[:3]=y[rng.permutation(3)]
            y[-1]=mix-y[:3].sum(0)
            logits=torch.zeros(1,3);logits[0,count-1]=1
            expected=torch_score(torch.from_numpy(y)[None],torch.from_numpy(ref)[None],
                torch.from_numpy(mix)[None],count,logits)
            actual=score(y,ref,mix,count,count,chunk=389)
            assert actual['assignment'][:count]==expected['assignment'][:count]
            for key in ('nmse','si_sdr','reference_power'):
                np.testing.assert_allclose(actual[key],expected[key],rtol=1e-9,atol=1e-9)
            rows.append(dict(count=count,trial=trial,active_assignment_equal=True))
    p=Path(__file__).with_name('numpy_waveform_score.py')
    write_json(ROOT/'reports/2026-10-13/NUMPY_WAVEFORM_SCORE_CPU_CHECK.json',
        dict(status='PASS',cases=rows,source_sha256=sha256(p),test_sha256=sha256(__file__),
             double_precision_statistics=True,rf_baseline_recheck_required=True))
    print('NUMPY_WAVEFORM_SCORE_PASS',len(rows),flush=True)


if __name__=='__main__':run()
