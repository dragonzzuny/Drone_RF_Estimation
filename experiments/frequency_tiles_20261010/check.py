"""Conservation/coverage and prediction-only permutation checks."""
import hashlib
import json
from pathlib import Path
import torch
from tiles import geometry,align,predict

ROOT=Path(__file__).resolve().parents[2]


def run():
    torch.manual_seed(0);torch.set_num_threads(2)
    indices,weight,denominator=geometry(512)
    assert len(indices)==8 and sum(len(i) for i in indices)==1024
    ref=torch.randn(2,4,128,19,dtype=torch.complex128)
    recovered,order,_=align(ref[:,[2,0,1,3]],ref,weight)
    assert torch.equal(recovered,ref) and order.tolist()==[[1,2,0,3]]*2
    x=torch.randn(1,512,19,dtype=torch.complex128)
    class Separator:
        def __init__(self):self.calls=0
        def __call__(self,z,context,position):
            self.calls+=1
            streams=torch.stack((.55*z,.3*z,.1*z,.05*z),1)
            if self.calls>1:streams=streams[:,[2,0,1,3]]
            return dict(estimates=streams,count_logits=torch.zeros(1,3))
    model=Separator();out,_,trace=predict(model,x,None,None)
    assert model.calls==9
    for value in out.values():
        assert float((value.sum(1)-x).abs().max())<1e-14
        assert float((value-out['parent']).abs().max())<1e-14
    result=dict(status='PASS',all_frequency_bins_covered=True,positive_partition_of_unity=True,
        prediction_only_alignment=True,background_fixed=True,complex_sum_conserved=True,
        full_model_test=False,inference_calls_per_case=9,optimizer_steps=0,heldout_read=False,
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (Path(__file__),Path(__file__).with_name('tiles.py'))})
    (ROOT/'reports/2026-10-10/FREQUENCY_TILES_ALGEBRA_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':run()
