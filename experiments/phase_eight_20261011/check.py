"""Synthetic exact-equivariant source/permutation contract, not RF accuracy."""
import hashlib
import json
import torch
import core


def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(2);torch.manual_seed(0)
    x=torch.randn(1,100,dtype=torch.complex128)
    fractions=torch.tensor([.6,.3,.1,0.],dtype=torch.float64)[None,:,None]
    expected=fractions*x[:,None];calls=[]
    def predictor(net,item):
        assert 'references' not in item and 'active' not in item and 'construction_count' not in item
        angle=float(item['mixture'][0,0].angle());shift=int((angle+4)*4)%3
        order=[(i+shift)%3 for i in range(3)]+[3]
        calls.append(order)
        return (fractions*item['mixture'][:,None])[:,order],torch.zeros(1,3,dtype=torch.float64)
    item=dict(mixture=x,references=torch.empty(0),construction_count=torch.tensor([3]),active=torch.ones(1,3))
    m4,m8,_,diagnostic=core.predict_eight(None,predictor,item)
    anchor_order=calls[0]
    torch.testing.assert_close(m4,expected[:,anchor_order],rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(m8,expected[:,anchor_order],rtol=1e-14,atol=1e-14)
    assert len(calls)==8
    result=dict(status='PASS',synthetic_only=True,forward_calls=8,no_reference_or_count_input=True,
        changed_output_permutations=len({tuple(o) for o in calls}),
        max_four_error=float((m4-expected[:,anchor_order]).abs().max()),
        max_eight_error=float((m8-expected[:,anchor_order]).abs().max()),
        source_sha256={str(p.relative_to(core.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (core.FOUR_SOURCE,core.Path(core.__file__),core.Path(__file__))})
    (core.ROOT/'reports/2026-10-11/PHASE_EIGHT_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':main()
