"""Skip unused decoders only; verify all primary positions at nonzero adapters."""
import gc
from pathlib import Path
import torch
from chunk_temporal_candidate import ChunkTemporalUNet
from dual_context_candidate import DualContextUNet
from long_temporal_candidate import ROOT
from sequence_data import sha256,write_json


def main():
    torch.set_num_threads(2);torch.manual_seed(0);rows=[]
    features=torch.randn(1,65,255);zs=[torch.randn(1,32,48,dtype=torch.complex64) for _ in range(3)]
    starts=[torch.tensor([40000+j*63872]) for j in range(3)]
    for factory in (ChunkTemporalUNet,DualContextUNet):
        net=factory().eval()
        with torch.no_grad():
            net.gates.expand.weight.normal_(std=.01)
            if hasattr(net,'ordered_projection'):net.ordered_projection.weight.normal_(std=.01)
            expected=net(zs,features,starts)
            for primary in (0,1,2):
                observed=net(zs,features,starts,primary_index=primary)
                torch.testing.assert_close(observed['estimates_primary'],expected['estimates'][primary],rtol=0,atol=0)
                torch.testing.assert_close(observed['count_logits'],expected['count_logits'],rtol=0,atol=0)
                rows.append(dict(architecture=factory.__name__,primary_index=primary,output_max_difference=0.))
        del net,expected,observed;gc.collect()
    result=dict(status='PASS',checks=rows,nonzero_added_modules=True,synthetic_weights_discarded=True,
        rf_training_updates=0,scope='Same encoders/context/connections; only unused neighbouring decoders skipped',
        source_sha256={n:sha256(Path(__file__).with_name(n)) for n in
                       ('test_selective_decode.py','chunk_temporal_candidate.py','dual_context_candidate.py')})
    write_json(ROOT/'reports/2026-10-13/SELECTIVE_DECODE_CPU_CHECK.json',result);print(result,flush=True)


if __name__=='__main__':main()
