"""Inspect position dependence of the actual retained TCN on constant inputs."""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/architecture_audit_20261009'))
import fit_diagnostic
import watch_epochs as w
from drone_rf.context_model import TemporalEncoder,align_context


def main():
    torch.set_num_threads(2);torch.manual_seed(604)
    receipt=w.read(ROOT/'local/native_frequency_20261009_v1/phase/CHECKPOINT.json')
    checkpoint=Path(receipt['path'])
    assert w.digest(checkpoint)==receipt['sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    net=TemporalEncoder('tcn').eval()
    net.load_state_dict({k[len('context_encoder.'):]:v for k,v in saved['model'].items()
                        if k.startswith('context_encoder.')})
    constant=torch.randn(1,65,1).expand(1,65,255)
    with torch.no_grad():
        encoded=net(constant)
        early=align_context(encoded,torch.tensor([4097]),500)
        center=align_context(encoded,torch.tensor([1_000_000]),500)
        position=float((early-center).square().mean().sqrt())
        edge=float((encoded[:,:,0]-encoded[:,:,127]).square().mean().sqrt())
    assert position>0 and edge>0
    source=ROOT/'experiments/rfuav_dense_gated_20261008/vendor/drone_rf/context_model.py'
    result=dict(status='PASS',source_sha256=w.digest(Path(__file__)),
        context_implementation_sha256=w.digest(source),parent_checkpoint_sha256=receipt['sha256'],
        input='synthetic time-constant65x255 features',crop_starts=[4097,1_000_000],
        fine_stft_frames=500,encoded_edge_center_rms_difference=edge,
        aligned_two_crop_positions_rms_difference=position,
        input_temporal_order_available=False,position_dependence_remains=True,
        recorded_iq_reads=0,training_updates=0,heldout_read=False,
        limitation='Feature-space differences, not RF waveform errors. Zero-padded TCN boundary position survives mean conditioning; hopping order does not. No causal claim about separation failures.')
    w.write(ROOT/'reports/2026-10-10/MEAN_CONTEXT_DIAGNOSIS.json',result)
    print(result)


if __name__=='__main__':main()
