"""CPU algebra/implementation checks for the actual 1--3-source waveform loss.

Uses synthetic complex tensors, not a reduced separator or an RF performance
trial. Frozen running models, loss code and study definitions are untouched.
"""
import itertools
import math
from pathlib import Path
import torch
import fit_diagnostic
from drone_rf.losses import pit_waveform_loss
from drone_rf.waveform import analyze,synthesize,complex_si_sdr
import watch_epochs as watch


def independent_loss(est,ref,active,mix):
    """Scalar complex arithmetic, separate from the broadcast Torch implementation."""
    def power(a):return sum(abs(v)**2 for v in a)/len(a)
    def center(a):
        mean=sum(a)/len(a)
        return [v-mean for v in a]
    costs=[];assignments=[]
    for e,r,a,x in zip(est.tolist(),ref.tolist(),active.tolist(),mix.tolist()):
        px=max(power(x),1e-8);pairs=[]
        for output in e[:3]:
            line=[];ec=center(output);ve=power(ec)
            for target,valid in zip(r,a):
                pr=power(target);den=max(pr,1e-6*px) if valid else px
                nmse=power([u-v for u,v in zip(output,target)])/den
                rc=center(target);cross=abs(sum(u*v.conjugate() for u,v in zip(ec,rc))/len(x))**2
                coherence=1-min(1,max(0,cross/max(ve*power(rc),1e-16)))
                line.append(nmse+(coherence if valid and pr>1e-6*px else 0))
            pairs.append(line)
        permutations=list(itertools.permutations(range(3)))
        scores=[sum(pairs[p[j]][j] for j in range(3))/3 for p in permutations]
        which=min(range(6),key=lambda j:scores[j]);assignments.append(permutations[which])
        bg=[u-sum(v) for u,v in zip(x,zip(*r))]
        costs.append(scores[which]+power([u-v for u,v in zip(e[-1],bg)])/px)
    return sum(costs)/len(costs),assignments


def main():
    torch.set_num_threads(2);torch.manual_seed(20261009)
    ref=torch.randn(3,3,257,dtype=torch.complex128)
    active=torch.arange(3)[None,:]<torch.arange(1,4)[:,None]
    ref=ref*active[...,None]*torch.tensor([1.,.3,.1])[None,:,None]
    mixture=ref.sum(1)
    raw=torch.randn(3,4,257,dtype=torch.complex128)
    projection=lambda r:r+(mixture-r.sum(1))[:,None]/4
    estimates=projection(raw)
    value=pit_waveform_loss(estimates,ref,active,mixture)
    scalar,orders=independent_loss(estimates,ref,active,mixture)
    assert math.isclose(float(value['loss']),scalar,rel_tol=1e-12,abs_tol=1e-12)
    assert value['assignment'].tolist()==[list(p) for p in orders]
    for order in itertools.permutations(range(3)):
        changed=estimates[:,list(order)+[3]]
        loss=pit_waveform_loss(changed,ref,active,mixture)['loss']
        torch.testing.assert_close(loss,value['loss'],rtol=1e-12,atol=1e-12)
    perfect=torch.cat((ref,torch.zeros_like(ref[:,:1])),1)
    perfect_loss=float(pit_waveform_loss(perfect,ref,active,mixture)['loss'])
    assert abs(perfect_loss)<1e-12
    p=torch.eye(4,dtype=torch.float64)-torch.ones(4,4,dtype=torch.float64)/4
    assert int(torch.linalg.matrix_rank(p))==3
    torch.testing.assert_close(p@p,p,rtol=0,atol=0)
    assert float((estimates.sum(1)-mixture).abs().max())<1e-14
    common=torch.randn(3,1,257,dtype=torch.complex128)
    null_error=float((projection(raw+common)-estimates).abs().max())
    assert null_error<1e-14
    assert float((projection(perfect)-perfect).abs().max())<1e-14
    # Distinct target decompositions produce the same single-channel mixture.
    shifted=perfect.clone();delta=torch.randn_like(shifted[:,0])
    shifted[:,0]+=delta;shifted[:,1]-=delta
    assert float((shifted.sum(1)-mixture).abs().max())<1e-14
    # Sum-zero errors for THREE sources do not have the two-source equal-energy law.
    e=torch.tensor([.2,-.1,-.1,0.],dtype=torch.float64)
    assert abs(float(e.sum()))<1e-15 and float(e[0].square())!=float(e[1].square())
    # Exact waveform/STFT roundtrip with the real implementation.
    wave=torch.randn(1,4096,dtype=torch.complex128)
    restored=synthesize(analyze(wave),wave.shape[-1])
    roundtrip=float((restored-wave).abs().square().mean()/wave.abs().square().mean())
    # The production helper intentionally builds a float32 window, even when
    # this diagnostic supplies complex128 inputs. Its overlap normalization
    # therefore has float32 rounding. Do not silently assume a float64 window.
    assert roundtrip<1e-12
    window64=torch.hann_window(512,periodic=True,dtype=torch.float64).sqrt()
    spec64=torch.stft(wave,n_fft=512,hop_length=128,window=window64,center=True,
        pad_mode='reflect',onesided=False,return_complex=True)
    restored64=torch.istft(spec64,n_fft=512,hop_length=128,window=window64,center=True,
        onesided=False,length=wave.shape[-1],return_complex=True)
    roundtrip64=float((restored64-wave).abs().square().mean()/wave.abs().square().mean())
    assert roundtrip64<1e-24
    # Complex gain invariance is intentional in SI-SDR, not in raw NMSE.
    estimate=wave+.2*torch.randn_like(wave)
    a=complex_si_sdr(estimate,wave);b=complex_si_sdr(estimate*(.3+.7j),wave)
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    result=dict(status='PASS',source_sha256={str(Path(__file__).relative_to(watch.ROOT)):watch.digest(Path(__file__)),
        'experiments/rfuav_dense_gated_20261008/vendor/drone_rf/losses.py':watch.digest(watch.ROOT/'experiments/rfuav_dense_gated_20261008/vendor/drone_rf/losses.py'),
        'experiments/rfuav_dense_gated_20261008/vendor/drone_rf/waveform.py':watch.digest(watch.ROOT/'experiments/rfuav_dense_gated_20261008/vendor/drone_rf/waveform.py')},
        independent_pit_loss=scalar,implemented_pit_loss=float(value['loss']),
        all_six_output_permutations_invariant=True,counts_checked=[1,2,3],
        perfect_reconstruction_loss=perfect_loss,uniform_projection_rank=3,
        projection_common_mode_null_error=null_error,all_feasible_target_waveforms_representable=True,
        same_sum_not_unique_separation=True,two_source_error_energy_law_does_not_extend=True,
        complex_stft_roundtrip_nmse=roundtrip,float64_window_roundtrip_nmse=roundtrip64,
        initial_check_failure='An initial 1e-24 assertion wrongly assumed a float64 production window; measured float32-window rounding before correcting the diagnostic tolerance to1e-12. Production code unchanged.',
        complex_si_gain_invariance=True,
        recorded_iq_reads=0,checkpoint_reads=0,training_updates=0,
        limitation='Algebra and loss implementation checks only; no proof of learnability, generalization, or RF separation performance.')
    watch.write(watch.ROOT/'reports/2026-10-09/THREE_SOURCE_MATH_CHECK.json',result)
    print(result)


if __name__=='__main__':main()
