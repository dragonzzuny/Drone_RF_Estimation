"""TRAIN-only, nondeployable STFT allocation diagnosis; no network training.

References and true count are explicitly used. Neither this diagnostic nor
its per-bin optimum is a bound on unrestricted waveform reconstruction.
"""
import argparse
import os
from pathlib import Path
import statistics as stats
import time
import traceback

import numpy as np
import torch
import train_comparison as worker
from drone_rf.waveform import analyze, synthesize, complex_si_sdr

w = worker.watch


def simplex(v):
    """Euclidean projection along the source axis of [sources, ...]."""
    ordered = v.sort(dim=0, descending=True).values
    partial = ordered.cumsum(0) - 1
    ranks = torch.arange(1, v.shape[0] + 1, dtype=v.dtype).reshape(-1, *([1] * (v.ndim - 1)))
    supported = ordered - partial / ranks > 0
    rho = supported.sum(0).clamp_min(1)
    theta = partial.gather(0, (rho - 1)[None])[0] / rho
    return (v - theta).clamp_min(0)


def allocations(spectra, mixture=None):
    if mixture is None:
        mixture = spectra.sum(0)
    magnitude = spectra.abs()
    power = magnitude.square()
    total = power.sum(0)
    abs_total = magnitude.sum(0)
    observed = mixture.abs().square()
    n = len(spectra)
    positive = observed > 0
    phase_sensitive = (spectra * mixture.conj()).real / torch.where(positive, observed, 1.)
    phase_sensitive = torch.where(positive, phase_sensitive, 1. / n)
    return dict(uniform=torch.full_like(power, 1. / n),
        oracle_power=torch.where(total > 0, power / torch.where(total > 0, total, 1.), 1. / n),
        oracle_magnitude=torch.where(abs_total > 0, magnitude / torch.where(abs_total > 0, abs_total, 1.), 1. / n),
        oracle_phase_sensitive_simplex=simplex(phase_sensitive))


def checks():
    # Disjoint support is reconstructed exactly by both reference ratios.
    disjoint = torch.tensor([[1+2j, 0], [0, 2-1j]], dtype=torch.complex128)
    for key, mask in allocations(disjoint).items():
        torch.testing.assert_close(mask.sum(0), torch.ones(2, dtype=torch.float64))
        if key != 'uniform':
            torch.testing.assert_close(mask * disjoint.sum(0), disjoint)
    # Orthogonal complex phases cannot be recovered by nonnegative real masks.
    orthogonal = torch.tensor([[1+0j], [1j]], dtype=torch.complex128)
    mask = allocations(orthogonal)['oracle_phase_sensitive_simplex']
    torch.testing.assert_close(mask, torch.full((2, 1), .5, dtype=torch.float64))
    torch.testing.assert_close((mask * orthogonal.sum(0) - orthogonal).abs().square(),
                              torch.full((2, 1), .5, dtype=torch.float64))
    # Exact cancellation and a negative unconstrained mask stay finite.
    for value in ([[1+0j], [-1+0j]], [[1+0j], [-.4+0j]]):
        for mask in allocations(torch.tensor(value, dtype=torch.complex128)).values():
            assert torch.isfinite(mask).all() and (mask >= 0).all()
            torch.testing.assert_close(mask.sum(0), torch.ones(1, dtype=torch.float64))
    rng = torch.Generator().manual_seed(42)
    s = torch.complex(torch.randn(3, 12, generator=rng, dtype=torch.float64),
                      torch.randn(3, 12, generator=rng, dtype=torch.float64))
    optimum = allocations(s)['oracle_phase_sensitive_simplex']
    best_cost = (optimum * s.sum(0) - s).abs().square().sum(0)
    for _ in range(100):
        trial = torch.rand(3, 12, generator=rng, dtype=torch.float64)
        trial /= trial.sum(0)
        assert torch.all(best_cost <= (trial * s.sum(0) - s).abs().square().sum(0) + 1e-12)
    return dict(status='PASS', disjoint_support=True, orthogonal_phases=True,
                exact_cancellation=True, negative_unconstrained_mask=True,
                simplex_cost_random_comparisons=1200)


@torch.inference_mode()
def run(study, trainfit, root, public):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Duplicate oracle diagnosis')
    test = checks()
    p = w.read(study / 'PROTOCOL.json')
    fitted = w.read(trainfit / 'PROTOCOL.json')
    sources = dict(p['source_sha256'])
    sources[str(Path(__file__).relative_to(w.ROOT))] = w.digest(Path(__file__))
    protocol = dict(status='REGISTERED_TRAIN_ONLY_ORACLE_ALLOCATION', source_sha256=sources,
        study_protocol_sha256=w.digest(study / 'PROTOCOL.json'),
        trainfit_protocol_sha256=w.digest(trainfit / 'PROTOCOL.json'),
        candidate_indices=fitted['train_indices'], count_filter=[2, 3], expected_cases=38,
        methods=['uniform', 'oracle_power', 'oracle_magnitude', 'oracle_phase_sensitive_simplex'],
        references_used=True, true_count_used=True, inference_deployable=False,
        reference_assignment='known construction identity, no PIT relabeling',
        stft='existing nfft512 hop128 sqrt-Hann; complex128 diagnostic arithmetic',
        numerical_check='Account explicitly for float32 mixture-sum rounding; check STFT linearity after subtracting its transformed rounding residual',
        target='original common-RF-band-limited recorded contributions including receiver noise',
        limitations='Oracle simplex minimizes per-bin squared STFT error with nonnegative masks summing to one. Not a waveform NMSE bound; not an identifiability or receiver-noise floor.',
        training_updates=0, heldout_read=False, independent_test=False, checks=test,
        registered_at=time.time())
    w.write(root / 'PROTOCOL.json', protocol)
    while not (trainfit / 'COMPLETE.json').exists():
        if (trainfit / 'FAILURE.json').exists():
            raise RuntimeError('Prior CPU diagnostic failed')
        if time.time() - protocol['registered_at'] > 7200:
            raise TimeoutError('Prior CPU diagnostic did not complete')
        w.write(root / 'STATE.json', dict(status='WAITING_CPU_TRAIN_FIT', pid=os.getpid(), time=time.time()))
        time.sleep(30)
    for rel, sha in sources.items():
        if w.digest(w.ROOT / rel) != sha:
            raise ValueError('Frozen source changed')
    # NativeMixtures.__getitem__ requires its sealed mixture feature cache even
    # though this diagnostic does not consume context features.
    data = worker.NativeMixtures(p['preparation'], 'train_pack', 1)
    rows = []
    for index in protocol['candidate_indices']:
        item = data[index]
        n = item['construction_count']
        if n not in (2, 3):
            continue
        ref = torch.as_tensor(item['references'][:n]).to(torch.complex128)
        mix = torch.as_tensor(item['mixture']).to(torch.complex128)
        spectrum = analyze(ref)
        observed = analyze(mix)
        power = ref.abs().square().mean(-1)
        mix_power = mix.abs().square().mean()
        rounding = mix - ref.sum(0)
        rounding_nmse = float(rounding.abs().square().mean() / mix_power)
        rounding_bound = float((n * np.finfo(np.float32).eps) ** 2 * power.sqrt().sum().square() / mix_power)
        if rounding_nmse > rounding_bound:
            raise ValueError('Mixture discrepancy exceeds float32 summation bound')
        residual = observed - spectrum.sum(0) - analyze(rounding)
        linearity_error = float(residual.abs().square().sum() / spectrum.abs().square().sum())
        if linearity_error > (32 * 512 * np.finfo(np.float64).eps) ** 2:
            raise ValueError('STFT linearity residual beyond float64 tolerance')
        weak = int(power.argmin())
        roundtrip = synthesize(spectrum, mix.numel())
        roundtrip_error = ((roundtrip - ref).abs().square().mean(-1) / power).max().item()
        if roundtrip_error > 1e-10:
            raise ValueError('STFT round trip failed')
        masks = allocations(spectrum, observed)
        spectral_costs = {}
        results = {}
        for name, mask in masks.items():
            torch.testing.assert_close(mask.sum(0), torch.ones_like(mask[0]), rtol=1e-10, atol=1e-10)
            estimate_stft = mask * observed
            spectral_costs[name] = float((estimate_stft - spectrum).abs().square().sum())
            estimate = synthesize(estimate_stft, mix.numel())
            nmse = ((estimate - ref).abs().square().mean(-1) / power).tolist()
            si = complex_si_sdr(estimate, ref).tolist()
            sum_error = float((estimate.sum(0) - mix).abs().square().mean() / mix.abs().square().mean())
            if not np.isfinite(nmse + si).all() or sum_error > 1e-10:
                raise ValueError('Invalid oracle waveform metric')
            results[name] = dict(nmse=nmse, si_sdr=si, sum_relative_error=sum_error)
        if any(spectral_costs['oracle_phase_sensitive_simplex'] > v + 1e-6 * max(1., v)
               for v in spectral_costs.values()):
            raise ValueError('Per-bin simplex optimum check failed')
        categories = [data.library.clips[int(i)]['category'] for i in data.rows[index]['indices'][:n]]
        rows.append(dict(index=index, count=n, categories=categories, reference_power=power.tolist(),
            weakest_index=weak, reference_roundtrip_max_nmse=roundtrip_error,
            mixture_rounding_nmse=rounding_nmse, float32_summation_bound=rounding_bound,
            stft_linearity_residual_over_reference_energy=linearity_error,
            stft_squared_errors=spectral_costs, results=results))
        w.write(root / 'STATE.json', dict(status='CPU_TRAIN_ORACLE', cases=len(rows),
            total=38, pid=os.getpid(), time=time.time()))
    if len(rows) != 38:
        raise ValueError('Wrong oracle cohort size')
    summaries = []
    for name in protocol['methods']:
        for n in (2, 3):
            subset = [r for r in rows if r['count'] == n]
            summaries.append(dict(method=name, count=n, cases=len(subset),
                mean_nmse=stats.mean(v for r in subset for v in r['results'][name]['nmse']),
                mean_si_sdr=stats.mean(v for r in subset for v in r['results'][name]['si_sdr']),
                weakest_nmse=stats.mean(r['results'][name]['nmse'][r['weakest_index']] for r in subset)))
    for rel, sha in sources.items():
        if w.digest(w.ROOT / rel) != sha:
            raise ValueError('Frozen source changed during diagnosis')
    result = dict(status='COMPLETE', diagnostic_protocol_sha256=w.digest(root / 'PROTOCOL.json'),
        study_protocol_sha256=protocol['study_protocol_sha256'], checks=test,
        summaries=summaries, rows=rows, references_used=True, true_count_used=True,
        training_updates=0, heldout_read=False, independent_test=False,
        limitation=protocol['limitations'])
    w.write(root / 'COMPLETE.json', result)
    w.write(public.with_suffix('.json'), result)
    lines = ['# 정답을 이용한 TRAIN 전력 배분 진단', '',
        '학습용 고정 TRAIN48 중 2·3성분 38혼합이다. 모든 마스크는 정답과 합성 개수를 사용하며 '
        '실제 추론 방법이나 새 기록 일반화 성능이 아니다. 모델·손실·GPU 학습을 변경하지 않았다.', '',
        '| 진단 | 성분 수 | 혼합 수 | NMSE ↓ | 복소 SI-SDR ↑ dB | 약신호 NMSE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['method']} | {row['count']} | {row['cases']} | {row['mean_nmse']:.6f} | {row['mean_si_sdr']:.3f} | {row['weakest_nmse']:.6f} |")
    lines += ['', '전력/크기 비율 마스크는 각 성분의 정답 STFT에서 계산한다. '
        'phase-sensitive simplex는 각 bin에서 음이 아닌 실수 마스크의 합을 1로 제한하고 '
        'STFT 제곱 오차를 최소화한다. 해당 결과는 중첩 STFT 합성 후 파형 NMSE의 최적값도, '
        '복소 출력을 자유롭게 예측하는 모델의 성능 한계도 아니다. 원 수신 잡음의 분리 가능성을 판정하지 않는다.', '']
    w.write(public.with_suffix('.md'), '\n'.join(lines))
    w.write(root / 'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))
    print(dict(status='COMPLETE', summaries=summaries), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('study', 'trainfit', 'run', 'public'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        run(args.study.resolve(), args.trainfit.resolve(), args.run.resolve(), args.public.resolve())
    except Exception:
        w.write(args.run / 'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
