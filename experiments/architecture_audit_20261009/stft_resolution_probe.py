"""TRAIN-only reference-power mask resolution diagnostic, not a separator.

All resolutions and cases are registered before I/Q is read. True individual
source powers are unavailable at inference. Scores are not performance bounds.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/rfuav_native_frequency_20261009'))
from native_data import NativeMixtures, sha256, write_json
from drone_rf.waveform import complex_si_sdr

SIZES = (512, 2048, 8192, 32768)
MODES = ('reference_frequency_mean_power', 'reference_time_frequency_power')


def analyze(wave, fft):
    window = torch.hann_window(fft, periodic=True).sqrt()
    return torch.stft(wave, n_fft=fft, hop_length=fft//4, window=window,
        center=True, pad_mode='reflect', onesided=False, return_complex=True)


def synthesize(z, fft, length):
    window = torch.hann_window(fft, periodic=True).sqrt()
    return torch.istft(z, n_fft=fft, hop_length=fft//4, window=window,
        center=True, onesided=False, return_complex=True, length=length)


def masks(spectra):
    power = spectra.abs().square()
    result = {}
    for mode, p in zip(MODES, (power.mean(-1, keepdim=True).expand_as(power), power)):
        total = p.sum(0, keepdim=True)
        result[mode] = torch.where(total > 0, p/torch.where(total > 0, total, 1.),
                                  torch.full_like(p, 1./spectra.shape[0]))
    return result


def scores(estimate, reference):
    e, r = estimate.to(torch.complex128), reference.to(torch.complex128)
    power = r.abs().square().mean(-1)
    nmse = (e-r).abs().square().mean(-1)/power
    si = complex_si_sdr(e, r)
    if not torch.isfinite(nmse).all():
        raise ValueError('Nonfinite NMSE')
    return dict(nmse=nmse.tolist(), si_sdr=[float(v) if torch.isfinite(v) else None for v in si],
                si_status=['finite' if torch.isfinite(v) else ('nan' if torch.isnan(v) else
                           'positive_infinity' if v > 0 else 'negative_infinity') for v in si])


def self_check():
    rng = torch.Generator().manual_seed(0)
    wave = torch.complex(torch.randn(2, 63872, generator=rng), torch.randn(2, 63872, generator=rng))
    checks = []
    for fft in SIZES:
        z = analyze(wave, fft)
        error = float((synthesize(z, fft, wave.shape[-1])-wave).abs().square().sum()/wave.abs().square().sum())
        if error > 1e-9:
            raise ValueError('Synthetic round trip failed')
        for mask in masks(z).values():
            torch.testing.assert_close(mask.sum(0), torch.ones_like(mask[0]), rtol=1e-6, atol=1e-6)
        for mask in masks(torch.zeros_like(z)).values():
            torch.testing.assert_close(mask, torch.full_like(mask, .5), rtol=0, atol=0)
        checks.append(dict(fft=fft, roundtrip_nmse=error))
    measured = scores(.5*wave, wave)
    np.testing.assert_allclose(measured['nmse'], [.25, .25], rtol=1e-12, atol=1e-12)
    return checks


def run(screen, output, public):
    if (output/'PROTOCOL.json').exists():
        raise ValueError('Refuse duplicate diagnostic registration')
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((screen/'PROTOCOL.json').read_text())
    preparation = Path(plan['preparation'])
    if sha256(preparation/'PREPARATION.json') != plan['preparation_sha256']:
        raise ValueError('Preparation changed')
    # Initialization checks metadata/schedules/features, without accessing source I/Q.
    data = NativeMixtures(preparation, 'train_pack', 1)
    used, selected = Counter(), []
    for i, row in enumerate(data.rows):
        count = int(row['count'])
        names = tuple(data.library.clips[int(j)]['category'] for j in row['indices'][:count])
        key = names + tuple(map(float, row['levels'][:count]))
        if used[key] < 2:
            selected.append(i)
            used[key] += 1
    if len(selected) != 48 or sum(int(data.rows[i]['count']) for i in selected) != 100:
        raise ValueError('Expected fixed 48 TRAIN cases and 100 reference appearances')
    sources = dict(plan['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__))
    for rel, digest in sources.items():
        if sha256(ROOT/rel) != digest:
            raise ValueError('Frozen source changed')
        target = output/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
    protocol = dict(status='REGISTERED_TRAIN_REFERENCE_RESOLUTION_DIAGNOSTIC',
        source_sha256=sources, screen_protocol_sha256=sha256(screen/'PROTOCOL.json'),
        preparation_sha256=plan['preparation_sha256'], train_rows_sha256=data.rows_hash,
        selected_indices=selected, selection='First two epoch1 TRAIN cases per category tuple and nominal level tuple; before IQ access',
        fft_sizes=list(SIZES), hop='n_fft/4', window='periodic sqrt-Hann', center=True,
        pad_mode='reflect', onesided=False, complex_input=True, sample_rate_hz=100_000_000,
        input_samples=63872, modes=list(MODES), source_assignment='fixed known source identity; no per-bin permutation',
        source_power_additivity_assumed=False, mixture_phase_preserved=True,
        requires_reference_iq=True, requires_true_count=True, model_updates=0,
        deployable=False, performance_bound=False, validation_iq_read=False, heldout_read=False,
        interpretation='Reference-assisted resolution sensitivity on fixed TRAIN examples; neither model performance nor inferability evidence',
        self_checks=self_check())
    write_json(output/'PROTOCOL.json', protocol)
    rows, start = [], time.time()
    with torch.inference_mode():
        for index in selected:
            item = data[index]
            count = int(item['construction_count'])
            refs = torch.as_tensor(item['references'][:count])
            mix = torch.as_tensor(item['mixture'])
            power = refs.to(torch.complex128).abs().square().mean(-1)
            if not bool((power > 0).all()):
                raise ValueError('Empty active reference')
            clips = [data.library.clips[int(i)] for i in data.rows[index]['indices'][:count]]
            resolutions = {}
            for fft in SIZES:
                spectra, z = analyze(refs, fft), analyze(mix, fft)
                roundtrip = scores(synthesize(spectra, fft, mix.numel()), refs)['nmse']
                if max(roundtrip) > 1e-9:
                    raise ValueError('Actual reference round trip failed')
                modes = {}
                for name, mask in masks(spectra).items():
                    estimate = synthesize(mask*z[None], fft, mix.numel())
                    relative = float((estimate.sum(0)-mix).to(torch.complex128).abs().square().sum()/mix.to(torch.complex128).abs().square().sum())
                    if relative > 1e-9:
                        raise ValueError('Mixture-sum reconstruction failed')
                    modes[name] = dict(**scores(estimate, refs), sum_relative_error=relative)
                resolutions[str(fft)] = dict(roundtrip_nmse=roundtrip, modes=modes)
            rows.append(dict(index=index, count=count, categories=[c['category'] for c in clips],
                pack_ids=[c['pack_id'] for c in clips], reference_power=power.tolist(),
                weakest_index=int(power.argmin()), resolutions=resolutions))
            write_json(output/'STATE.json', dict(status='CPU_TRAIN_REFERENCE_DIAGNOSIS',
                cases=len(rows), total=len(selected), seconds=time.time()-start, pid=os.getpid(), time=time.time()))
    summary = []
    for count in (1, 2, 3):
        subset = [r for r in rows if r['count'] == count]
        for fft in SIZES:
            for mode in MODES:
                metrics = [r['resolutions'][str(fft)]['modes'][mode] for r in subset]
                si = [v for m in metrics for v in m['si_sdr']]
                summary.append(dict(count=count, cases=len(subset), fft=fft, mode=mode,
                    mean_nmse=float(np.mean([v for m in metrics for v in m['nmse']])),
                    mean_si_sdr=float(np.mean(si)) if all(v is not None for v in si) else None,
                    nonfinite_si=sum(v is None for v in si),
                    weakest_nmse=float(np.mean([m['nmse'][r['weakest_index']] for m, r in zip(metrics, subset)]))))
    for rel, digest in sources.items():
        if sha256(ROOT/rel) != digest or sha256(output/'source_snapshot'/rel) != digest:
            raise ValueError('Source changed during diagnostic')
    result = dict(protocol, status='COMPLETE', protocol_sha256=sha256(output/'PROTOCOL.json'),
        seconds=time.time()-start, summary=summary, rows=rows)
    write_json(output/'COMPLETE.json', result)
    write_json(output/'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))
    write_json(public.with_suffix('.json'), result)
    lines = ['# TRAIN 전용 STFT 해상도 진단', '',
        '정답 파형의 전력을 알고 만든 마스크다. 실제 모델 점수나 성능 한계가 아니다. '
        '동일 TRAIN 48혼합·100성분, 같은 대역·원 중심 간격·전력·위상을 유지했다. '
        '검증/보류 I/Q를 열지 않았고 모델 학습은 없다. 모든 설정을 파형 접근 전에 등록했다.', '',
        '모든 마스크는 혼합 STFT의 복소 위상을 사용한다. 독립 신호 전력의 합이 혼합 전력과 같다고 가정하지 않는다. '
        '두 마스크 모두 정답 신호별 전력에 의존하므로 실제 추론에 사용하지 않는다. '
        'SI-SDR은 평균 제거 후 일정한 복소 배율에 불변인 지표다.', '',
        '|성분 수|혼합 수|FFT (창 μs)|정답 전력 마스크|평균 NMSE↓|약한 성분 NMSE↓|복소 SI-SDR dB↑|',
        '|---:|---:|---:|---|---:|---:|---:|']
    for s in summary:
        si = '비유한 값 있음' if s['mean_si_sdr'] is None else f"{s['mean_si_sdr']:.3f}"
        label = '주파수별 시간평균' if s['mode'] == MODES[0] else '시간·주파수별'
        lines.append(f"|{s['count']}|{s['cases']}|{s['fft']} ({s['fft']/100:.2f})|{label}|{s['mean_nmse']:.6f}|{s['weakest_nmse']:.6f}|{si}|")
    lines += ['', '단일 성분은 왕복 복원 대조다. FFT 길이와 hop이 함께 변하므로 시간/주파수 해상도 각각의 '
              '독립 효과를 분리한 실험은 아니다. 정답 전력 마스크는 일반적인 최적 복소 추정기와 같지 않다. '
              '평균이 좋아진 설정도 학습한 네트워크의 개선이나 미학습 기종 일반화를 증명하지 않는다.', '',
              f"규약 SHA-256: `{result['protocol_sha256']}`. 전체 행은 동명의 JSON에 보존했다.", '']
    public.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='COMPLETE', summary=summary, seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--screen', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--public', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    try:
        run(args.screen.resolve(), args.output.resolve(), args.public.resolve())
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output/'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
