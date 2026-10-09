"""Fixed U-Net TRAIN48 inference probe of power-weighted mixture projection.

One unchanged forward per mixture. The final raw STFT head is observed using a
hook, then its mixture residual is allocated uniformly or by estimated power.
References are used only for scoring. This is not a trained weighted projector.
The unprojected common component is not identified by the original loss, so
post-hoc raw-power weights need not be calibrated uncertainty estimates.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

import frequency_order_probe as prior
import watch_epochs as watch
from native_data import NativeMixtures
from models import build
from drone_rf.waveform import analyze, synthesize, waveform_metrics


def power_projection(raw, mixture):
    power = raw.abs().square()
    total = power.sum(1, keepdim=True)
    weights = torch.where(total > 0, power / total.clamp_min(torch.finfo(power.dtype).tiny),
                          torch.full_like(power, 1 / raw.shape[1]))
    return raw + weights * (mixture - raw.sum(1))[:, None]


def evaluate(net, item):
    captured = []
    hook = net.output.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
    batch = {k: torch.as_tensor(np.asarray(item[k])[None]) for k in
             ('mixture', 'references', 'active', 'context_features', 'crop_start')}
    try:
        with torch.inference_mode():
            z = analyze(batch['mixture'])
            result = net(z, batch['context_features'], batch['crop_start'])
            if len(captured) != 1:
                raise ValueError('Expected one raw output head')
            scale = z.abs().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
            head = captured[0].float()
            raw = torch.complex(head[:, 0::2], head[:, 1::2]) * scale[:, None]
            uniform = raw + (z - raw.sum(1))[:, None] / 4
            torch.testing.assert_close(uniform, result['estimates'], rtol=1e-5, atol=1e-6)
            count = int(item['construction_count'])
            scores = {}
            for name, spec in [('uniform', uniform), ('raw_power', power_projection(raw, z))]:
                estimate = synthesize(spec, batch['mixture'].shape[-1])
                score = waveform_metrics(estimate, batch['references'], batch['active'], batch['mixture'])
                nmse, si = (score[k][0, :count].tolist() for k in ('nmse', 'si_sdr'))
                if (not np.isfinite(nmse + si).all()
                        or float(score['sum_relative_error'].max()) > 1e-9):
                    raise ValueError('Nonfinite metric or mixture sum failure')
                scores[name] = dict(nmse=nmse, si_sdr=si,
                    reference_power=score['reference_power'][0, :count].tolist(),
                    assignment=score['assignment'][0, :count].tolist(),
                    sum_relative_error=float(score['sum_relative_error'].max()))
    finally:
        hook.remove()
    return scores


def run(screen, root, public):
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Duplicate registration')
    root.mkdir(parents=True, exist_ok=True)
    plan = watch.read(screen / 'PROTOCOL.json')
    train = NativeMixtures(plan['preparation'], 'train_pack', 1)
    used, indices = Counter(), []
    for i, row in enumerate(train.rows):
        n = int(row['count'])
        names = tuple(train.library.clips[int(j)]['category'] for j in row['indices'][:n])
        key = names + tuple(map(float, row['levels'][:n]))
        if used[key] < 2:
            indices.append(i)
            used[key] += 1
    if len(indices) != 48:
        raise ValueError('Expected the existing deterministic TRAIN48 rule')
    checkpoint = screen / 'unet_mean/SELECTED_004.pt'
    val_path = screen / 'unet_mean/VALIDATION_004.json'
    val_rows = watch.read(val_path)['rows']
    checks = [next(r for r in val_rows if r['count'] == n) for n in (1, 2, 3)]
    source = dict(plan['source_sha256'])
    for p in (Path(__file__), Path(prior.__file__), Path(watch.__file__)):
        source[str(p.relative_to(watch.ROOT))] = watch.digest(p)
    protocol = dict(status='REGISTERED_TRAIN_POWER_PROJECTION_PROBE',
        source_sha256=source, study_protocol_sha256=watch.digest(screen / 'PROTOCOL.json'),
        preparation_sha256=plan['preparation_sha256'], checkpoint_sha256=watch.digest(checkpoint),
        checkpoint_epoch=4, checkpoint_updates=300, parameters=32_142_859,
        validation_sha256=watch.digest(val_path), train_indices=indices,
        baseline_reproduction_indices=[r['index'] for r in checks],
        validation_use='Three already scored rows, uniform CPU/GPU agreement only',
        modes=['uniform', 'raw_power'], rule='v_j=abs(raw_STFT_j)^2; exact-zero bins use 1/4',
        prior_art='https://arxiv.org/html/1811.08521 section4.2 estimated-power weighting',
        model_updates=0, learned_weighting_trained=False, one_forward_per_mixture=True,
        train_only_comparison=True, heldout_read=False,
        backend='CPU float32, two threads; PIT independently scored per inference mode',
        gate='Both counts2/3 NMSE and weakest NMSE lower, complex SI-SDR higher before full DEV consideration',
        caveat='Post-hoc raw head has unconstrained common component; not calibrated uncertainty or learned-weight result')
    for rel, digest in source.items():
        if watch.digest(watch.ROOT / rel) != digest:
            raise ValueError('Source changed')
        target = root / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(watch.ROOT / rel, target)
    watch.write(root / 'PROTOCOL.json', protocol)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['best']['epoch'] != 4 or saved['protocol_sha256'] != protocol['study_protocol_sha256']:
        raise ValueError('Wrong selected model')
    net = build('unet_mean').eval()
    net.load_state_dict(saved['model'])
    if sum(p.numel() for p in net.parameters()) != protocol['parameters']:
        raise ValueError('Unexpected capacity')
    del saved
    validation = NativeMixtures(plan['preparation'], 'validation_pack', 1)
    backend_checks = []
    for row in checks:
        # Only uniform baseline is scored; no candidate validation selection here.
        actual = prior.evaluate(net, validation[row['index']], False)
        np.testing.assert_allclose(actual['nmse'], row['nmse'], rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(actual['si_sdr'], row['si_sdr'], rtol=0, atol=.01)
        backend_checks.append(dict(index=row['index'], passed=True))
    rows, began = [], time.time()
    for index in indices:
        item = train[index]
        count = int(item['construction_count'])
        modes = evaluate(net, item)
        clips = [train.library.clips[int(j)] for j in train.rows[index]['indices'][:count]]
        rows.append(dict(index=index, count=count, categories=[c['category'] for c in clips],
            weakest_index=int(np.argmin(modes['uniform']['reference_power'])), modes=modes))
        watch.write(root / 'PARTIAL.json', dict(rows=rows, partial=True))
        watch.write(root / 'STATE.json', dict(status='TRAIN_CPU_INFERENCE', cases=len(rows),
                    total=48, seconds=time.time()-began, pid=os.getpid(), time=time.time()))
    summary = []
    for count in (1, 2, 3):
        sub = [r for r in rows if r['count'] == count]
        for mode in protocol['modes']:
            summary.append(dict(count=count, cases=len(sub), mode=mode,
                mean_nmse=float(np.mean([v for r in sub for v in r['modes'][mode]['nmse']])),
                mean_si_sdr=float(np.mean([v for r in sub for v in r['modes'][mode]['si_sdr']])),
                weakest_nmse=float(np.mean([r['modes'][mode]['nmse'][r['weakest_index']] for r in sub]))))
    gate = all(summary[2*c-1]['mean_nmse'] < summary[2*c-2]['mean_nmse'] and
               summary[2*c-1]['mean_si_sdr'] > summary[2*c-2]['mean_si_sdr'] and
               summary[2*c-1]['weakest_nmse'] < summary[2*c-2]['weakest_nmse'] for c in (2, 3))
    for rel, digest in source.items():
        if watch.digest(watch.ROOT/rel) != digest or watch.digest(root/'source_snapshot'/rel) != digest:
            raise ValueError('Source changed during probe')
    result = dict(protocol, status='COMPLETE', protocol_sha256=watch.digest(root/'PROTOCOL.json'),
                  rows=rows, summary=summary, backend_checks=backend_checks,
                  train_gate_passed=gate, seconds=time.time()-began)
    watch.write(root/'COMPLETE.json', result)
    watch.write(public.with_suffix('.json'), result)
    lines = ['# U-Net e4: 합 일치 보정의 전력 가중 추론 진단', '',
        '고정한 원 규모 U-Net e4/300업데이트와 TRAIN48을 사용했다. 한 번의 원래 순전파에서 '
        '보정 전 복소 STFT를 관측하고, 남은 혼합 잔차를 동일 비율 또는 추정 전력 비율로 나눴다. '
        '정답·실제 개수·기종은 가중치 계산에 사용하지 않았다. 각 방식의 PIT 대응은 따로 채점했다.', '',
        '| 성분 수 | 혼합 수 | 보정 가중치 | NMSE ↓ | 복소 SI-SDR ↑ dB | 약신호 NMSE ↓ |',
        '|---:|---:|---|---:|---:|---:|']
    for s in summary:
        lines.append(f"|{s['count']}|{s['cases']}|{s['mode']}|{s['mean_nmse']:.6f}|{s['mean_si_sdr']:.3f}|{s['weakest_nmse']:.6f}|")
    lines += ['', f'전체 개발검증 검토를 위한 TRAIN 선별 기준 충족: {gate}. '
        '새 학습이나 일반화 검증 결과가 아니다. 기존 검증 3행은 원래 추론의 CPU/GPU 수치 일치만 확인했다.', '',
        '전력 가중 투영은 [Wisdom 등의 선행 방법](https://arxiv.org/html/1811.08521)의 후보를 참고했다. '
        '이 모델은 균등 보정으로 학습했으므로 추정 전력이 보정 불확실성을 잘 표현한다는 보장은 없다. '
        '보정 전 출력에 공통 성분을 더해도 원래 최종 출력은 같아지는 자유도가 있다. '
        '따라서 이 추론 변경의 성공·실패를 가중 투영을 포함해 새로 학습한 모델의 성능으로 해석하지 않는다.', '']
    watch.write(public.with_suffix('.md'), '\n'.join(lines))
    watch.write(root/'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))
    print(json.dumps(dict(status='COMPLETE', summary=summary, train_gate_passed=gate)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('screen', 'run', 'public'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    try:
        run(a.screen.resolve(), a.run.resolve(), a.public.resolve())
    except Exception:
        a.run.mkdir(parents=True, exist_ok=True)
        watch.write(a.run/'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
