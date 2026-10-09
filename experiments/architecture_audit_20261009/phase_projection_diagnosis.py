"""Post-hoc CPU diagnosis of the frozen local e1 mixture projection.

Selected failure/moderate cases only; not a new performance estimate. Hooks
observe the head, never change model output or provide targets to forward.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback

import numpy as np
import torch

import phase_data as data
import phase_packing as pp
import watch_epochs as watch
from native_data import NativeMixtures
from drone_rf.waveform import waveform_metrics


def run(study, root, public):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Duplicate diagnostic registration')
    plan = watch.read(study / 'PROTOCOL.json')
    scores = study / 'local/VALIDATION_001.json'
    validation = watch.read(scores)
    checkpoint = study / 'local/ACTUAL_001.pt'
    gap = lambda r: 10 * math.log10(max(r['reference_power']) / min(r['reference_power']))
    all_rows = validation['rows']
    indices = [r['index'] for r in sorted(
        [r for r in all_rows if r['count'] == 2 and gap(r) >= 20],
        key=lambda r: statistics.mean(r['nmse']), reverse=True)[:2]]
    indices += [next(r['index'] for r in all_rows if r['count'] == 2 and gap(r) < 10)]
    indices += [max([r for r in all_rows if r['count'] == 3 and gap(r) >= 20],
                   key=lambda r: statistics.mean(r['nmse']))['index']]
    indices += [next(r['index'] for r in all_rows if r['count'] == 3 and gap(r) < 10),
                next(r['index'] for r in all_rows if r['count'] == 1)]
    sources = dict(plan['source_sha256'])
    sources[str(Path(__file__).relative_to(watch.ROOT))] = watch.digest(Path(__file__))
    for rel, sha in sources.items():
        if watch.digest(watch.ROOT / rel) != sha:
            raise ValueError('Frozen diagnostic dependency changed')
        target = root / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(watch.ROOT / rel, target)
    protocol = dict(status='REGISTERED_POSTHOC_CPU_PROJECTION_DIAGNOSIS',
        source_sha256=sources, checkpoint_sha256=watch.digest(checkpoint),
        validation_sha256=watch.digest(scores), study_protocol_sha256=watch.digest(study / 'PROTOCOL.json'),
        indices=indices, selection='two worst2 gap>=20dB; first2 gap<10; worst3 gap>=20; first3 gap<10; first1',
        scope='chosen failure examples, not population performance or causal experiment',
        backend='CPU float32, two threads', assignment='original projected-output PIT mapping held fixed',
        model_updates=0, heldout_read=False, threshold_selection=False)
    watch.write(root / 'PROTOCOL.json', protocol)
    torch.set_num_threads(2)
    dataset = NativeMixtures(plan['preparation'], 'validation_pack', 1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['arm'] != 'local' or saved['epoch'] != 1 or saved['protocol_sha256'] != protocol['study_protocol_sha256']:
        raise ValueError('Wrong checkpoint')
    net = pp.PhasePackedWaveNet('local').eval()
    net.load_state_dict(saved['model'])
    del saved
    captured = []
    hook = net.net.output.register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
    rows, began = [], time.time()
    with torch.no_grad():
        for index in indices:
            captured.clear()
            item = data.batch(data.example(dataset, index), 'cpu')
            predicted = data.predict(net, item)
            score = waveform_metrics(predicted['estimates'], item['references'], item['active'], item['mixture'])
            count = int(item['construction_count'])
            reference = all_rows[index]
            active = item['active'][0]
            np.testing.assert_allclose(score['nmse'][0][active].numpy(), reference['nmse'], rtol=1e-3, atol=1e-5)
            np.testing.assert_allclose(score['si_sdr'][0][active].numpy(), reference['si_sdr'], rtol=0, atol=.01)
            if len(captured) != 1 or score['sum_relative_error'].max() > 1e-9:
                raise ValueError('Unexpected head execution or mixture inconsistency')
            scale = item['long_mixture'].abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            offsets = (item['crop_start'] % pp.PACK)[:, None] + torch.arange(pp.FINE)[None]
            raw = pp.unpack_iq(captured[0].float(), sources=4).gather(2, offsets[:, None].expand(-1, 4, -1)) * scale[:, None]
            correction = (item['mixture'] - raw.sum(1)) / 4
            torch.testing.assert_close(raw + correction[:, None], predicted['estimates'], rtol=1e-5, atol=1e-6)
            assignment = score['assignment'][0][:count]
            raw_aligned = raw[0, assignment].to(torch.complex128)
            target = item['references'][0, :count].to(torch.complex128)
            error = raw_aligned - target
            delta = correction[0].to(torch.complex128)
            power = target.abs().square().mean(-1)
            before = error.abs().square().mean(-1) / power
            addition = delta.abs().square().mean() / power
            cross = 2 * (error.conj() * delta).mean(-1).real / power
            after = score['nmse'][0, :count]
            np.testing.assert_allclose((before + addition + cross).numpy(), after.numpy(), rtol=1e-4, atol=1e-5)
            row = dict(index=index, count=count, categories=reference['categories'],
                actual_power_gap_db=gap(reference), weakest_index=reference['weakest_index'],
                fixed_assignment=assignment.tolist(), reference_power=power.tolist(),
                raw_nmse_fixed_assignment=before.tolist(), projection_energy_over_reference=addition.tolist(),
                signed_cross_term=cross.tolist(), projected_nmse=after.tolist(),
                correction_to_mixture_power=float(correction.abs().square().mean()/item['mixture'].abs().square().mean()),
                cpu_gpu_nmse_max_difference=float(np.max(np.abs(after.numpy()-reference['nmse']))),
                reference_used_only_in_scoring=True)
            rows.append(row)
            watch.write(root / 'PARTIAL.json', dict(rows=rows, partial=True))
            watch.write(root / 'STATE.json', dict(status='CPU_DIAGNOSIS', cases=len(rows), total=len(indices),
                        pid=os.getpid(), time=time.time()))
            print(json.dumps(row), flush=True)
    hook.remove()
    for rel, sha in sources.items():
        if watch.digest(watch.ROOT / rel) != sha:
            raise ValueError('Source changed during diagnosis')
    result = dict(status='COMPLETE', protocol_sha256=watch.digest(root/'PROTOCOL.json'),
        checkpoint_sha256=protocol['checkpoint_sha256'], rows=rows, seconds=time.time()-began,
        post_hoc=True, independent_test=False, heldout_read=False, model_updates=0,
        decomposition='NMSE_after=NMSE_raw+projection_energy/reference_power+signed_cross_term',
        assignment='projected-output PIT fixed for all terms; raw is not re-matched')
    watch.write(root/'COMPLETE.json', result)
    watch.write(public.with_suffix('.json'), result)
    lines = ['# 새 local e1: 합 일치 보정 전후 오차 진단', '',
        '기존 개발 검증에서 선택한 실패·중간 조건6개를 같은 CPU로 재계산했다. '
        '정답은 채점에만 쓰며 모델 입력·가중치를 변경하지 않았다. '
        '원래 투영 출력의 PIT 대응을 고정해 보정 전후를 비교한다. 원래 GPU 점수 재현도 확인했다.', '',
        '|검증행|성분 수|실제 전력차 dB|약신호 NMSE 보정 전|보정항 에너지/정답전력|부호 있는 교차항|보정 후|',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        j = r['weakest_index']
        lines.append(f"|{r['index']}|{r['count']}|{r['actual_power_gap_db']:.2f}|"
            f"{r['raw_nmse_fixed_assignment'][j]:.6f}|{r['projection_energy_over_reference'][j]:.6f}|"
            f"{r['signed_cross_term'][j]:.6f}|{r['projected_nmse'][j]:.6f}|")
    lines += ['', '교차항을 포함한 합이 실제 보정 후 NMSE와 일치하는지 검산했다. '
        '세 항을 각각 독립적인 원인 비율로 해석하지 않는다. 보정 전 출력은 혼합 합 일치를 '
        '만족하지 않으며, 일부 약신호 NMSE가 낮아도 더 좋은 분리기라는 뜻은 아니다. '
        '동일 보정은 이전 WaveNet에도 사용됐다. 따라서 이 진단만으로 보정 자체가 전체 실패의 '
        '원인이라거나 제거하면 학습이 개선된다고 결론 내리지 않는다. 전체 검증 성능은 원래630행을 유지한다.', '']
    watch.write(public.with_suffix('.md'), '\n'.join(lines))
    watch.write(root/'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('study', 'run', 'public'): p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    try:
        run(a.study.resolve(), a.run.resolve(), a.public.resolve())
    except Exception:
        a.run.mkdir(parents=True, exist_ok=True)
        watch.write(a.run/'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
