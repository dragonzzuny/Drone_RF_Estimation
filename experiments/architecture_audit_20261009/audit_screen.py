"""Audit all three completed architecture arms on CPU; no held-out evaluation."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

import watch_epochs as watch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/rfuav_native_frequency_20261009'))
from report import summarize


def better(candidate, baseline):
    if any(g['mean_si_sdr'] is None for a in (candidate, baseline) for g in a['by_count'][1:]):
        return None
    return all(c['mean_nmse'] < b['mean_nmse'] and c['mean_si_sdr'] > b['mean_si_sdr']
               for c, b in zip(candidate['by_count'][1:], baseline['by_count'][1:]))


def audit(run, output):
    state = watch.snapshot(run)
    if state['status'] != 'COMPLETED':
        raise ValueError('All registered training must finish first')
    plan = watch.read(run/'PROTOCOL.json')
    complete = watch.read(run/'COMPLETE.json')
    protocol_sha = watch.digest(run/'PROTOCOL.json')
    if complete['protocol_sha256'] != protocol_sha or complete['updates_per_arm'] != 375:
        raise ValueError('Completion protocol/budget mismatch')
    torch.set_num_threads(2)
    checkpoints = []
    for arm, capacity in plan['arms'].items():
        folder = run/arm
        history = state['histories'][arm]
        for epoch in range(1, 6):
            chosen = min(history[:epoch+1], key=lambda x: x['selection_nmse'])
            path = folder/f'SELECTED_{epoch:03d}.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if (saved['protocol_sha256'] != protocol_sha or saved['arm'] != arm or
                    saved['best']['epoch'] != chosen['epoch'] or
                    not watch.close(saved['best']['metric'], chosen['selection_nmse'])):
                raise ValueError('Selected checkpoint metadata mismatch')
            if sum(t.numel() for t in saved['model'].values()) != capacity:
                raise ValueError('Model size changed')
            if any(not torch.isfinite(t).all() for t in saved['model'].values()):
                raise ValueError('Nonfinite selected weights')
            checkpoints.append(dict(arm=arm, prefix_epoch=epoch, selected_epoch=chosen['epoch'],
                                    sha256=watch.digest(path)))
            del saved
            gc.collect()
        if watch.digest(folder/'BEST.pt') != checkpoints[-1]['sha256']:
            raise ValueError('Final selected model is not BEST')
        last = torch.load(folder/'LAST.pt', map_location='cpu', weights_only=False)
        if last['epoch'] != 5 or last['updates'] != 375 or last['protocol_sha256'] != protocol_sha:
            raise ValueError('Resume checkpoint incomplete')
        if sum(t.numel() for t in last['model'].values()) != capacity:
            raise ValueError('Resume capacity mismatch')
        if any(not torch.isfinite(t).all() for t in last['model'].values()):
            raise ValueError('Nonfinite last weights')
        if any(int(v['step']) != 375 for v in last['optimizer']['state'].values()):
            raise ValueError('Optimizer step mismatch')
        del last
        gc.collect()
    for epoch in range(1, 6):
        common = watch.read(run/f'COMMON_EPOCH_{epoch:03d}.json')
        if common['epochs_per_arm'] != epoch or common['updates_per_arm'] != 75*epoch:
            raise ValueError('Common epoch budget mismatch')
        for arm in plan['arms']:
            chosen = min(state['histories'][arm][:epoch+1], key=lambda x: x['selection_nmse'])
            if common['selected'][arm]['epoch'] != chosen['epoch']:
                raise ValueError('Common selection mismatch')
    selected = {}
    for arm in plan['arms']:
        best = state['common_selected'][arm]
        selected[arm] = dict(epoch=best['epoch'], **summarize(run/arm/f"VALIDATION_{best['epoch']:03d}.json"))
    result = dict(status='PASS', independent_test=False, heldout_read=False,
        protocol_sha256=protocol_sha, epochs_per_arm=5, updates_per_arm=375,
        parameters=plan['arms'], checkpoints=checkpoints, selected=selected,
        history=state['histories'],
        training_seconds={arm: sum(e['train_seconds'] for e in state['events'] if e['arm']==arm)
                          for arm in plan['arms']},
        cycle15_over_cycle10_joint_improvement=better(selected['wavenet_cycle15'], selected['wavenet_cycle10']),
        wave10_over_unet_joint_improvement=better(selected['wavenet_cycle10'], selected['unet_mean']),
        wave15_over_unet_joint_improvement=better(selected['wavenet_cycle15'], selected['unet_mean']),
        audit_sources={p.name:watch.digest(p) for p in (Path(__file__), Path(watch.__file__))},
        timestamp=time.time())
    watch.write(output.with_suffix('.json'), result)
    lines = ['# 원 규모 WaveNet·U-Net: 동일 학습량 구조 비교 완료', '',
        'RFUAV 단일 데이터셋·동일 RF 대역·원 수신 중심주파수 차이 유지. '
        '각각 새 초기화 seed 0, 동일12,000학습 혼합·5epoch·375업데이트, '
        '같은630개발검증·같은 최솟값 선택 규칙을 사용했다. '
        '모델 크기와 계산량은 다르고 각 계열의 최적 학습률·수렴 성능을 비교한 실험은 아니다.', '',
        '| 모델 | 파라미터 수 | 선택 epoch | NMSE 2개 ↓ | NMSE 3개 ↓ | SI-SDR 2개 ↑ dB | SI-SDR 3개 ↑ dB | 약신호 NMSE 2/3 ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for arm, value in selected.items():
        a,b=value['by_count'][1:]
        lines.append(f"| {watch.NAMES[arm]} | {plan['arms'][arm]:,} | {value['epoch']} | "
            f"{watch.fmt(a['mean_nmse'],6)} | {watch.fmt(b['mean_nmse'],6)} | "
            f"{watch.fmt(a['mean_si_sdr'],3)} | {watch.fmt(b['mean_si_sdr'],3)} | "
            f"{watch.fmt(a['weakest_nmse'],6)} / {watch.fmt(b['weakest_nmse'],6)} |")
    lines += ['', '두 혼합 개수에서 NMSE 감소와 복소 SI-SDR 상승을 모두 요구하는 등록 기준:', '',
        f"- 긴 수용범위 WaveNet 대 기본 WaveNet: {result['cycle15_over_cycle10_joint_improvement']}",
        f"- 기본 WaveNet 대 U-Net: {result['wave10_over_unet_joint_improvement']}",
        f"- 긴 수용범위 WaveNet 대 U-Net: {result['wave15_over_unet_joint_improvement']}", '',
        'WaveNet 두 군은 같은4,601,867개 초기 파라미터에서 팽창률만 변경했다. '
        '모든 모델의 긴20.89ms 보조 특징은 시간 평균을 사용하므로 긴 시간 순서 활용의 대조는 아니다. '
        '새 초기화 비교를 이전부터 학습한 U-Net의 성능과 구조만의 효과로 직접 비교하지 않는다.', '',
        '세 군의초기+5epoch 검증630행, 소스/준비 명세 해시, 모델 크기·유한 가중치, '
        '375optimizer 업데이트, 모든 선택 checkpoint와 동일 예산 선택을 검사했다. '
        '미학습 기종 확인 자료는 열지 않았다. 한seed·반복 개발 검증이며 독립 시험이나 물리 드론 대수 검증은 아니다.', '',
        '[모든 완료 epoch 기록](ARCHITECTURE_PROGRESS.md). 선택된epoch의 기종 조합·국소 전력차별 결과는 동반JSON에 포함한다.', '']
    watch.write(output.with_suffix('.md'), '\n'.join(lines))
    print(json.dumps({k:result[k] for k in ('status','cycle15_over_cycle10_joint_improvement',
                       'wave10_over_unet_joint_improvement','wave15_over_unet_joint_improvement')}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    try:
        if args.wait:
            while not (args.run/'COMPLETE.json').exists():
                if (args.run/'FAILURE.json').exists():
                    raise RuntimeError('Training failed; audit not executed')
                current=watch.read(args.run/'STATE.json')
                if not watch.live_process(current):
                    raise RuntimeError('Training worker absent; audit not executed')
                time.sleep(30)
        audit(args.run.resolve(),args.output.resolve())
    except Exception as exc:
        watch.write(args.output.with_name(args.output.name+'_AUDIT_FAILURE.json'),
                    dict(error=repr(exc),time=time.time()))
        raise
