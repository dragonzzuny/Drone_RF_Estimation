"""Wait for the frozen phase-context study, then audit completed CPU artifacts.

No training, I/Q reads, checkpoint selection changes, or held-out access.
Torch is imported only once training has completed to bound idle memory use.
"""
import argparse
import gc
import os
from pathlib import Path
import shutil
import time

import phase_watch
import watch_epochs as watch


def audit(root, output):
    import torch
    torch.set_num_threads(2)
    state = phase_watch.snapshot(root)
    if state['status'] != 'COMPLETED' or state['common_epoch'] != 5:
        raise ValueError('The complete registered study is required')
    protocol = state['protocol_sha256']
    records = []
    for arm, capacity in state['parameters'].items():
        folder = root / arm
        history = state['histories'][arm]
        for epoch in range(1, 6):
            chosen = min(history[:epoch + 1], key=lambda r: r['selection_nmse'])
            for kind in ('ACTUAL', 'SELECTED'):
                path = folder / f'{kind}_{epoch:03d}.pt'
                saved = torch.load(path, map_location='cpu', weights_only=False)
                if saved['arm'] != arm or saved['protocol_sha256'] != protocol:
                    raise ValueError('Wrong checkpoint identity')
                if kind == 'ACTUAL':
                    if saved['epoch'] != epoch or saved['updates'] != 75 * epoch:
                        raise ValueError('Wrong actual checkpoint budget')
                elif (saved['best']['epoch'] != chosen['epoch'] or
                      not watch.close(saved['best']['metric'], chosen['selection_nmse'])):
                    raise ValueError('Selected checkpoint differs from registered rule')
                if (sum(t.numel() for t in saved['model'].values()) != capacity or
                        any(not torch.isfinite(t).all() for t in saved['model'].values())):
                    raise ValueError('Wrong capacity/nonfinite checkpoint weights')
                records.append(dict(arm=arm, kind=kind, prefix_epoch=epoch,
                    selected_epoch=chosen['epoch'] if kind == 'SELECTED' else epoch,
                    sha256=watch.digest(path)))
                del saved
                gc.collect()
        if watch.digest(folder / 'BEST.pt') != watch.digest(folder / 'SELECTED_005.pt'):
            raise ValueError('Final selected checkpoint differs from BEST')
        saved = torch.load(folder / 'LAST.pt', map_location='cpu', weights_only=False)
        if (saved['protocol_sha256'] != protocol or saved['arm'] != arm or
                saved['epoch'] != 5 or saved['updates'] != 375):
            raise ValueError('Incomplete resume checkpoint')
        if (not saved['optimizer']['state'] or
                any(int(v['step']) != 375 for v in saved['optimizer']['state'].values())):
            raise ValueError('Optimizer did not complete registered updates')
        if any(not torch.isfinite(t).all() for t in saved['model'].values()):
            raise ValueError('Nonfinite resume model')
        actual = torch.load(folder / 'ACTUAL_005.pt', map_location='cpu', weights_only=False)
        if (saved['model'].keys() != actual['model'].keys() or
                any(not torch.equal(t, actual['model'][k]) for k, t in saved['model'].items())):
            raise ValueError('Resume and actual final model differ')
        del saved, actual
        gc.collect()
    selected = state['common_selected']
    local, long = selected['local']['by_count'][1:], selected['long']['by_count'][1:]
    joint = all(a['mean_si_sdr'] is not None and b['mean_si_sdr'] is not None
                and b['mean_nmse'] < a['mean_nmse'] and b['mean_si_sdr'] > a['mean_si_sdr']
                for a, b in zip(local, long))
    result = dict(status='PASS', protocol_sha256=protocol, selected=selected,
        histories=state['histories'], checkpoints=records, epochs_per_arm=5,
        updates_per_arm=375, parameters=state['parameters'], long_over_local_joint_improvement=joint,
        heldout_read=False, independent_test=False, new_iq_read=False,
        observer_source_sha256=watch.digest(Path(phase_watch.__file__)),
        audit_source_sha256=watch.digest(Path(__file__)), time=time.time())
    watch.write(output.with_suffix('.json'), result)
    text = ['# 긴 복소 I/Q 문맥 비교: 완료 검산', '',
        '같은 RFUAV 원 대역·고정 원기록 분할·새 초기값·4,641,795파라미터·각 5 epoch/375업데이트. '
        '개발 630혼합의 2/3성분 평균 NMSE 최소 규칙으로 선택했다. '
        'local도 전체 RMS·시간 평균 전력 특징을 받으며 구간 밖 복소 표본만 가린다.', '',
        '|군|선택epoch|NMSE 2/3↓|복소 SI-SDR 2/3↑ dB|약신호 NMSE 2/3↓|',
        '|---|---:|---|---|---|']
    for arm, metrics in selected.items():
        a, b = metrics['by_count'][1:]
        pair = lambda key: f"{watch.fmt(a[key],6)}/{watch.fmt(b[key],6)}"
        text.append(f"|{arm}|{metrics['epoch']}|{pair('mean_nmse')}|{pair('mean_si_sdr')}|{pair('weakest_nmse')}|")
    text += ['', f'두 혼합 개수에서 NMSE 감소와 SI-SDR 상승을 모두 충족: **{joint}**.',
        '초기+각 5 epoch의 검증 행·집계·소스 해시, 실제/선택 가중치 20개, 최종 375업데이트·'
        '재시작 상태와 마지막 모델의 일치를 검사했다. 한 seed의 개발 비교이며 독립 시험·'
        '미학습 기종 일반화·물리 드론 대수·완전 수렴을 입증하지 않는다. '
        '약신호·개수 실패는 전체 epoch 결과와 함께 보고한다.', '',
        '[전체 epoch와 진행](PHASE_CONTEXT_PROGRESS.md).', '']
    watch.write(output.with_suffix('.md'), '\n'.join(text))
    print(f'PASS long_over_local_joint_improvement={joint}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    try:
        snapshot = args.run / 'completion_auditor_source'
        snapshot.mkdir(exist_ok=True)
        modules = (Path(__file__), Path(phase_watch.__file__), Path(watch.__file__))
        expected = {m.name: watch.digest(m) for m in modules}
        for module in modules:
            dest = snapshot / module.name
            if dest.exists() and watch.digest(dest) != expected[module.name]:
                raise ValueError('Auditor source snapshot changed')
            if not dest.exists():
                shutil.copyfile(module, dest)
        if args.wait:
            while not (args.run / 'COMPLETE.json').exists():
                state = phase_watch.snapshot(args.run)
                if state['status'] in ('FAILED', 'WORKER_NOT_RUNNING'):
                    raise RuntimeError('Training did not complete: ' + state['status'])
                if any(watch.digest(m) != expected[m.name] for m in modules):
                    raise ValueError('Live auditor source changed')
                time.sleep(30)
        audit(args.run.resolve(), args.output.resolve())
    except Exception as exc:
        watch.write(args.output.with_name(args.output.name + '_AUDIT_FAILURE.json'),
                    dict(error=repr(exc), time=time.time(), pid=os.getpid()))
        raise
