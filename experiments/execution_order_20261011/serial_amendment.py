"""Change execution priority only; retain the frozen five-candidate trainer."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / 'experiments/five_candidates_20261010'
sys.path.insert(0, str(LEGACY))
import serial_runner as runner

ORDER = ('septda', 'ordered_context', 'frozen_tf', 'balanced_head', 'wave_guard')
ORIGINAL_ORDER = tuple(runner.ORDER)
LABEL_BY_NAME = dict(zip(runner.ORDER, runner.LABELS))
BASE_WRITE = runner.write
BASE_VERIFY = runner.verify


def validate_amendment(root, amendment):
    a = runner.read(amendment)
    assert a['status'] == 'EXECUTION_ORDER_ONLY'
    assert a['original_order'] == list(ORIGINAL_ORDER)
    assert a['execution_order'] == list(ORDER)
    assert sorted(ORDER) == sorted(ORIGINAL_ORDER)
    assert Path(a['run']).resolve() == root
    assert runner.digest(root/'PROTOCOL.json') == a['protocol_sha256']
    assert runner.read(root/'PROTOCOL.json')['order'] == list(ORIGINAL_ORDER)
    assert runner.digest(Path(__file__)) == a['wrapper_sha256']
    assert Path(a['trainer']).resolve() == LEGACY/'train.py'
    assert runner.digest(a['trainer']) == a['trainer_sha256']
    assert runner.digest(root/'READY.json') == a['ready_sha256']
    assert a['maximum_epochs'] == 50 and a['updates_per_epoch'] == 75
    return a


def configure(root, amendment):
    a = validate_amendment(root, amendment)
    amendment_sha = runner.digest(amendment)
    provenance = dict(execution_order=list(ORDER), original_order=list(ORIGINAL_ORDER),
                      execution_amendment_sha256=amendment_sha,
                      execution_amendment=str(amendment))

    def verify(verified_root, protocol):
        assert verified_root == root
        assert runner.digest(amendment) == amendment_sha
        validate_amendment(root, amendment)
        BASE_VERIFY(verified_root, protocol)

    def write(path, value):
        path = Path(path)
        if path in (root/'QUEUE_STATE.json', root/'COMPLETE.json',
                    runner.PUBLIC/'FIVE_CANDIDATES_PROGRESS.json'):
            value = dict(value, **provenance)
            if value.get('status') == 'WAITING_SEPTDA_50':
                value['next_candidate'] = ORDER[1]
        elif path == runner.PUBLIC/'FIVE_CANDIDATES_PROGRESS_KO.md':
            value += ('\n2026-10-11 실행 순서 변경: SepTDA → 시간 순서 보존 → 본체 고정 '
                      '→ 특징 규모 보정 → 파형 업데이트 보호. 기존 규약·학습 예산은 유지한다. '
                      '[변경 근거](../2026-10-11/EXECUTION_ORDER_KO.md).\n')
        BASE_WRITE(path, value)

    runner.ORDER = ORDER
    runner.LABELS = tuple(LABEL_BY_NAME[name] for name in ORDER)
    runner.write = write
    runner.verify = verify
    return a


def preflight(root, amendment):
    a = configure(root, amendment)
    p = runner.read(root/'PROTOCOL.json')
    runner.verify(root, p)
    ready = runner.read(root/'READY.json')
    assert ready['protocol_sha256'] == a['protocol_sha256']
    for name, sha in ready['preflight_sha256'].items():
        assert runner.digest(root/name/'PREFLIGHT.json') == sha
        assert runner.read(root/name/'PREFLIGHT.json')['status'] == 'PASS'
    assert set(ready['preflight_sha256']) == set(ORDER[1:])
    # The imported runner resolves train.py beside its own frozen __file__.
    assert Path(runner.__file__).with_name('train.py') == Path(a['trainer'])
    return dict(status='PASS', execution_order=list(ORDER),
                completed={n: len(runner.completed(root, n)) for n in ORDER},
                original_sources_verified=len(p['source_sha256']),
                original_pins_verified=len(p['pinned_files']),
                model_preflights_verified=len(ready['preflight_sha256']),
                trainer=a['trainer'], protocol_sha256=a['protocol_sha256'],
                amendment_sha256=runner.digest(amendment), time=time.time(),
                model_training_changed=False, iq_payload_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--amendment', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    root, amendment = args.run.resolve(), args.amendment.resolve()
    if args.preflight_only:
        import json
        result = preflight(root, amendment)
        BASE_WRITE(root/'ORDER_AMENDMENT_CHECK.json', result)
        print(json.dumps(result, ensure_ascii=False))
        return
    configure(root, amendment)
    try:
        runner.run(root, args.python)
    except Exception:
        # A rejected duplicate launcher must not overwrite the active queue.
        pid_file = root/'QUEUE_PID'
        if pid_file.exists() and pid_file.read_text().strip() == str(os.getpid()):
            runner.write(root/'QUEUE_FAILURE.json', dict(
                traceback=traceback.format_exc(), pid=os.getpid(), time=time.time()))
            runner.write(root/'QUEUE_STATE.json', dict(status='FAILED', pid=os.getpid(), time=time.time()))
        raise


if __name__ == '__main__':
    main()
