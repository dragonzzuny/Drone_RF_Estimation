"""One explicitly adaptive TRAIN4 follow-up, not a preregistered main trial.

The previous balanced input probe reduced 9/10 component NMSEs but damaged one
already accurate component. Compare both heads at intermediate new-layer LR,
with exactly the original full-model runner and 64 updates. No heldout reads.
"""
import argparse
import fcntl
from pathlib import Path
import shutil
import time
import traceback
import balanced_head_probe as base

w = base.w


def register(root, dependency, evidence):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Refuse duplicate adaptive probe')
    previous = w.read(dependency/'PROTOCOL.json')
    result = w.read(dependency/'COMPLETE.json')
    checked = w.read(evidence)
    assert checked['status'] == 'PASS' and checked['repeated_control_within_tolerance']
    assert checked['studies']['input_balance']['complete_sha256'] == w.digest(dependency/'COMPLETE.json')
    assert not checked['studies']['input_balance']['all_four_mean_directions_improved']
    assert result['initial_predictions_exactly_equal'] and result['weights_discarded']
    sources = dict(previous['source_sha256'])
    for path in (Path(__file__), Path(base.__file__)):
        sources[str(path.relative_to(w.ROOT))] = w.digest(path)
    protocol = dict(previous)
    protocol.update(status='REGISTERED_ADAPTIVE_INTERMEDIATE_LR_PROBE', source_sha256=sources,
        dependency=str(dependency), dependency_protocol_sha256=w.digest(dependency/'PROTOCOL.json'),
        dependency_complete_sha256=w.digest(dependency/'COMPLETE.json'),
        reviewed_evidence_sha256=w.digest(evidence),
        new_lr=1e-4, old_lr=1e-5, updates_per_arm=64,
        lr_rule='Adaptive after the lr1e-3 TRAIN4 result; same lr1e-4 for both new heads. No optimal-LR claim.',
        purpose='Test whether the observed single-component damage is reduced at a smaller new-head update rate',
        change='Unbalanced versus balanced additional-layer inputs at identical lr1e-4; other settings retained',
        acceptance='Final64 balanced versus unbalanced: lower NMSE and higher complex SI-SDR for BOTH counts; additionally inspect all10 components and compare to the previous lr1e-3 endpoints',
        initialization='Freshly reload identical retained parent; all preceding diagnostic weights discarded',
        limitation='Adaptive hyperparameter diagnosis on repeatedly used TRAIN4, not a confirmatory or generalization experiment',
        registered_at=time.time())
    for rel, sha in sources.items():
        if w.digest(w.ROOT/rel) != sha:
            raise ValueError('Frozen source changed')
        destination = root/'source_snapshot'/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(w.ROOT/rel, destination)
    w.write(root/'PROTOCOL.json', protocol)
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('dependency','evidence','run','public'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        with (root/'.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            p = register(root, args.dependency.resolve(), args.evidence.resolve())
            base.run(root, p, args.public.resolve())
    except Exception:
        w.write(root/'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
