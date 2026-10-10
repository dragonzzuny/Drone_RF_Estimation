"""Frozen execution order and filesystem helpers; no CUDA initialization."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'reports/2026-10-10'
RUN = ROOT / 'local/five_candidates_20261010_v2'
SEP = ROOT / 'local/septda_continuation_20261010_v1'
ORDER = ('septda', 'frozen_tf', 'ordered_context', 'balanced_head', 'wave_guard')
LABELS = ('SepTDA 참고 U-Net', '본체 고정 시간·주파수 보강', '시간 순서 게이트',
          '특징 규모 보정 성분 상호작용', '파형 업데이트 보호')
SPECS = {
 'frozen_tf': dict(origin='tf_axis_adaptation_20261010_v1', subdir='frozen_backbone',
                   parameters=37406475, trainable=5263616, schedule_start=1),
 'ordered_context': dict(origin='ordered_context_20261010_v1', subdir='',
                        parameters=32142924, trainable=32142924, schedule_start=3),
 'balanced_head': dict(origin=None, subdir='', parameters=32180747,
                      trainable=32180747, schedule_start=3),
 'wave_guard': dict(origin='wave_update_guard_20261010_v1', subdir='',
                   parameters=32142859, trainable=32142859, schedule_start=1),
}
MILESTONES = (5, 10, 20, 30, 40, 50)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def schedule(name, epoch):
    return (SPECS[name]['schedule_start'] + epoch - 2) % 5 + 1


def verify(root, p):
    for rel, sha in p['source_sha256'].items():
        assert digest(ROOT/rel) == digest(root/'source_snapshot'/rel) == sha, rel
    for path, sha in p['pinned_files'].items():
        assert digest(path) == sha, path


def state(root, **values):
    write(root/'STATE.json', dict(values, pid=os.getpid(), time=time.time()))
