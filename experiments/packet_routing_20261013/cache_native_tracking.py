"""Reuse frozen native e2 C24 waveforms; separate observed features and scoring."""
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from evaluate_parent_full_context import ROOT, read, write, sha
from tracking_cache import save_case
from routing_statistics import score
from diagnose_count_routing import verify_metrics
from native_full_context import components

OUT = ROOT/'local/native_tracking_cache_20261013_v1'
PARENT = ROOT/'local/incumbent_tracking_c24_20261013_v1'
HERE = Path(__file__).resolve().parent


def state(status, **kw):
    write(OUT/'STATE.json', dict(status=status, time=time.time(), pid=os.getpid(), **kw))


def run():
    assert read(PARENT/'STATE.json')['status'] == 'COMPLETE_AUDITED'
    assert sha(PARENT/'RESULT.json') == read(PARENT/'AUDIT.json')['result_sha256']
    previous = read(PARENT/'RESULT.json')
    sources = dict(read(PARENT/'PLAN.json')['source_sha256'])
    for path in (Path(__file__), ROOT/'reports/2026-10-13/NATIVE_TRACKING_PLAN_KO.md'):
        sources[str(path.relative_to(ROOT))] = sha(path)
    inputs = {str(p): sha(p) for p in (PARENT/'RESULT.json', PARENT/'AUDIT.json', PARENT/'PLAN.json')}
    assert not (OUT/'PREPARATION.json').exists()
    write(OUT/'PREPARATION.json', dict(created=time.time(), source_sha256=sources, input_sha256=inputs,
        cases=24, group='C24', scope='Previously exposed native e2 development; final50 only', heldout_access=False))
    for name, digest in sources.items():
        assert sha(ROOT/name) == digest
        dest = OUT/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        assert sha(dest) == digest
    sys.path.insert(0, str(ROOT/'experiments/rfuav_native_frequency_20261009'))
    from native_data import NativeMixtures
    phase = read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')
    data = NativeMixtures(Path(phase['preparation']), 'validation_pack', 1)
    assert data.rows_hash == phase['dev_rows_sha256']
    dest = OUT/'C24'
    dest.mkdir()
    items = []
    for position, case in enumerate(previous['cases']):
        index = case['index']
        state('CPU_PREPARATION', case=position+1, total=24)
        path = PARENT/f'OUTPUT_{index:03d}.npy'
        assert sha(path) == case['output_sha256']
        output = np.load(path, mmap_mode='r', allow_pickle=False)
        refs = components(data, index)
        baseline = case['metrics']['parent']
        metadata = dict(categories=case['categories'], nominal_levels_db=case['nominal_levels_db'],
            parent_output=str(path), parent_output_sha256=sha(path))
        item = save_case(dest, index, output, refs, baseline['count'], baseline['predicted_count'], metadata)
        verify_metrics(item['baseline_metrics'], baseline)
        with np.load(item['scoring'], allow_pickle=False) as z:
            psd = score(z, np.asarray(case['routing']['psd']['orders']), item['count'], item['predicted_count'])
        verify_metrics(psd, case['metrics']['psd'])
        item['constrained_baseline'] = psd
        item['transferred_B99_tracker'] = case['metrics']['learned50']
        items.append(item)
        del output, refs
        gc.collect()
    assert len(items) == 24
    assert all(sha(ROOT/p) == h for p, h in sources.items())
    assert all(sha(p) == h for p, h in inputs.items())
    result = dict(status='VERIFIED', groups={'C24': items}, cases=24,
        source_sha256=sources, preparation_sha256=sha(OUT/'PREPARATION.json'),
        references_separate_from_observed_features=True, optimizer_updates=0, heldout_access=False)
    write(OUT/'RESULT.json', result)
    write(OUT/'AUDIT.json', dict(status='PASS', result_sha256=sha(OUT/'RESULT.json'),
        all24_parent_and_psd_metrics_reproduced=True))
    write(ROOT/'reports/2026-10-13/NATIVE_TRACKING_CACHE_CHECK.json', dict(status='PASS', cases=24,
        result_sha256=sha(OUT/'RESULT.json'), preparation_sha256=sha(OUT/'PREPARATION.json')))
    state('COMPLETE_AUDITED', cases=24)


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        write(OUT/'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        state('FAILED')
        raise
