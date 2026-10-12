"""Fixed final tracker transfer to the existing C630 incumbent; no training."""
import fcntl
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from evaluate_parent_full_context import ROOT, read, write, sha, LOCK

OUT = ROOT/'local/incumbent_tracking_c24_20261013_v1'
TRACK = ROOT/'local/learned_tracking_20261013_v1'
PHASE = ROOT/'local/phase_eight_20261011_v1'
HERE = Path(__file__).resolve().parent


def state(status, **kw):
    write(OUT/'STATE.json', dict(status=status, pid=os.getpid(), time=time.time(), **kw))


def run():
    OUT.mkdir(exist_ok=True)
    assert read(TRACK/'STATE.json')['status'] == 'COMPLETE_AUDITED'
    assert read(ROOT/'local/tracking_direct_audit_20261013_v1/AUDIT.json')['status'] == 'PASS'
    phase = read(PHASE/'PROTOCOL.json')
    checkpoint = Path(phase['checkpoint'])
    assert sha(checkpoint) == phase['checkpoint_sha256']
    sources = dict(read(TRACK/'PROTOCOL.json')['source_sha256'])
    for name, digest in read(ROOT/'local/spectral_routing_c24_20261013_v1/PLAN.json')['source_sha256'].items():
        if name in sources:
            assert sources[name] == digest
        sources[name] = digest
    for p in (Path(__file__), ROOT/'reports/2026-10-13/INCUMBENT_TRACKING_PLAN_KO.md'):
        sources[str(p.relative_to(ROOT))] = sha(p)
    files = {str(p): sha(p) for p in (checkpoint, TRACK/'ACTUAL_050.pt', TRACK/'RESULT.json',
        TRACK/'PROTOCOL.json', PHASE/'PROTOCOL.json', PHASE/'COMPLETE.json',
        ROOT/'local/tracking_direct_audit_20261013_v1/RESULT.json')}
    indices = read(ROOT/'reports/2026-10-13/NATIVE_FULL_CONTEXT_CPU_CHECK.json')['indices']
    plan = dict(registered=time.time(), source_sha256=sources, input_sha256=files,
        indices=indices, methods=['parent', 'psd', 'learned50'], optimizer_updates=0,
        samples=2088960, fs_hz=100000000, checkpoint= str(checkpoint),
        tracker_fixed_epoch=50, heldout_access=False, references_in_routing=False,
        scope='Previously exposed C24, unchanged incumbent and transferred B-trained tracking; no new final test')
    assert not (OUT/'PLAN.json').exists()
    write(OUT/'PLAN.json', plan)
    for name, digest in sources.items():
        assert sha(ROOT/name) == digest
        target = OUT/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
        assert sha(target) == digest
    with LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK')
        fcntl.flock(lock, fcntl.LOCK_EX)
        import torch
        from sequence_data import ROLES
        from models import build, predict
        from study import batch
        from sequence_eval import predict_eight, score as gpu_score
        from full_context_eval import evaluate
        from native_full_context import components, selected_indices
        from tracking_cache import observed_features
        from learned_tracking_model import TrackingEmbedding
        from spectral_routing import PERMS, profiles
        from spectral_routing_constrained import fit
        from audit_learned_tracking import apply_orders
        from numpy_waveform_score import score
        from diagnose_count_routing import verify_metrics, summary
        sys.path.insert(0, str(ROOT/'experiments/rfuav_native_frequency_20261009'))
        from native_data import NativeMixtures
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        data = NativeMixtures(Path(phase['preparation']), 'validation_pack', 1)
        assert data.rows_hash == phase['dev_rows_sha256']
        assert selected_indices(data) == indices
        net = build('unet_mean').cuda().eval().requires_grad_(False)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        net.load_state_dict(saved['model'])
        del saved
        tracker = TrackingEmbedding().eval().requires_grad_(False)
        saved = torch.load(TRACK/'ACTUAL_050.pt', map_location='cpu', weights_only=False, mmap=True)
        assert saved['epoch'] == 50 and saved['updates'] == 600
        tracker.load_state_dict(saved['model'])
        del saved
        # Reproduce one original short case per count before whole-window inference.
        prior = read(PHASE/'COMPLETE.json')['eight_phase']
        expected = prior['rows']
        preflight = []
        for k in (1, 2, 3):
            index = next(i for i in indices if int(data.rows[i]['count']) == k)
            item = batch(data[index], 'cuda')
            with torch.no_grad():
                _, y, logits, _ = predict_eight(net, predict, item)
            actual = gpu_score(y, item['references'], item['mixture'], k, logits)
            verify_metrics(actual, expected[index])
            preflight.append(dict(index=index, count=k, metrics=actual))
        write(OUT/'PREFLIGHT.json', dict(status='PASS', cases=preflight))
        del item, y, logits
        rows = []
        for position, index in enumerate(indices):
            began = time.time()
            raw = data.rows[index]
            k = int(raw['count'])
            refs = components(data, index)
            def progress(done, total):
                state('FULL_INFERENCE', case=position+1, total_cases=24,
                    index=index, windows=done, total_windows=total)
            output, case = evaluate(net, predict, refs, np.array(data.features[index]), k, 'cuda', progress)
            predicted = case['metrics']['predicted_count']
            assert predicted == expected[index]['predicted_count']
            output_path = OUT/f'OUTPUT_{index:03d}.npy'
            np.save(output_path, output, allow_pickle=False)
            feature = observed_features(output)
            observed = np.concatenate([feature['vectors'],
                np.log1p(feature['powers']/float(feature['mixture_power']))[..., None].astype(np.float32)], axis=-1)
            with torch.no_grad():
                embedding = tracker(torch.from_numpy(observed)[None])[0].numpy().astype(np.float64)
            vectors, powers, bounds = profiles(output)
            np.testing.assert_array_equal(powers, feature['powers'])
            np.testing.assert_array_equal(bounds, feature['bounds'])
            receipts = {}
            # Routing sees only estimated outputs, observed powers and predicted count.
            for name, v in [('psd', vectors), ('learned50', np.pad(embedding, ((0,0),(0,0),(0,384))))]:
                _, choices, objective = fit(v, powers, predicted)
                receipts[name] = dict(orders=[list(PERMS[int(q)]) for q in choices], objective=objective)
            baseline = score(output, refs, refs.sum(0), k, predicted)
            verify_metrics(baseline, case['metrics'])
            metrics = {'parent': baseline}
            for name in ('psd', 'learned50'):
                routed = apply_orders(output, bounds, receipts[name]['orders'])
                metrics[name] = score(routed, refs, refs.sum(0), k, predicted)
                assert metrics[name]['sum_relative_error'] < 1e-9
                del routed
            result = dict(index=index, count=k, categories=[data.library.clips[int(i)]['category'] for i in raw['indices'][:k]],
                nominal_levels_db=raw['levels'][:k].tolist(), metrics=metrics, routing=receipts,
                output_sha256=sha(output_path), seconds=time.time()-began)
            write(OUT/f'CASE_{index:03d}.json', result)
            rows.append(result)
            print('CASE_COMPLETE', position+1, index, k, {n:m['nmse'] for n,m in metrics.items()}, flush=True)
            del output, refs, feature, observed, embedding, vectors, powers
            gc.collect()
            torch.cuda.empty_cache()
        groups = {name:{str(k):summary([r['metrics'][name] for r in rows if r['count']==k]) for k in (1,2,3)}
            for name in ('parent','psd','learned50')}
        assert all(sha(ROOT/name)==h for name,h in sources.items())
        assert all(sha(name)==h for name,h in files.items())
        result = dict(status='PASS', cases=rows, by_method=groups, plan_sha256=sha(OUT/'PLAN.json'),
            optimizer_updates=0, current_best_changed=False, heldout_access=False, scope=plan['scope'])
        write(OUT/'RESULT.json', result)
        write(OUT/'AUDIT.json', dict(status='PASS', result_sha256=sha(OUT/'RESULT.json'),
            short_original_reproduced=True, parent_gpu_numpy_scores_reproduced=True,
            background_unchanged=True, sources_and_checkpoints_unchanged=True))
        write(ROOT/'reports/2026-10-13/INCUMBENT_TRACKING_RESULT.json', result)
        state('COMPLETE_AUDITED', cases=24)


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        write(OUT/'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        state('FAILED')
        raise
