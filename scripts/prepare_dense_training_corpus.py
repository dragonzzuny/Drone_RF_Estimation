"""Cover the existing RFUAV training recordings without changing source roles.

CPU-only preparation of a successor corpus, not live training admission. The
last context overlaps its predecessor when a file has a partial final block.
Validation caches stay identical; held-out IQ is never opened.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

from drone_rf.data import DEVELOPMENT_CATEGORIES, sha256
from drone_rf.similarity import shifted_coherence

LENGTH = 2097152


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def dense_offsets(samples, length=LENGTH):
    if length < 1 or samples < length:
        raise ValueError('Source must contain a complete context')
    offsets = list(range(0, samples - length + 1, length))
    if offsets[-1] + length < samples:
        offsets.append(samples - length)
    return offsets


def training_sources(inventory, clips):
    """Only original train sources already in the sealed parent are eligible."""
    roles, parents = {}, defaultdict(list)
    for clip in clips:
        if clip['role'] not in ('train_pack', 'validation_pack') or clip['category'] not in DEVELOPMENT_CATEGORIES:
            raise ValueError('Unexpected parent role/category')
        if roles.setdefault(clip['pack_id'], clip['role']) != clip['role']:
            raise ValueError('Parent pack crosses roles')
        parents[clip['source_path']].append(clip)
    expected = {c['source_path'] for c in clips if c['role'] == 'train_pack'}
    sources = []
    for row in inventory['files']:
        if row['relative_path'] not in expected:
            continue
        if row['role'] != 'train_pack' or row['category'] not in DEVELOPMENT_CATEGORIES:
            raise ValueError('A parent training source changed role/category')
        if row['data_type'] != 'Complex Float' or row['fs_hz'] != 100000000 or row['bytes'] != row['samples_cf32'] * 8:
            raise ValueError('Unexpected native IQ geometry')
        for clip in parents[row['relative_path']]:
            if any(clip[k] != row[k] for k in ('role', 'category', 'pack_id', 'fs_hz', 'center_hz')):
                raise ValueError('Parent source metadata mismatch')
        hashes = {c['source_sha256'] for c in parents[row['relative_path']]}
        if len(hashes) != 1:
            raise ValueError('Inconsistent prior whole-file hashes')
        offsets = dense_offsets(row['samples_cf32'])
        sources.append(dict(row, expected_sha256=hashes.pop(), offsets=offsets))
    if not sources or len(sources) != len(expected) or len({r['relative_path'] for r in sources}) != len(expected):
        raise ValueError('Missing or duplicated parent training sources')
    return sorted(sources, key=lambda r: r['relative_path'])


def stream_contexts(path, samples, length, consume):
    """Hash every raw byte once and emit byte-exact contexts with bounded RAM."""
    expected = dense_offsets(samples, length)
    block = length * 8
    digest, emitted, cursor, previous = hashlib.sha256(), [], 0, b''
    with path.open('rb') as stream:
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        while raw := stream.read(block):
            digest.update(raw)
            if len(raw) == block:
                offset, context = cursor // 8, raw
            else:
                if not previous or len(raw) % 8:
                    raise ValueError('Incomplete complex source/context')
                offset, context = (cursor + len(raw) - block) // 8, (previous + raw)[-block:]
            if offset not in expected or len(emitted) >= len(expected) or offset != expected[len(emitted)]:
                raise ValueError('Raw source length changed')
            consume(offset, context)
            emitted.append(offset)
            cursor += len(raw)
            previous = raw
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(stream.fileno(), cursor - len(raw), len(raw), os.POSIX_FADV_DONTNEED)
    if cursor != samples * 8 or emitted != expected:
        raise ValueError('Short/changed raw source')
    return digest.hexdigest()


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return 0


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'RUN.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'CONFIG.json').exists():
            raise RuntimeError('Use a fresh output; do not overwrite a partial corpus')
        args.cache.mkdir(parents=True, exist_ok=False)
        os.sched_setaffinity(0, {14, 15})
        os.nice(10)
        freeze = json.loads((args.snapshot / 'FREEZE.json').read_text())
        for path, digest in freeze['files'].items():
            if sha256(path) != digest:
                raise ValueError('Preparation snapshot changed')
        inventory = json.loads(args.inventory.read_text())
        parent = json.loads(args.parent_manifest.read_text())
        if parent['inventory_sha256'] != sha256(args.inventory):
            raise ValueError('Inventory differs from sealed parent corpus')
        sources = training_sources(inventory, parent['clips'])
        validation = [c for c in parent['clips'] if c['role'] == 'validation_pack']
        expected_clips = sum(len(r['offsets']) for r in sources)
        expected_bytes = expected_clips * (LENGTH * 8 + 128)
        if shutil.disk_usage(args.cache).free < expected_bytes + 20 * 1024**3:
            raise RuntimeError('Insufficient cache space with 20 GiB reserve')
        started = time.time()
        config = dict(status='DENSE_TRAIN_ONLY_PREPARATION', started=started, pid=os.getpid(),
            parent_manifest_sha256=sha256(args.parent_manifest), inventory_sha256=sha256(args.inventory),
            snapshot_sha256=sha256(args.snapshot / 'FREEZE.json'),
            source_files=len(sources), source_packs=len({r['pack_id'] for r in sources}),
            training_categories=sorted({r['category'] for r in sources}),
            source_bytes=sum(r['bytes'] for r in sources), source_seconds=sum(r['samples_cf32'] / r['fs_hz'] for r in sources),
            old_training_clips=sum(c['role'] == 'train_pack' for c in parent['clips']),
            old_cached_training_seconds=sum(c['samples'] / c['fs_hz'] for c in parent['clips'] if c['role'] == 'train_pack'),
            expected_training_clips=expected_clips, clip_samples=LENGTH, expected_new_cache_bytes=expected_bytes,
            sampling='contiguous full contexts; final full context covers remaining tail with overlap',
            unique_training_sample_coverage=1., original_roles_changed=False, new_independent_recordings=0,
            validation_cache_count=len(validation), validation_cache_policy='unchanged parent entries',
            heldout_payload_read=False, gpu_used=False, live_training_modified=False,
            training_admitted=False, similarity_probe_samples=8192, similarity_max_lag=1024,
            similarity_flag_rho_squared=.98)
        write(args.output / 'CONFIG.json', config)
        train, completed, last_update = [], [], 0.

        def progress(stage, force=False, **details):
            nonlocal last_update
            now = time.time()
            if not force and now - last_update < 3:
                return
            write(args.output / 'PROGRESS.json', dict(stage=stage, time=now, pid=os.getpid(),
                elapsed_seconds=now-started, completed_sources=len(completed), total_sources=len(sources),
                cached_training_clips=len(train), expected_training_clips=expected_clips,
                gpu_used=False, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **details))
            last_update = now

        progress('STREAM_TRAINING_RAW', True)
        for source in sources:
            while available_memory() < 2 * 1024**3:
                progress('WAIT_MEMORY', True)
                time.sleep(10)
            raw_path = Path(source['path'])
            before = raw_path.stat()
            if (before.st_size, before.st_mtime_ns) != (source['bytes'], source['mtime_ns']):
                raise ValueError('Training source changed since inventory')
            source_clips = []

            def consume(offset, raw):
                data = np.frombuffer(raw, dtype='<c8')
                if len(data) != LENGTH or not np.isfinite(data).all():
                    raise ValueError('Invalid native complex context')
                clip_id = hashlib.sha256(f'{source["relative_path"]}:{offset}:{LENGTH}'.encode()).hexdigest()[:24]
                target = args.cache / (clip_id + '.npy')
                if target.exists():
                    raise FileExistsError(target)
                temp = target.with_suffix('.npy.tmp')
                with temp.open('wb') as output:
                    np.save(output, data, allow_pickle=False)
                os.replace(temp, target)
                complex128 = data.astype(np.complex128)
                power = np.abs(complex128) ** 2
                mean_power = float(power.mean())
                c = dict(clip_id=clip_id, cache_path=str(target), source_path=source['relative_path'],
                    source_sha256=source['expected_sha256'], pack_id=source['pack_id'], category=source['category'],
                    role='train_pack', fs_hz=source['fs_hz'], center_hz=source['center_hz'],
                    offset_samples=offset, samples=LENGTH, cache_sha256=sha256(target),
                    raw_clip_sha256=hashlib.sha256(raw).hexdigest(), mean_power=mean_power,
                    dc_power_fraction=float(abs(complex128.mean())**2 / max(mean_power,1e-30)),
                    local_power_quantiles=np.quantile(power.reshape(-1,16384).mean(1), [.05,.5,.95]).tolist(),
                    normalization_applied=False, resampling_applied=False, finite=True,
                    positive_power=bool(mean_power > 0))
                train.append(c)
                source_clips.append(clip_id)
                progress('STREAM_TRAINING_RAW', current_file=source['relative_path'], current_offset_samples=offset)

            digest = stream_contexts(raw_path, source['samples_cf32'], LENGTH, consume)
            after = raw_path.stat()
            if digest != source['expected_sha256'] or (after.st_size,after.st_mtime_ns) != (before.st_size,before.st_mtime_ns):
                raise ValueError('Whole training source changed; successor corpus stays unadmitted')
            record = dict(path=source['relative_path'], pack_id=source['pack_id'], role='train_pack',
                bytes=source['bytes'], sha256=digest, clips=source_clips, all_source_samples_covered=True)
            completed.append(record)
            with (args.output / 'SOURCE_RECORDS.jsonl').open('a') as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            write(args.output / 'CACHE_MANIFEST.json', dict(status='BUILDING_DENSE_TRAIN_CORPUS',
                clips=train+validation, inventory_sha256=config['inventory_sha256'], training_admitted=False))
            progress('STREAM_TRAINING_RAW', True)
            print(json.dumps(dict(files=len(completed), total=len(sources), clips=len(train))), flush=True)
        if len(train) != expected_clips:
            raise ValueError('Context count differs from plan')
        progress('CHECK_CROSS_ROLE_SIMILARITY', True)
        # Same limited screen as the parent: not a proof of independent sessions.
        validation_probes = []
        for clip in validation:
            if sha256(clip['cache_path']) != clip['cache_sha256']:
                raise ValueError('Parent validation cache changed')
            validation_probes.append(np.load(clip['cache_path'],mmap_mode='r',allow_pickle=False)[:8192].copy())
        best = [None] * len(validation)
        flagged, pairs, maximum = [], 0, 0.
        for clip in train:
            signal = np.load(clip['cache_path'], mmap_mode='r', allow_pickle=False)[:8192].copy()
            for i, reference in enumerate(validation_probes):
                stat = shifted_coherence(signal, reference, 1024)
                pairs += 1
                maximum = max(maximum, stat['rho_squared'])
                row = dict(train_clip=clip['clip_id'], validation_clip=validation[i]['clip_id'], **stat)
                if best[i] is None or stat['rho_squared'] > best[i]['rho_squared']:
                    best[i] = row
                if stat['rho_squared'] >= .98:
                    flagged.append(row)
            progress('CHECK_CROSS_ROLE_SIMILARITY', compared_pairs=pairs, total_pairs=len(train)*len(validation))
        raw_hashes = defaultdict(list)
        for clip in train+validation:
            raw_hashes[clip['raw_clip_sha256']].append(clip)
        crossing = [[dict(clip_id=c['clip_id'],role=c['role']) for c in group]
                    for group in raw_hashes.values() if len({c['role'] for c in group}) > 1]
        similarity = dict(pairs=pairs, max_coarse_rho_squared=maximum, flags=flagged,
            best_per_validation=best, exact_cross_role_clip_copies=crossing,
            probe_samples=8192, max_lag=1024, validation_cached_iq_read=True,
            validation_raw_or_heldout_iq_read=False, independent_recordings_proven=False)
        write(args.output / 'SIMILARITY.json', similarity)
        for path, digest in freeze['files'].items():
            if sha256(path) != digest:
                raise ValueError('Preparation source changed')
        if sha256(args.parent_manifest) != config['parent_manifest_sha256'] or sha256(args.inventory) != config['inventory_sha256']:
            raise ValueError('Parent metadata changed')
        manifest = dict(status='DENSE_TRAIN_CACHE_COMPLETE_REVIEW_REQUIRED', clips=train+validation,
            inventory_sha256=config['inventory_sha256'], parent_manifest_sha256=config['parent_manifest_sha256'],
            source_count=len({c['source_path'] for c in train+validation}),
            training_source_bytes_rehashed=config['source_bytes'], exact_cross_role_duplicate_groups=len(crossing),
            training_admitted=False, live_training_modified=False, dense_training_sampling=config['sampling'])
        write(args.output / 'CACHE_MANIFEST.json', manifest)
        zero_power = [c['clip_id'] for c in train if not c['positive_power']]
        summary = dict(status='DENSE_TRAIN_CORPUS_READY_FOR_REVIEW' if not flagged and not crossing and not zero_power else 'DENSE_TRAIN_CORPUS_REQUIRES_REVIEW',
            time=time.time(), elapsed_seconds=time.time()-started, training_clips=len(train),
            validation_clips=len(validation), original_training_files=len(sources), original_training_packs=config['source_packs'],
            unique_training_seconds=config['source_seconds'], unique_training_sample_coverage=1.,
            new_independent_recordings=0, max_coarse_rho_squared=maximum, similarity_pairs=pairs,
            similarity_flags=len(flagged), exact_cross_role_clip_copies=len(crossing),
            zero_power_contexts=zero_power, zero_power_note='Retained as recorded; not valid for per-source power normalization.',
            heldout_iq_read=False, validation_raw_iq_read=False, validation_cache_iq_read=True,
            gpu_used=False, live_training_modified=False, training_admitted=False,
            files={name:sha256(args.output/name) for name in ('CONFIG.json','CACHE_MANIFEST.json','SIMILARITY.json','SOURCE_RECORDS.jsonl')},
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(args.output / 'COMPLETE.json', summary)
        progress('COMPLETE', True)
        print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('inventory','parent-manifest','output','cache','snapshot'):
        parser.add_argument('--'+arg,type=Path,required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'FAILURE.json',dict(time=time.time(),pid=os.getpid(),traceback=traceback.format_exc(),
            live_training_modified=False,training_admitted=False))
        raise
