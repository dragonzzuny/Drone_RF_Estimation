"""Recompute the completed fixed-TRAIN diagnosis without new waveform reads."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as stats


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    require(abs(a - b) <= 1e-12 * max(1., abs(a), abs(b)), 'Numerical mismatch')


def run(study, diagnosis, output):
    p = read(study / 'PROTOCOL.json')
    q = read(diagnosis / 'PROTOCOL.json')
    d = read(diagnosis / 'COMPLETE.json')
    require(d['status'] == 'COMPLETE' and not (diagnosis / 'FAILURE.json').exists(), 'Not complete')
    require(d['diagnostic_protocol_sha256'] == digest(diagnosis / 'PROTOCOL.json'), 'Diagnostic protocol changed')
    require(d['study_protocol_sha256'] == q['study_protocol_sha256'] == digest(study / 'PROTOCOL.json'), 'Study changed')
    require(d['source_sha256'] == q['source_sha256'], 'Source registration changed')
    for rel, sha in q['source_sha256'].items():
        require(digest(ROOT / rel) == sha, 'Source changed: ' + rel)
    require(d['model_updates'] == 0 and not d['heldout_read'] and not d['independent_test'], 'Wrong diagnostic scope')
    indices = q['train_indices']
    require(indices == d['train_indices'] and len(set(indices)) == len(indices) == 48, 'TRAIN48 identities')
    models = q['checkpoints']
    require(models == ['parent/e0', 'retained_unet/e1', 'source_interaction/e1'], 'Wrong model set')
    require(len(d['rows']) == 144 and len(d['summaries']) == 9, 'Incomplete result')
    require(len(d['backend_checks']) == 9 and all(c['passed'] for c in d['backend_checks']), 'Backend check failed')
    require(len(d['added_head_weight_changes']) > 0, 'Missing head inspection')
    table = {}
    for model in models:
        rows = [r for r in d['rows'] if r['model'] == model]
        require([r['index'] for r in rows] == indices, 'Different TRAIN inputs')
        arm = model.split('/')[0]
        checkpoint = Path(p['parent_checkpoint']) if arm == 'parent' else study / arm / 'ACTUAL_001.pt'
        validation = Path(p['validation_identity_template']) if arm == 'parent' else study / arm / 'VALIDATION_001.json'
        require(digest(checkpoint) == d['checkpoint_validation_sha256'][model]['checkpoint'], 'Weights changed')
        require(digest(validation) == d['checkpoint_validation_sha256'][model]['validation'], 'Validation changed')
        if arm != 'parent':
            receipt = read(study / arm / 'EPOCH_001.json')
            require(receipt['updates'] == 75 and receipt['protocol_sha256'] == q['study_protocol_sha256'], 'Epoch mismatch')
        expected = read(validation)['rows']
        checks = [c for c in d['backend_checks'] if c['model'] == model]
        require([c['index'] for c in checks] == [next(r['index'] for r in expected if r['count'] == n) for n in (1, 2, 3)], 'Backend identities')
        table[model] = {r['index']: r for r in rows}
        for r in rows:
            n = r['count']
            require(n in (1, 2, 3), 'Wrong source count')
            require(len(r['categories']) == len(r['nmse']) == len(r['si_sdr']) == len(r['reference_power']) == n, 'Missing sources')
            require(all(float('-inf') < v < float('inf') for k in ('nmse', 'si_sdr', 'reference_power') for v in r[k]), 'Nonfinite metrics')
            require(all(v >= 0 for v in r['nmse']) and all(v > 0 for v in r['reference_power']), 'Invalid metrics')
            require(r['sum_relative_error'] <= 1e-9 and r['predicted_count'] in (1, 2, 3), 'Invalid output')
            require(r['weakest_index'] == min(range(n), key=lambda i: r['reference_power'][i]), 'Weak source identity')
        for n, size in ((1, 10), (2, 24), (3, 14)):
            rr = [r for r in rows if r['count'] == n]
            require(len(rr) == size, 'Wrong TRAIN strata')
            summary = next(r for r in d['summaries'] if r['model'] == model and r['count'] == n)
            require(summary['cases'] == size, 'Summary cases')
            close(summary['mean_nmse'], stats.mean(v for r in rr for v in r['nmse']))
            close(summary['mean_si_sdr'], stats.mean(v for r in rr for v in r['si_sdr']))
            close(summary['weakest_nmse'], stats.mean(r['nmse'][r['weakest_index']] for r in rr))
            require(summary['all_nmse_below_point_one'] == sum(all(v < .1 for v in r['nmse']) for r in rr), 'Success count')
    paired = []
    for baseline, candidate in ((models[0], models[1]), (models[0], models[2]), (models[1], models[2])):
        for n in (1, 2, 3):
            pairs = [(table[baseline][i], table[candidate][i]) for i in indices if table[baseline][i]['count'] == n]
            for a, b in pairs:
                for key in ('index', 'count', 'categories', 'reference_power', 'weakest_index'):
                    require(a[key] == b[key], 'Pairing changed: ' + key)
            dn = [b['nmse'][j] - a['nmse'][j] for a, b in pairs for j in range(n)]
            ds = [b['si_sdr'][j] - a['si_sdr'][j] for a, b in pairs for j in range(n)]
            weak_n = [b['nmse'][a['weakest_index']] - a['nmse'][a['weakest_index']] for a, b in pairs]
            weak_s = [b['si_sdr'][a['weakest_index']] - a['si_sdr'][a['weakest_index']] for a, b in pairs]
            paired.append(dict(baseline=baseline, candidate=candidate, count=n, cases=len(pairs),
                mean_nmse_delta=stats.mean(dn), mean_si_sdr_delta=stats.mean(ds),
                weakest_nmse_delta=stats.mean(weak_n), weakest_si_sdr_delta=stats.mean(weak_s),
                components_both_improved=sum(a < 0 and b > 0 for a, b in zip(dn, ds)),
                components_both_worsened=sum(a > 0 and b < 0 for a, b in zip(dn, ds))))
    power_diagnostics = []
    for role, records in (('TRAIN48', list(table[models[0]].values())),
            ('DEVELOPMENT630', read(Path(p['validation_identity_template']))['rows'])):
        for n in (2, 3):
            sir = []
            for row in records:
                if row['count'] != n:
                    continue
                power = row['reference_power']
                weak = min(range(n), key=lambda j: power[j])
                sir.append(10 * math.log10(power[weak] / sum(power[j] for j in range(n) if j != weak)))
            power_diagnostics.append(dict(role=role, count=n, mixtures=len(sir),
                weakest_sir_min_db=min(sir), weakest_sir_median_db=stats.median(sir),
                weakest_sir_max_db=max(sir), weakest_sir_below_minus_20_db=sum(s < -20 for s in sir),
                definition='10log10(Pweak/sum(other reference powers)); not SNR and no cross terms'))
    result = dict(status='PASS', auditor_sha256=digest(Path(__file__)),
        completed_diagnosis_sha256=digest(diagnosis / 'COMPLETE.json'),
        study_protocol_sha256=q['study_protocol_sha256'], frozen_sources_verified=len(q['source_sha256']),
        train_mixtures=48, models=3, scored_rows=144, reproduced_backend_cases=9,
        recomputed_summaries=d['summaries'], paired=paired, power_diagnostics=power_diagnostics, waveform_reads=0,
        limitation='Already seen TRAIN fit; neither an independent test nor a matched TRAIN/development distribution.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('study', 'diagnosis', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.study.resolve(), args.diagnosis.resolve(), args.output.resolve())
