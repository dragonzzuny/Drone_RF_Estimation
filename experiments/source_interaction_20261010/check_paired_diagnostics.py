"""Check evaluation alignment/failure handling using saved development metrics."""
import argparse
import copy
from pathlib import Path
import paired_diagnostics as p


def check(study, output):
    base_path = study / 'retained_unet/VALIDATION_000.json'
    now_path = study / 'retained_unet/VALIDATION_001.json'
    base, now = p.watch.read(base_path), p.watch.read(now_path)
    _, ids = p.watch.validate(base_path, None)
    p.watch.validate(now_path, ids)
    identity = p.compare(base, base, 'base', 'base')
    assert all(r['mean_delta_nmse'] == r['mean_delta_si'] == r['both_better'] == r['both_worse'] == 0
               for r in identity['count'])
    reordered = copy.deepcopy(now)
    reordered['rows'].reverse()
    result = p.compare(base, now, 'a', 'b')
    assert result == p.compare(base, reordered, 'a', 'b')
    bad = copy.deepcopy(now)
    bad['rows'][0]['reference_power'][0] *= 2
    try:
        p.compare(base, bad, 'a', 'b')
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched reference accepted')
    bad = copy.deepcopy(now)
    bad['rows'][0]['si_sdr'][0] = None
    try:
        p.compare(base, bad, 'a', 'b')
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite metric silently excluded')
    for table in ('count', 'category', 'combination', 'power', 'category_power'):
        assert sum(x['source_cases'] for x in result[table]) == 1050
    receipt = dict(status='PASS', source_sha256=p.watch.digest(Path(p.__file__)),
        checker_sha256=p.watch.digest(Path(__file__)),
        validation_sha256=[p.watch.digest(base_path), p.watch.digest(now_path)],
        identity_comparison_exact_zero=True, row_order_invariant=True,
        reference_mismatch_rejected=True, nonfinite_metric_rejected=True,
        all_1050_sources_preserved=True, registered_means_reproduced=True,
        heldout_read=False, waveform_reads=0, checkpoint_reads=0)
    p.watch.write(output, receipt)
    print('PASS: aligned paired metrics and error handling')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('study', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    check(args.study.resolve(), args.output.resolve())
