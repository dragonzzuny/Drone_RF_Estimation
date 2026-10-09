"""Show all prespecified TRAIN clips, not a best-case waveform selection."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(source, output):
    d = json.loads((source / 'COMPLETE.json').read_text())
    if d['status'] != 'COMPLETE' or len(d['rows']) != 24:
        raise ValueError('Completed TRAIN24 diagnosis required')
    with np.load(source / 'CURVES.npz', allow_pickle=False) as cache:
        curves, lags = cache['magnitudes'], cache['lags']
    if curves.shape != (24, 32769) or not np.array_equal(lags, np.arange(32769)):
        raise ValueError('Curve geometry mismatch')
    groups = defaultdict(list)
    for index, row in enumerate(d['rows']):
        groups[row['pack_id']].append(index)
        for p in row['peaks']:
            np.testing.assert_allclose(curves[index, p['lag_samples']], p['magnitude'], rtol=1e-6, atol=1e-8)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True)
    keep = lags >= 513
    x = lags[keep] / 100.
    summary = []
    for ax, (pack, indices) in zip(axes.flat, sorted(groups.items())):
        if len(indices) != 3:
            raise ValueError('Expected three predetermined clips per group')
        for j, i in enumerate(indices):
            ax.plot(x, curves[i, keep], color='#0072B2', alpha=.35, lw=.6,
                label='Each of 3 TRAIN clips' if j == 0 else None)
        median = np.median(curves[indices], axis=0)
        ax.plot(x, median[keep], color='#D55E00', lw=.9, label='Median of 3 clips')
        peak = int(np.argmax(median[513:]) + 513)
        title = d['rows'][indices[0]]['category'].replace('DJI ', '') + ' / ' + pack.split('/')[-2]
        ax.set_title(title, fontsize=10)
        ax.annotate(f'{peak/100:.2f} us', xy=(peak/100, median[peak]),
            xytext=(peak/100 + 35, .113), fontsize=8,
            arrowprops=dict(arrowstyle='-', lw=.6, color='#444444'))
        ax.grid(alpha=.18)
        ax.set_xlim(5.13, 327.68)
        ax.set_ylim(0, max(.125, float(curves[:, keep].max())*1.1))
        summary.append(dict(pack_id=pack, indices=indices, median_peak_lag_samples=peak,
            median_peak_lag_microseconds=peak/100, median_correlation=float(median[peak])))
    fig.supylabel('Normalized |complex correlation|', x=.012, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel('Lag (microseconds; native 100 MS/s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center', bbox_to_anchor=(.5, .043), frameon=False)
    fig.suptitle('TRAIN records: longer-lag complex correlation', fontsize=14)
    fig.text(.5, .012, 'All 24 prespecified clips from 8 TRAIN recording groups. '
        'Same clips are not independent recordings.\n'
        'Ordinary autocorrelation does not establish FHSS/OFDM identity, source separability, or a recovery bound.',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(.02, .085, 1, .965))
    fig.savefig(output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'), dpi=200)
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    output.with_suffix('.json').write_text(json.dumps(dict(status='COMPLETE',
        source_complete_sha256=sha(source / 'COMPLETE.json'), curves_sha256=sha(source / 'CURVES.npz'),
        plotter_sha256=sha(Path(__file__)), groups=summary,
        pdf_sha256=sha(output.with_suffix('.pdf')), png_sha256=sha(output.with_suffix('.png'))), indent=2) + '\n')
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
