"""Plot all completed phase-context epochs with their original raw metrics."""
import argparse
import csv
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import phase_watch
import watch_epochs as watch


def render(study, output):
    snapshot = phase_watch.snapshot(study)
    if snapshot['common_epoch'] != 5 or not (study/'COMPLETE.json').exists():
        raise ValueError('Both registered five-epoch arms must be complete')
    rows, hashes, identities = [], {}, None
    for arm in ('local', 'long'):
        for epoch in range(6):
            path = study/arm/f'VALIDATION_{epoch:03d}.json'
            metrics, identities = watch.validate(path, identities)
            hashes[f'{arm}/VALIDATION_{epoch:03d}.json'] = watch.digest(path)
            for row in metrics['by_count']:
                if row['count'] in (2, 3):
                    rows.append(dict(arm=arm, epoch=epoch, updates=75*epoch,
                                     count=row['count'], mean_nmse=row['mean_nmse'],
                                     mean_si_sdr=row['mean_si_sdr']))
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.7), sharex=True, sharey='row')
    styles = [('local', 'Outside-crop I/Q masked', '#0072B2', 'o'),
              ('long', 'Full 20.89 ms I/Q visible', '#D55E00', 's')]
    for column, count in enumerate((2, 3)):
        for arm, label, color, marker in styles:
            data = [r for r in rows if r['arm'] == arm and r['count'] == count]
            for index, metric in enumerate(('mean_nmse', 'mean_si_sdr')):
                axes[index, column].plot([r['epoch'] for r in data], [r[metric] for r in data],
                                         color=color, marker=marker, markersize=5,
                                         linewidth=1.7, label=label)
        axes[0, column].set_title(f'{count} recorded components | 210 mixtures',
                                  fontweight='bold', pad=10)
        axes[0, column].set_yscale('log')
        axes[0, column].axhline(1, color='#777777', ls=':', lw=.8)
        axes[1, column].axhline(0, color='#777777', ls=':', lw=.8)
        axes[1, column].set_xlabel('Completed epoch (75 optimizer updates each)')
        for ax in axes[:, column]:
            ax.set_xticks(range(6))
            ax.set_xlim(-.15, 5.15)
            ax.grid(axis='y', which='major', alpha=.18)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel('Raw I/Q NMSE (log scale)')
    axes[1, 0].set_ylabel('Complex SI-SDR (dB)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', frameon=False,
               bbox_to_anchor=(.53, .929))
    fig.suptitle('Does longer complex context improve reconstruction?', fontsize=16, y=.99)
    fig.text(.53, .944, 'Matched 4.64M-parameter WaveNets | RFUAV development | one training seed',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.09, right=.98, top=.84, bottom=.21, hspace=.22, wspace=.10)
    fig.text(.09, .137, 'NMSE: lower is better. SI-SDR: higher is better. Dotted lines: NMSE = 1 and SI-SDR = 0 dB.', fontsize=9)
    fig.text(.09, .11, 'Every actual epoch shown, including initialization; no output gain correction.', fontsize=9)
    fig.text(.09, .083, 'Both arms receive full-mixture RMS and time-averaged power features.', fontsize=9)
    fig.text(.09, .056, 'The only arm difference is visibility of complex I/Q outside the 0.63872 ms target crop.', fontsize=9)
    fig.text(.09, .029, 'Correlated development crops; this initial-budget screen is not an independent test or a convergence study.', fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(output.with_suffix('.'+ext), dpi=300, facecolor='white')
    plt.close(fig)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    watch.write(output.with_suffix('.csv'), stream.getvalue())
    manifest = dict(status='COMPLETE', source_sha256=watch.digest(Path(__file__)),
                    protocol_sha256=snapshot['protocol_sha256'], validation_sha256=hashes,
                    plotted_metric_rows=len(rows), epochs=list(range(6)),
                    waveform_reads=0, heldout_read=False, independent_test=False,
                    files={ext: watch.digest(output.with_suffix('.'+ext)) for ext in ('pdf', 'png', 'csv')})
    watch.write(output.with_suffix('.json'), manifest)
    print(manifest)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    render(a.study.resolve(), a.output.resolve())
