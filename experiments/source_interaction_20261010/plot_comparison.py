"""Plot verified completed epoch receipts, including all nonselected epochs."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_comparison as worker

w = worker.watch


def run(study, output):
    p = w.read(study / 'PROTOCOL.json')
    _, identities = w.validate(Path(p['validation_identity_template']), None)
    histories = {}
    evidence = {}
    for arm in worker.ARMS:
        history = []
        for epoch in range(4):
            receipt = study / arm / f'EPOCH_{epoch:03d}.json'
            if epoch and not receipt.exists():
                break
            validation = study / arm / f'VALIDATION_{epoch:03d}.json'
            measured, _ = w.validate(validation, identities)
            if epoch:
                e = w.read(receipt)
                if e['protocol_sha256'] != w.digest(study / 'PROTOCOL.json') or e['updates'] != epoch * 75:
                    raise ValueError('Receipt mismatch')
                evidence[f'{arm}/receipt{epoch}'] = w.digest(receipt)
            history.append(measured)
            evidence[f'{arm}/validation{epoch}'] = w.digest(validation)
        histories[arm] = history
    complete = all(len(h) == 4 for h in histories.values())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    curves = []
    colors = {'retained_unet': '#0072B2', 'source_interaction': '#D55E00'}
    labels = {'retained_unet': 'Existing U-Net', 'source_interaction': 'U-Net + source head'}
    for col, count in enumerate((2, 3)):
        for row, key in enumerate(('mean_nmse', 'mean_si_sdr')):
            ax = axes[row, col]
            for arm, history in histories.items():
                values = [next(g[key] for g in h['by_count'] if g['count'] == count) for h in history]
                epochs = list(range(len(history)))
                curves.append(dict(arm=arm, count=count, metric=key, epochs=epochs, values=values))
                ax.plot(epochs, values, color=colors[arm], label=labels[arm],
                    linestyle='-' if arm == 'retained_unet' else '--',
                    marker='o' if arm == 'retained_unet' else 'x', markersize=5, linewidth=1.8)
            ax.set_title(f'{count} recorded contributions')
            ax.grid(alpha=.2)
            ax.set_xticks(range(4))
            ax.set_xlim(-.08, 3.08)
            ax.set_ylabel('Raw complex I/Q NMSE (lower)' if row == 0 else 'Complex SI-SDR, dB (higher)')
            if row == 0:
                ax.set_ylim(bottom=0, top=max(1, ax.get_ylim()[1]))
            else:
                ax.set_xlabel('Additional epoch (75 updates each)')
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.5, .08), ncol=2, frameon=False)
    fig.suptitle(('Completed' if complete else 'Interim') + ' matched source-head comparison', fontsize=14)
    fig.text(.5, .015, 'Same retained parent, RFUAV split, AdamW 1e-5, data schedule and seed 0.\n'
        '630 development mixtures (5 recording groups); single-pass inference. Actual epochs; no held-out test.\n'
        'Overlapping lines indicate small differences; paired changes are reported separately.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, .95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'), dpi=220)
    plt.close(fig)
    common = min(map(len, histories.values()))
    paired = []
    for n in (2, 3):
        for key in ('mean_nmse', 'mean_si_sdr'):
            values = []
            for epoch in range(common):
                control, candidate = [next(g[key] for g in histories[a][epoch]['by_count'] if g['count'] == n)
                                      for a in worker.ARMS]
                values.append(candidate - control)
            paired.append(dict(count=n, metric=key, epochs=list(range(common)), candidate_minus_control=values))
    w.write(output.with_suffix('.json'), dict(status='COMPLETE' if complete else 'PARTIAL',
        plotter_sha256=w.digest(Path(__file__)), protocol_sha256=w.digest(study / 'PROTOCOL.json'),
        evidence_sha256=evidence, curves=curves, paired=paired, error_bars=False,
        pdf_sha256=w.digest(output.with_suffix('.pdf')), png_sha256=w.digest(output.with_suffix('.png'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve())
