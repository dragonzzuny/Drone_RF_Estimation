"""Constant-scalar oracle diagnostics on the EXISTING development validation.

These require reference count/waveforms and are NOT deployable separators or
performance bounds for nonlinear models. They show what simple rescaling of
an unseparated mixture can achieve in the same unscaled NMSE metric.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from study import admitted_dataset
from drone_rf.data import sha256
from drone_rf.context_training_data import write_json

PREPARATION = Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/rfuav_architecture_20261008/dense_preparation')
SAVED = PREPARATION.parent / 'dense_gpu_run/unet_mean/VALIDATION_001.json'


def run(output):
    if output.exists():
        raise RuntimeError('Refuse overwrite')
    data = admitted_dataset(PREPARATION, 'validation_pack', 1)
    saved = json.loads(SAVED.read_text())
    rows = []
    for original in saved['rows']:
        item = data[original['index']]
        count = item['construction_count']
        refs = item['references'][:count].astype(np.complex128)
        mix = item['mixture'].astype(np.complex128)
        energy = np.sum(np.abs(refs)**2, axis=-1)
        np.testing.assert_allclose(energy/refs.shape[-1], original['reference_power'], rtol=1e-6)
        coefficients = (refs*mix.conj()).sum(-1)/np.vdot(mix,mix).real
        values = {}
        for name, gains in (('oracle_count_equal_share', np.full(count,1/count)),
                            ('oracle_positive_scalar', np.maximum(0.,coefficients.real)),
                            ('oracle_complex_scalar', coefficients)):
            values[name] = (np.sum(np.abs(gains[:,None]*mix-refs)**2, axis=-1)/energy).tolist()
        rows.append(dict(index=original['index'], count=count, categories=original['categories'],
            selected_unet_nmse=original['nmse'], **values))
    summary = []
    for count in (2,3):
        selected = [r for r in rows if r['count']==count]
        fields=('selected_unet_nmse','oracle_count_equal_share','oracle_positive_scalar','oracle_complex_scalar')
        summary.append(dict(count=count,cases=len(selected),
            **{f:float(np.mean([np.mean(r[f]) for r in selected])) for f in fields},
            unet_wins_over_complex_scalar_cases=int(sum(np.mean(r['selected_unet_nmse'])<np.mean(r['oracle_complex_scalar']) for r in selected))))
    result = dict(scope='Reused development validation; no held-out data',
        oracle_warning='Reference-assisted constant scalars/count, not deployable separation and not nonlinear bounds.',
        scalar_outputs='Copies of the mixture times one constant per component; no temporal separation.',
        native_center_offsets_preserved=False, no_heldout_iq=True,
        source_sha256=sha256(Path(__file__)), validation_result_sha256=sha256(SAVED),
        preparation_sha256=sha256(PREPARATION/'PREPARATION.json'), summary=summary, rows=rows)
    write_json(output, result)
    print(json.dumps(summary,indent=2), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cpus=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,set(cpus[-4:-2] if len(cpus)>=4 else cpus[:2]))
    os.nice(10)
    run(args.output)
