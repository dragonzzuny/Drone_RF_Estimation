"""Admission checks for within-dataset, within-band synthetic mixtures."""
import math


def band_group(center_hz):
    """Receiver-center group; does not assert spectral overlap or occupied edges."""
    center = float(center_hz)
    if not math.isfinite(center):
        raise ValueError('Finite receiver center required')
    if 2.4e9 <= center < 2.5e9:
        return '2.4GHz'
    if 5.7e9 <= center < 5.9e9:
        return '5.8GHz'
    raise ValueError('Unreviewed receiver band')


def validate_sources(sources):
    if not 1 <= len(sources) <= 3:
        raise ValueError('One to three recorded components required')
    datasets = {r['dataset'] for r in sources}
    if len(datasets) != 1 or not next(iter(datasets)):
        raise ValueError('Cross-dataset mixture is excluded')
    bands = {band_group(r['center_hz']) for r in sources}
    if len(bands) != 1:
        raise ValueError('Cross-band mixture is excluded')
    rates = {float(r['fs_hz']) for r in sources}
    if len(rates) != 1 or not all(math.isfinite(r) and r > 0 for r in rates):
        raise ValueError('Matching positive sample rates required')
    # Unknown model/serial is not evidence that receivers are identical.
    receivers = {r.get('receiver_model') for r in sources if r.get('receiver_model')}
    if len(receivers) > 1:
        raise ValueError('Different known receiver models require a separate experiment')
    return dict(dataset=next(iter(datasets)), rf_band=next(iter(bands)),
                fs_hz=next(iter(rates)))
