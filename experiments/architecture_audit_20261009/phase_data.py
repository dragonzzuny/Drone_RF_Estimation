"""Full observed complex mixture; same fine targets as frozen native study."""
import numpy as np
import torch

import phase_packing as pp
import fit_diagnostic  # Establish the pinned native/vendor import paths.
from drone_rf.context_data import component_gains


def example(dataset, index):
    item = dataset[index]
    row = dataset.rows[index]
    count = int(row['count'])
    indices = row['indices'][:count]
    gains = component_gains([dataset.library.clips[int(i)]['mean_power'] for i in indices],
                            row['levels'][:count], row['phases'][:count])
    mixture = np.zeros(pp.SAMPLES, np.complex64)
    for i, gain in zip(indices, gains):
        mixture += (dataset.library._array(int(i)) * gain).astype(np.complex64)
    start = int(item['crop_start'])
    np.testing.assert_array_equal(mixture[start:start + pp.FINE], item['mixture'])
    if not np.isfinite(mixture).all():
        raise ValueError('Nonfinite observed long I/Q')
    return dict(item, long_mixture=mixture)


def batch(item, device):
    keys = ('mixture', 'references', 'active', 'context_features', 'crop_start',
            'construction_count', 'long_mixture')
    return {k: torch.as_tensor(np.asarray(item[k]), device=device)[None] for k in keys}


def predict(net, item):
    # Whitelist only observed inputs. No targets, source identity or true count.
    return net(item['long_mixture'], item['context_features'], item['crop_start'])
