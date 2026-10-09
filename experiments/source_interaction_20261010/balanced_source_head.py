"""Candidate: balance shared features and source features before attention.

Motivated by the TRAIN4 e2 mechanism diagnosis, not an RF-validated method.
No extra parameters. The original output path is preserved. The three raw
complex pairs share ONE positive normalization factor at each TF location,
which preserves their relative amplitudes and phases before the learned map.
"""
import math
from source_head import SourceInteractionHead


class BalancedSourceInteractionHead(SourceInteractionHead):
    def correction(self, features, raw_sources):
        # The equal group-energy factors account for 64 common and 2 per-source
        # coordinates. They do not use source labels, counts, or target energy.
        eps = 1e-6
        f_scale = (features.square().mean(-1, keepdim=True) + eps**2).sqrt()
        r_scale = (raw_sources.square().mean((1, 2), keepdim=True) + eps**2).sqrt()
        f = features / f_scale * math.sqrt(66 / (2 * 64))
        r = raw_sources / r_scale * math.sqrt(66 / (2 * 2))
        return super().correction(f, r)


def augment(net, **kwargs):
    if isinstance(net.output, SourceInteractionHead):
        raise ValueError('Head already augmented')
    net.output = BalancedSourceInteractionHead(net.output, **kwargs)
    return net
