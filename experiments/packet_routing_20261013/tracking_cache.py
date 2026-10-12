"""Separate observed-only features from reference-dependent scoring/labels."""
from pathlib import Path
import numpy as np
from spectral_routing import profiles
from routing_statistics import collect,score,training_labels
from evaluate_parent_full_context import read,write,sha


def observed_features(output):
    # No reference argument. The sum of estimated slots reconstructs the input.
    vectors,powers,bounds=profiles(output)
    mix=output.sum(0).astype(np.complex128)
    power=float(np.mean(abs(mix)**2))
    assert power>0
    return dict(vectors=vectors.astype(np.float32),powers=powers,bounds=np.asarray(bounds),
                mixture_power=np.asarray(power))


def save_case(directory,index,output,reference,count,predicted_count,metadata,training=False):
    stem=directory/f'CASE_{index:03d}';features=stem.with_suffix('.features.npz')
    scoring=stem.with_suffix('.scoring.npz');label_path=stem.with_suffix('.labels.npz')
    assert not features.exists() and not scoring.exists()
    feature=observed_features(output);np.savez_compressed(features,**feature)
    cache=collect(output,reference,reference.sum(0));np.savez_compressed(scoring,**cache)
    base=score(cache,np.tile([0,1,2],(len(feature['bounds']),1)),count,predicted_count)
    item=dict(index=index,count=count,predicted_count=predicted_count,metadata=metadata,
        features=str(features),features_sha256=sha(features),scoring=str(scoring),scoring_sha256=sha(scoring),
        baseline_metrics=base,training=training,reference_statistics_in_feature_file=False)
    if training:
        labels,weights=training_labels(cache,count);np.savez_compressed(label_path,labels=labels,weights=weights)
        item.update(labels=str(label_path),labels_sha256=sha(label_path))
    write(stem.with_suffix('.json'),item)
    return item


def feature_input(item):
    path=Path(item['features']);assert sha(path)==item['features_sha256']
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)=={'vectors','powers','bounds','mixture_power'}
        vectors=z['vectors'];powers=z['powers'];power=float(z['mixture_power'])
        # Unscaled spectrum shape plus relative token power; no product label.
        result=np.concatenate([vectors,np.log1p(powers/power)[...,None].astype(np.float32)],axis=-1)
        assert result.shape[1:]==(3,513) and np.isfinite(result).all()
        return result.copy(),powers.copy(),z['bounds'].copy()
