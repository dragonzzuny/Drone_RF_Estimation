"""Build schedules from explicitly approved, within-dataset source classes."""
import collections
import numpy as np

from .context_training_data import DTYPE, strata
from .mixture_constraints import validate_sources


def make_within_dataset_schedule(clips, allowed_classes, role, epochs=50,
                                examples_per_count=800, length=63872):
    if role not in ('train_pack','validation_pack') or epochs<1 or examples_per_count<1:
        raise ValueError('Invalid development schedule')
    if len({c['dataset'] for c in clips if c['role']==role}) != 1:
        raise ValueError('Cross-dataset training is excluded; prepare independent experiments')
    groups=collections.defaultdict(list)
    for i,c in enumerate(clips):
        if c['role']==role:
            groups[c['source_id']].append(i)
    allowed={tuple(sorted(row['sources'])) for row in allowed_classes}
    conditions={}
    for count in (1,2,3):
        conditions[count]=[(ids,levels) for ids,levels in strata(groups,count)
                           if tuple(ids) in allowed]
        if not conditions[count]:
            raise ValueError(f'No approved {count}-source classes in {role}')
    output=[]
    for epoch in range(1,epochs+1):
        rng=np.random.default_rng(np.random.SeedSequence([0,epoch,145 if role=='train_pack' else 146]))
        rows=np.zeros(3*examples_per_count,dtype=DTYPE)
        rows['indices']=-1
        rows['epoch']=epoch
        position=0
        for count in (1,2,3):
            for number in range(examples_per_count):
                ids,levels=conditions[count][((epoch-1)*examples_per_count+number)%len(conditions[count])]
                indices=[int(rng.choice(groups[k])) for k in ids]
                selected=[clips[i] for i in indices]
                validate_sources(selected)
                sizes={c['samples'] for c in selected}
                if len(sizes)!=1 or min(sizes)<length:
                    raise ValueError('Incompatible context lengths')
                row=rows[position]
                row['count']=count
                row['indices'][:count]=indices
                row['levels'][:count]=levels
                row['phases'][:count]=rng.uniform(-np.pi,np.pi,count)
                row['crop_start']=rng.integers(0,min(sizes)-length+1)
                position+=1
        output.extend(rows[rng.permutation(len(rows))])
    return np.array(output,dtype=DTYPE)
