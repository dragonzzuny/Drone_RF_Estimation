"""Long mixture-only crop; source targets/scoring windows stay byte-identical."""
import sys
import numpy as np
import torch
from waveform_models import LEGACY, LONG_SAMPLES, TARGET_SAMPLES
sys.path.insert(0,str(LEGACY))
from study import admitted_dataset as short_dataset
from drone_rf.context_data import component_gains


class LongContextMixtures:
    def __init__(self,preparation,role,epoch=1):
        self.base=short_dataset(preparation,role,epoch)
        self.rows=self.base.rows
        self.library=self.base.library

    def __len__(self):
        return len(self.base)

    def __getitem__(self,index):
        item=self.base[index]
        row=self.rows[index]; count=int(row['count']); crop=int(row['crop_start'])
        start=int(np.clip(crop-(LONG_SAMPLES-TARGET_SAMPLES)//2,0,2097152-LONG_SAMPLES))
        clips=[self.library.clips[int(i)] for i in row['indices'][:count]]
        gains=component_gains([c['mean_power'] for c in clips],row['levels'][:count],row['phases'][:count])
        long_mix=np.zeros(LONG_SAMPLES,np.complex64)
        for idx,gain in zip(row['indices'][:count],gains):
            long_mix+=(self.library._array(int(idx))[start:start+LONG_SAMPLES]*gain).astype(np.complex64)
        if not np.array_equal(long_mix[crop-start:crop-start+TARGET_SAMPLES],item['mixture']):
            raise RuntimeError('Long crop differs from the frozen scoring mixture')
        item.update(long_mixture=long_mix,long_start=start)
        return item


def admitted_dataset(preparation,role,epoch=1):
    return LongContextMixtures(preparation,role,epoch)


def batch(item,device):
    keys=('mixture','long_mixture','long_start','references','active','context_features','crop_start','construction_count')
    return {k:torch.as_tensor(item[k],device=device)[None] for k in keys}
