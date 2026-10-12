"""Whole C observations in the same coordinates/precision as frozen short C."""
import numpy as np
from sequence_data import LENGTH,WINDOW,component_gains


def selected_indices(data):
    selected=[];seen=set()
    for i,row in enumerate(data.rows):
        count=int(row['count'])
        cats=tuple(data.library.clips[int(j)]['category'] for j in row['indices'][:count])
        key=(count,cats,tuple(float(v) for v in row['levels'][:count]))
        if key not in seen:seen.add(key);selected.append(i)
    counts=[int(data.rows[i]['count']) for i in selected]
    assert len(selected)==24 and [counts.count(k) for k in (1,2,3)]==[5,12,7]
    return selected


def components(data,index):
    row=data.rows[index];count=int(row['count']);ids=row['indices'][:count]
    clips=[data.library.clips[int(i)] for i in ids]
    assert all(c['role']=='validation_pack' and 'VTSBW=20/' in c['pack_id'] for c in clips)
    assert len({c['common_center_hz'] for c in clips})==1
    gains=component_gains([c['mean_power'] for c in clips],row['levels'][:count],row['phases'][:count])
    refs=np.zeros((3,LENGTH),np.complex64)
    for j,(i,gain) in enumerate(zip(ids,gains)):
        refs[j]=(data.library._array(int(i))*gain).astype(np.complex64)
    original=data[index];start=int(row['crop_start'])
    assert np.array_equal(refs[:,start:start+WINDOW],original['references'])
    assert np.array_equal(refs.sum(0)[start:start+WINDOW],original['mixture'])
    return refs
