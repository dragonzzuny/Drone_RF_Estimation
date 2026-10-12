"""Reference-free streaming assembly over one admitted20.8896ms observation."""
import itertools
import numpy as np
import torch
from sequence_data import WINDOW,LENGTH
from sequence_eval import score,predict_eight,merge
from study import batch

PERMS=list(itertools.permutations(range(3)))


class StreamingMerge:
    def __init__(self,length):
        self.total=np.zeros((4,length),np.complex128)
        self.weights=np.zeros(length)
        self.window=np.maximum(np.hanning(WINDOW),.001)
        self.receipts=[]

    def append(self,y,start):
        assert y.shape==(4,WINDOW) and 0<=start<=self.weights.size-WINDOW
        overlap=self.weights[start:start+WINDOW]>0;p=(0,1,2);costs=[]
        if overlap.any():
            anchor=self.total[:3,start:start+WINDOW][:,overlap]/self.weights[start:start+WINDOW][overlap]
            costs=[float(np.mean(abs(y[list(q)][:,overlap]-anchor)**2)) for q in PERMS]
            p=PERMS[int(np.argmin(costs))]
        self.total[:,start:start+WINDOW]+=y[[*p,3]]*self.window
        self.weights[start:start+WINDOW]+=self.window
        self.receipts.append(dict(start=start,permutation=list(p),costs=costs))

    def finish(self):
        assert np.all(self.weights>0)
        # In-place division avoids another complete complex128 observation.
        self.total/=self.weights
        return self.total.astype(np.complex64),self.receipts


def check_streaming():
    rng=np.random.default_rng(0)
    ys=[(rng.normal(size=(4,WINDOW))+1j*rng.normal(size=(4,WINDOW))).astype(np.complex64) for _ in range(5)]
    starts=[j*(WINDOW//2) for j in range(5)]
    old,receipt=merge(ys,starts,3*WINDOW);new=StreamingMerge(3*WINDOW)
    for y,start in zip(ys,starts):new.append(y,start)
    result,orders=new.finish()
    assert np.array_equal(old,result) and orders==receipt
    return dict(status='PASS',sample_difference=0,permutations_equal=True,synthetic_windows=5)


@torch.no_grad()
def evaluate(net,predict,refs,features,count,device,progress=None,attach_neighbours=None):
    assert refs.shape==(3,LENGTH)
    mixture=refs.sum(0);starts=list(range(0,LENGTH-WINDOW+1,WINDOW//2))
    if starts[-1]!=LENGTH-WINDOW:starts.append(LENGTH-WINDOW)
    accumulated=StreamingMerge(LENGTH);net.eval()
    for i,start in enumerate(starts):
        item=batch(dict(mixture=mixture[start:start+WINDOW].copy(),
            references=np.zeros((3,WINDOW),np.complex64),active=np.zeros(3,bool),
            context_features=features,crop_start=start,construction_count=count),device)
        if attach_neighbours is not None:item=attach_neighbours(item,mixture,start,device)
        _,y,logits,_=predict_eight(net,predict,item)
        accumulated.append(y[0].cpu().numpy(),start)
        if progress is not None:progress(i+1,len(starts))
    assembled,receipt=accumulated.finish();del accumulated
    y=torch.from_numpy(assembled)[None].to(device)
    target=torch.from_numpy(refs)[None].to(device);mix=torch.from_numpy(mixture)[None].to(device)
    metrics=score(y,target,mix,count,logits);global_order=metrics['assignment']
    local=[]
    for start in range(0,LENGTH,WINDOW):
        stop=min(start+WINDOW,LENGTH)
        r=score(y[...,start:stop],target[...,start:stop],mix[...,start:stop],count,logits)
        aligned=y[:,global_order[:count],start:stop].to(torch.complex128)
        reference=target[:,:count,start:stop].to(torch.complex128)
        power=reference.abs().square().mean(-1)
        fixed=((aligned-reference).abs().square().mean(-1)/torch.where(power>0,power,1.))[0].tolist()
        local.append(dict(start=start,stop=stop,local_assignment=r['assignment'],
            active_order_differs=r['assignment'][:count]!=global_order[:count],
            fixed_global_nmse=fixed,local_pit_nmse=r['nmse'],reference_power=r['reference_power'],
            zero_power_sources=int((power==0).sum())))
    return assembled,dict(metrics=metrics,window_count=len(starts),overlap_matching=receipt,
        local_scoring=local,active_order_changes=sum(v['active_order_differs'] for v in local),
        limitation='Local optimal order can change at low activity; inspect power/error, not the count alone')
