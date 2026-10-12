"""Known narrowband sources with adversarial slot swaps; no RF result claim."""
from pathlib import Path
import hashlib,json
import numpy as np
from spectral_routing import route,profiles,WINDOW


def main():
    checks=[]
    for count in (2,3):
        n=np.arange(4*WINDOW)
        refs=np.stack([np.exp(2j*np.pi*f*n/512).astype(np.complex64) for f in (31,137,379)[:count]])
        y=np.zeros((4,len(n)),np.complex64)
        for b,p in enumerate(((0,1,2),(2,0,1),(1,2,0),(0,2,1))):
            for j in range(count):y[p[j],b*WINDOW:(b+1)*WINDOW]=refs[j,b*WINDOW:(b+1)*WINDOW]
        template=np.zeros_like(y);template[:count]=refs
        psd,_,_=profiles(template);bank=dict(categories=[str(j) for j in range(count)],psd=psd[0,:count].tolist())
        for method in ('fit_dictionary','observation_clusters'):
            fixed,receipt=route(y,bank,method,count)
            # Each complete source must occupy one consistent output, up to a
            # global permutation in the observation-only clustering arm.
            errors=np.array([[np.mean(abs(fixed[i]-refs[j])**2) for i in range(3)] for j in range(count)])
            matches=errors.argmin(1);assert len(set(matches))==count and np.max(errors.min(1))==0
            assert np.array_equal(fixed[3],y[3]);checks.append(dict(count=count,method=method,max_nmse=0))
    root=Path(__file__).resolve().parents[2]
    result=dict(status='PASS',synthetic_checks=checks,rf_evaluation=False,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('spectral_routing.py'))})
    (root/'reports/2026-10-13/SPECTRAL_ROUTING_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
