"""Check preserved good slots and removal of an absent dictionary component."""
from pathlib import Path
import json,hashlib
import numpy as np
from spectral_routing import WINDOW,profiles
from spectral_routing_v2 import route


def main():
    rows=[];n=np.arange(4*WINDOW)
    templates=np.zeros((4,len(n)),np.complex64)
    for j,f in enumerate((31,137,379)):templates[j]=np.exp(2j*np.pi*f*n/512)
    spectra,_,_=profiles(templates)
    bank=dict(categories=['A','B','C'],psd=spectra[0].tolist())
    for count in (2,3):
        y=templates.copy()
        if count==2:y[1]=0 # absent B; active A and C
        mixture=y.sum(0)
        for method in ('count_dictionary','slot_initialized_clusters'):
            z,receipt=route(y,bank,method,count,mixture)
            active=[j for j in range(3) if np.any(y[j]!=0)]
            errors=np.array([[np.mean(abs(z[i]-y[j])**2) for i in range(3)] for j in active])
            assert len(set(errors.argmin(1)))==count and np.max(errors.min(1))==0
            if method=='count_dictionary' and count==2:assert set(receipt['selected_categories'])=={'A','C'}
            rows.append(dict(method=method,count=count,consistent_source_nmse=0))
    root=Path(__file__).resolve().parents[2]
    result=dict(status='PASS',checks=rows,rf_evaluation=False,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('spectral_routing_v2.py'))})
    (root/'reports/2026-10-13/SPECTRAL_ROUTING_V2_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
