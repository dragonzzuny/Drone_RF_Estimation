import numpy as np
from pathlib import Path
from numpy_waveform_score import score as direct
from routing_statistics import collect,score,training_labels
from spectral_routing import PERMS,WINDOW
from sequence_data import ROOT,write_json,sha256


def run():
    rng=np.random.default_rng(101304);rows=[]
    for count in (1,2,3):
        n=WINDOW*2+5120
        ref=np.zeros((3,n),np.complex64)
        ref[:count]=(rng.normal(size=(count,n))+1j*rng.normal(size=(count,n)))
        ref[:count]*=np.geomspace(.07,1,count)[:,None]
        output=np.zeros((4,n),np.complex64)
        output[:3]=ref+(rng.normal(size=ref.shape)+1j*rng.normal(size=ref.shape))*.04
        mix=ref.sum(0);output[3]=mix-output[:3].sum(0)
        cache=collect(output,ref,mix);labels,weights=training_labels(cache,count)
        assert np.all(labels[:,:count]==np.arange(count))
        for j in range(count):np.testing.assert_allclose(weights[labels==j].sum(),1/count,atol=1e-7)
        for trial in range(12):
            orders=np.array([PERMS[rng.integers(6)] for _ in cache['bounds']]);routed=np.empty_like(output)
            for p,(a,b) in zip(orders,cache['bounds']):routed[:,a:b]=output[[*p,3],a:b]
            expected=direct(routed,ref,mix,count,count);actual=score(cache,orders,count,count)
            assert actual['assignment'][:count]==expected['assignment'][:count]
            for key in ('nmse','si_sdr','reference_power','sum_relative_error'):
                np.testing.assert_allclose(actual[key],expected[key],rtol=1e-9,atol=1e-9)
            rows.append(dict(count=count,trial=trial,passed=True))
    write_json(ROOT/'reports/2026-10-13/ROUTING_STATISTICS_CPU_CHECK.json',dict(status='PASS',cases=rows,
        source_sha256=sha256(Path(__file__).with_name('routing_statistics.py')),test_sha256=sha256(__file__),
        cache_contains_reference_statistics=True,rf_cache_reproduction_required=True))
    print('ROUTING_STATISTICS_PASS',len(rows),flush=True)


if __name__=='__main__':run()
