"""CPU data/evaluation checks; no model training or new validation I/Q access."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import phase_data as adapter
import phase_evaluation as evaluation
import phase_packing as pp
from native_data import NativeMixtures, sha256, write_json


def fixture(count):
    torch.manual_seed(19+count)
    refs=torch.randn(1,3,128,dtype=torch.complex64)
    refs[:,count:]=0
    active=torch.arange(3)[None]<count
    mixture=refs.sum(1)
    estimates=refs[:,[2,0,1]]+.01*torch.randn_like(refs)
    estimates=torch.cat((estimates,(mixture-estimates.sum(1))[:,None]),1)
    logits=torch.zeros(1,3);logits[0,count-1]=3
    item=dict(mixture=mixture,references=refs,active=active,construction_count=torch.tensor([count]))
    source={'levels':np.zeros(3)}
    clips=[dict(category=str(i),pack_id=str(i)) for i in range(count)]
    row=evaluation.row_metrics(dict(estimates=estimates,count_logits=logits),item,source,clips,0)
    expected=[1,2,0][:count]
    assert row['assignment'][:count]==expected
    assert row['predicted_count']==count
    assert row['sum_relative_error']<1e-9
    for i,slot in enumerate(expected):
        reference=refs[0,i].numpy().astype(np.complex128)
        estimate=estimates[0,slot].numpy().astype(np.complex128)
        manual_nmse=np.sum(abs(estimate-reference)**2)/np.sum(abs(reference)**2)
        r,e=reference-reference.mean(),estimate-estimate.mean()
        projection=np.vdot(r,e)/np.vdot(r,r)*r
        manual_si=10*np.log10(np.sum(abs(projection)**2)/np.sum(abs(e-projection)**2))
        np.testing.assert_allclose(row['nmse'][i],manual_nmse,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(row['si_sdr'][i],manual_si,rtol=1e-10,atol=1e-10)
    return dict(count=count,active_assignment=expected,manual_complex_metrics_agree=True)


def run(preparation,output):
    torch.set_num_threads(2)
    began=time.time()
    fixtures=[fixture(c) for c in (1,2,3)]
    checked=[]
    for epoch in range(1,6):
        source=NativeMixtures(preparation,'train_pack',epoch)
        for count in (1,2,3):
            for index in np.flatnonzero(source.rows['count']==count)[:3]:
                item=adapter.example(source,int(index))
                assert item['long_mixture'].shape==(pp.SAMPLES,)
                assert item['references'].shape==(3,pp.FINE)
                assert int(item['construction_count'])==count
                np.testing.assert_array_equal(item['references'].sum(0),item['mixture'])
                values=adapter.batch(item,'cpu')
                assert values['long_mixture'].shape==(1,pp.SAMPLES)
                assert values['crop_start'].dtype==torch.int64
                class Spy:
                    def __call__(self,*args):
                        assert len(args)==3
                        assert args[0] is values['long_mixture']
                        assert args[1] is values['context_features']
                        assert args[2] is values['crop_start']
                        return 'observed inputs only'
                assert adapter.predict(Spy(),values)=='observed inputs only'
                checked.append(dict(epoch=epoch,index=int(index),count=count,crop_start=int(item['crop_start'])))
        del source
    result=dict(status='PASS',device='cpu',trained=False,validation_iq_read=False,heldout_read=False,
        checked_train_examples=checked,full_mixture_fine_crop_exact=True,observed_forward_whitelist=True,
        waveform_metric_fixtures=fixtures,source_sha256={p.name:sha256(p) for p in
            (Path(__file__),Path(adapter.__file__),Path(evaluation.__file__),Path(pp.__file__))},
        preparation_sha256=sha256(preparation/'PREPARATION.json'),seconds=time.time()-began)
    write_json(output,result)
    print(json.dumps(dict(status=result['status'],train_cases=len(checked),seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.preparation.resolve(),args.output)
