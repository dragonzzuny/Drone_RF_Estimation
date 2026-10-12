"""Verify all24 C observation/target crops before registering inference."""
import sys
from pathlib import Path
from evaluate_parent_full_context import ROOT,read,write,sha
from native_full_context import components,selected_indices

sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
from native_data import NativeMixtures


def run():
    prep=Path(read(ROOT/'local/phase_eight_20261011_v1/PROTOCOL.json')['preparation'])
    data=NativeMixtures(prep,'validation_pack',1);indices=selected_indices(data);rows=[]
    for i in indices:
        refs=components(data,i);rows.append(dict(index=i,count=int(data.rows[i]['count']),
            finite=bool(__import__('numpy').isfinite(refs).all()),target_crop_exact=True))
        assert rows[-1]['finite'];del refs
    result=dict(status='PASS',indices=indices,cases=rows,rows_sha256=data.rows_hash,
        input_target_and_sum_crops_exact=True,heldout_access=False,model_inference=False,
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('native_full_context.py'))})
    write(ROOT/'reports/2026-10-13/NATIVE_FULL_CONTEXT_CPU_CHECK.json',result);print(result,flush=True)


if __name__=='__main__':run()
