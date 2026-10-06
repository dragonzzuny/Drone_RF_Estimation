import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from drone_rf.similarity import shifted_coherence
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cpu_corpus import admitted_files,extract_stream


class CPUPreparationChecks(unittest.TestCase):
    def test_shift_phase_and_gain_recovered(self):
        rng=np.random.default_rng(1);y=rng.normal(size=4096)+1j*rng.normal(size=4096)
        x=np.zeros_like(y);x[37:]=2*np.exp(.7j)*y[:-37]+3j
        value=shifted_coherence(x,y,64)
        self.assertEqual(value['lag_samples'],37)
        self.assertGreater(value['rho_squared'],1-1e-12)

    def test_fft_matches_direct_overlap_calculation(self):
        rng=np.random.default_rng(2);x=rng.normal(size=200)+1j*rng.normal(size=200)
        y=rng.normal(size=200)+1j*rng.normal(size=200);values=[]
        for lag in range(-30,31):
            a=x[max(lag,0):200+min(lag,0)];b=y[max(-lag,0):200-max(lag,0)]
            a=a-a.mean();b=b-b.mean()
            values.append(abs(np.vdot(b,a))**2/(np.vdot(a,a).real*np.vdot(b,b).real))
        got=shifted_coherence(x,y,30)
        self.assertAlmostEqual(got['rho_squared'],max(values),places=12)
        self.assertEqual(got['lag_samples'],int(np.argmax(values))-30)

    def test_constant_input_not_declared_duplicate(self):
        value=shifted_coherence(np.ones(64),np.ones(64),16)
        self.assertEqual(value['rho_squared'],0)
        self.assertFalse(value['variance_defined'])

    def test_stream_hash_and_byte_exact_intervals(self):
        data=np.arange(1000,dtype=np.float32).view(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'source.iq';p.write_bytes(data.tobytes())
            got,buffers=extract_stream(p,[0,137,470],30,p.stat().st_size)
            self.assertEqual(got,hashlib.sha256(p.read_bytes()).hexdigest())
            for offset,raw in zip([0,137,470],buffers):
                self.assertEqual(bytes(raw),data[offset:offset+30].tobytes())

    def test_reject_pack_shared_between_roles(self):
        base=dict(aircraft_category=True,category='DJI AVATA2',data_type='Complex Float',
            fs_hz=100_000_000,bytes=96_000_000,samples_cf32=12_000_000,pack_id='same',relative_path='one')
        rows=[dict(base,path='one',role='train_pack'),dict(base,path='two',role='validation_pack')]
        with self.assertRaises(ValueError):admitted_files({'files':rows})

    def test_controller_and_holdout_excluded(self):
        valid=dict(aircraft_category=True,category='DJI AVATA2',data_type='Complex Float',role='train_pack',
            fs_hz=100_000_000,bytes=96_000_000,samples_cf32=12_000_000,pack_id='pack',path='one',relative_path='one')
        rows=[valid,dict(aircraft_category=False),dict(aircraft_category=True,role='heldout_model_not_for_selection')]
        self.assertEqual(admitted_files({'files':rows}),[valid])

    def test_short_source_cannot_produce_overlapping_quarter_clips(self):
        row=dict(aircraft_category=True,category='DJI AVATA2',data_type='Complex Float',role='train_pack',
            fs_hz=100_000_000,bytes=80_000_000,samples_cf32=10_000_000,pack_id='pack',path='one',relative_path='one')
        with self.assertRaises(ValueError):admitted_files({'files':[row]})


if __name__=='__main__':unittest.main()
