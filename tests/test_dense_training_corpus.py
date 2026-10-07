import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_dense_training_corpus import dense_offsets, stream_contexts, training_sources


class DenseTrainingChecks(unittest.TestCase):
    def test_every_sample_covered_without_duplicate_contexts(self):
        for samples in (8,9,15,16,17,45,100):
            offsets=dense_offsets(samples,8)
            self.assertEqual(len(offsets),len(set(offsets)))
            covered=set()
            for offset in offsets:
                self.assertGreaterEqual(offset,0)
                self.assertLessEqual(offset+8,samples)
                covered.update(range(offset,offset+8))
            self.assertEqual(covered,set(range(samples)))
        self.assertEqual(dense_offsets(16,8),[0,8])
        self.assertEqual(dense_offsets(17,8),[0,8,9])

    def test_stream_emits_exact_full_and_tail_contexts_and_complete_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.iq'
            for samples in (8,9,16,17,45):
                data=(np.arange(samples)+1j*np.arange(samples)[::-1]).astype('<c8')
                raw=data.tobytes();path.write_bytes(raw);emitted=[]
                digest=stream_contexts(path,samples,8,lambda off,block:emitted.append((off,block)))
                self.assertEqual(digest,hashlib.sha256(raw).hexdigest())
                self.assertEqual([off for off,_ in emitted],dense_offsets(samples,8))
                for off,block in emitted:
                    self.assertEqual(block,data[off:off+8].tobytes())

    def test_changed_size_and_short_context_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.iq'
            path.write_bytes(np.zeros(17,dtype='<c8').tobytes())
            for expected in (16,18,7):
                with self.assertRaises(ValueError):
                    stream_contexts(path,expected,8,lambda *args:None)

    def sample(self):
        source=dict(relative_path='known.iq',path='known.iq',pack_id='train.xml',role='train_pack',
            category='DJI AVATA2',data_type='Complex Float',fs_hz=100000000,center_hz=5.8e9,
            samples_cf32=5000000,bytes=40000000,mtime_ns=1)
        clip={k:source[k] for k in ('pack_id','role','category','fs_hz','center_hz')}
        clip.update(source_path='known.iq',source_sha256='verified_prior_sha')
        return source,clip

    def test_only_existing_training_sources_selected(self):
        source,clip=self.sample()
        inventory={'files':[source,dict(relative_path='heldout.iq',role='heldout_model_not_for_selection'),
                            dict(relative_path='controller.iq',role='excluded_controller_or_nonaircraft')]}
        selected=training_sources(inventory,[clip])
        self.assertEqual([r['relative_path'] for r in selected],['known.iq'])
        self.assertEqual(selected[0]['expected_sha256'],'verified_prior_sha')
        for role in ('validation_pack','heldout_model_not_for_selection'):
            changed=copy.deepcopy(inventory);changed['files'][0]['role']=role
            with self.assertRaisesRegex(ValueError,'role'):
                training_sources(changed,[clip])

    def test_parent_role_leakage_missing_file_and_conflicting_hash_rejected(self):
        source,clip=self.sample();inventory={'files':[source]}
        with self.assertRaisesRegex(ValueError,'crosses roles'):
            training_sources(inventory,[clip,dict(clip,role='validation_pack')])
        with self.assertRaisesRegex(ValueError,'Missing'):
            training_sources({'files':[]},[clip])
        with self.assertRaisesRegex(ValueError,'hashes'):
            training_sources(inventory,[clip,dict(clip,source_sha256='different')])


if __name__=='__main__':unittest.main()
