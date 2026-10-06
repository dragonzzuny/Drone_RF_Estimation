import unittest
import itertools
import numpy as np
from drone_rf.mixture_constraints import validate_sources
from drone_rf.within_dataset_schedule import make_within_dataset_schedule


class MixtureConstraintChecks(unittest.TestCase):
    def source(self, dataset='RFUAV', center=2.44e9, fs=1e8, **extra):
        return dict(dataset=dataset,center_hz=center,fs_hz=fs,**extra)

    def test_resampled_other_dataset_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'Cross-dataset'):
            validate_sources([self.source(),self.source(dataset='DroneDetect_V2')])

    def test_same_dataset_different_band_rejected(self):
        with self.assertRaisesRegex(ValueError,'Cross-band'):
            validate_sources([self.source(),self.source(center=5.8e9)])

    def test_rate_and_receiver_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'sample rates'):
            validate_sources([self.source(),self.source(fs=6e7)])
        with self.assertRaisesRegex(ValueError,'receiver models'):
            validate_sources([self.source(receiver_model='A'),self.source(receiver_model='B')])

    def test_same_band_with_different_centers_is_explicitly_allowed(self):
        result=validate_sources([self.source(),self.source(center=2.4375e9)])
        self.assertEqual(result['dataset'],'RFUAV')
        self.assertEqual(result['rf_band'],'2.4GHz')

    def test_invalid_metadata_rejected(self):
        for sources in [[],[self.source()]*4,[self.source(center=float('nan'))],
                        [self.source(center=915e6)],[self.source(fs=0)],[self.source(dataset='')]]:
            with self.subTest(sources=sources),self.assertRaises(ValueError):
                validate_sources(sources)

    def test_schedule_respects_roles_band_and_reproducibility(self):
        clips=[dict(self.source(center=f),source_id=k,role=role,samples=100)
            for role in ('train_pack','validation_pack')
            for k,f in [('A',5.76e9),('B',5.77e9),('C',5.8e9),('D',2.44e9),('E',2.46e9)]]
        allowed=[dict(sources=list(c)) for names in [('A','B','C'),('D','E')]
                 for n in (1,2,3) for c in itertools.combinations(names,n)]
        args=dict(epochs=2,examples_per_count=12,length=50)
        first=make_within_dataset_schedule(clips,allowed,'train_pack',**args)
        second=make_within_dataset_schedule(clips,allowed,'train_pack',**args)
        self.assertTrue(np.array_equal(first,second))
        validation=make_within_dataset_schedule(clips,allowed,'validation_pack',**args)
        for rows,role in [(first,'train_pack'),(validation,'validation_pack')]:
            self.assertEqual(len(rows),72)
            for row in rows:
                chosen=[clips[int(i)] for i in row['indices'][:int(row['count'])]]
                validate_sources(chosen)
                self.assertTrue(all(c['role']==role for c in chosen))
                self.assertLessEqual(row['crop_start']+50,100)
                if row['count']==3:self.assertEqual({c['source_id'] for c in chosen},{'A','B','C'})

    def test_approved_class_cannot_bypass_dataset_rule(self):
        clips=[dict(self.source(dataset=d),source_id=k,role='train_pack',samples=100)
               for k,d in [('A','RFUAV'),('B','RFUAV'),('C','Other')]]
        allowed=[dict(sources=list(c)) for n in (1,2,3) for c in itertools.combinations('ABC',n)]
        with self.assertRaisesRegex(ValueError,'Cross-dataset'):
            make_within_dataset_schedule(clips,allowed,'train_pack',epochs=1,examples_per_count=3,length=50)


if __name__=='__main__':
    unittest.main()
