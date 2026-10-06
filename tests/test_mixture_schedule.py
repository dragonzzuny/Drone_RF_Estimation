from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_mixture_schedule import schedule,LENGTH


class ScheduleChecks(unittest.TestCase):
    def test_deterministic_balanced_roles_and_valid_crops(self):
        clips=[dict(category=c,role=r,samples=2097152) for r in ('train_pack','validation_pack') for c in ('A','B','C')]
        train=schedule(clips,'train_pack',2,4);again=schedule(clips,'train_pack',2,4)
        validation=schedule(clips,'validation_pack',1,4)
        self.assertTrue(np.array_equal(train,again))
        self.assertEqual(len(train),72)
        self.assertTrue(np.all(train['first']<3) and np.all(train['second']<3))
        self.assertTrue(np.all(validation['first']>=3) and np.all(validation['second']>=3))
        for epoch in (1,2):
            values,counts=np.unique(train[train['epoch']==epoch]['sir_db'],return_counts=True)
            self.assertEqual(values.tolist(),[-10,0,10]);self.assertEqual(counts.tolist(),[12,12,12])
        self.assertTrue(np.all(train['crop_start']>=0) and np.all(train['crop_start']+LENGTH<=2097152))

    def test_missing_pair_and_short_source_rejected(self):
        with self.assertRaises(ValueError):schedule([dict(category='A',role='train_pack',samples=2097152)],'train_pack')
        clips=[dict(category=c,role='train_pack',samples=10) for c in ('A','B')]
        with self.assertRaises(ValueError):schedule(clips,'train_pack')


if __name__=='__main__':unittest.main()
