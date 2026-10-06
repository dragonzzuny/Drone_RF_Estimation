import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_raw_regions import check


class DataPolicyChecks(unittest.TestCase):
    def test_heldout_and_controller_files_are_not_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'inventory.json'
            p.write_text(json.dumps({'files':[
                {'aircraft_category':False,'role':'excluded_controller_or_nonaircraft','path':'does-not-exist'},
                {'aircraft_category':True,'role':'heldout_model_not_for_selection','path':'also-does-not-exist'}]}))
            result=check(p,root/'qc.json')
            self.assertEqual(result['files_read'],0)

    def test_false_aircraft_flag_cannot_admit_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'inventory.json'
            p.write_text(json.dumps({'files':[{'aircraft_category':True,'role':'train_pack',
                'category':'RadioMaster BOXER','path':'does-not-exist'}]}))
            with self.assertRaises(ValueError):check(p,root/'qc.json')


if __name__=='__main__':unittest.main()
