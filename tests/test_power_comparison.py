import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_power_trial import check_pair, summarize


class MatchedPowerChecks(unittest.TestCase):
    def fixture(self, root):
        folders = [root / name for name in ('baseline', 'trial')]
        for folder in folders:
            ev = folder / 'evaluation_25'
            ev.mkdir(parents=True)
            selection = dict(completed_epochs=25, updates=1875, best_sha256='selected', parent_checkpoint_sha256='parent')
            (folder / 'STAGE_25.json').write_text(json.dumps(selection))
            (ev / 'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE', canonical_metrics=True,
                records=12, selected_checkpoint_sha256='selected')))
            records = []
            for sir in (-10, 0, 10):
                for i in range(4):
                    records.append(dict(case_id=f'{sir}_{i}', day='D1', pair='A_B', sir_db=sir, window=i,
                        status='SCORED', stage=25, nmse=[.2, .4], mean_complex_nmse=.3,
                        input_si_db=[float(sir), -float(sir)], weak_source_index=0 if sir < 0 else 1 if sir > 0 else None,
                        pit_nmse=[.2, .4], both_gain_positive=True, si_status=['finite', 'finite'], si_db=[1., 2.]))
            (ev / 'RECORDS.json').write_text(json.dumps(records))
            (ev / 'SOURCE_RECEIPTS.json').write_text(json.dumps([dict(raw_crop_sha256='same')]))
        return folders

    def test_accept_roundoff_and_reject_different_input(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = self.fixture(Path(temp))
            path = b / 'evaluation_25/RECORDS.json'
            records = json.loads(path.read_text())
            records[0]['input_si_db'][0] += 2e-13
            path.write_text(json.dumps(records))
            rows, _ = check_pair(a, b, 25)
            self.assertAlmostEqual(summarize(rows[0])['overall_nmse'], .3)
            records[0]['input_si_db'][0] += .01
            path.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'Input SI-SDR'):
                check_pair(a, b, 25)

    def test_reject_different_parent_budget_or_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = self.fixture(Path(temp))
            path = b / 'STAGE_25.json'
            original = json.loads(path.read_text())
            for key, value in [('parent_checkpoint_sha256', 'other'), ('updates', 1)]:
                changed = dict(original, **{key: value})
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    check_pair(a, b, 25)
            path.write_text(json.dumps(original))
            (b / 'evaluation_25/SOURCE_RECEIPTS.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Source receipts'):
                check_pair(a, b, 25)


if __name__ == '__main__':
    unittest.main()
