"""Queue tests use mocked process identities/signals, never live processes."""
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from within_band_priority_queue import PriorityQueue
from drone_rf.data import sha256


def same(actual, expected):
    return bool(actual and actual['state'] not in ('Z', 'X') and
                all(actual.get(k) == expected[k] for k in ('pid','start_ticks','argv','exe')))


class PriorityQueueChecks(unittest.TestCase):
    def fixture(self, root):
        q=PriorityQueue.__new__(PriorityQueue)
        q.root=root/'new';q.q=q.root/'queue';q.previous=root/'old'
        q.q.mkdir(parents=True);(q.previous/'queue').mkdir(parents=True)
        old=dict(pid=101,start_ticks='1',argv=['scheduler'],exe='/python',state='S')
        worker=dict(pid=102,start_ticks='2',argv=['trainer'],exe='/python',state='R')
        q.request=dict(controller=old.copy(),worker=worker.copy())
        identities={101:old.copy(),102:worker.copy()}
        q.identity=lambda pid:identities.get(pid)
        q.same=same;q.check=Mock();q.state=Mock();q.common=Mock()
        (q.previous/'queue/STATE.json').write_text(json.dumps(dict(status='CONTEXT_THREE_RUNNING',child=worker)))
        def kill(pid,sig):
            if sig==signal.SIGSTOP:identities[pid]['state']='T'
            elif sig==signal.SIGTERM:identities[pid]['terminating']=True
            elif sig==signal.SIGCONT:
                if identities[pid].get('terminating'):identities.pop(pid)
                else:identities[pid]['state']='R'
        return q,identities,kill

    def test_adoption_retires_only_scheduler(self):
        with tempfile.TemporaryDirectory() as d:
            q,ids,kill=self.fixture(Path(d))
            with patch('within_band_priority_queue.os.kill',side_effect=kill) as signals:
                q.adopt()
            self.assertEqual([c.args[0] for c in signals.call_args_list],[101,101,101])
            self.assertEqual(ids[102]['state'],'R')
            self.assertTrue((q.q/'ADOPTION.json').exists())

    def test_pid_reuse_never_signals(self):
        with tempfile.TemporaryDirectory() as d:
            q,ids,kill=self.fixture(Path(d));ids[102]['start_ticks']='other'
            with patch('within_band_priority_queue.os.kill') as signals:
                with self.assertRaisesRegex(RuntimeError,'Pinned process changed'):q.adopt()
            signals.assert_not_called()

    def test_phase_race_restores_scheduler(self):
        with tempfile.TemporaryDirectory() as d:
            q,ids,kill=self.fixture(Path(d))
            (q.previous/'queue/STATE.json').write_text(json.dumps(dict(status='OTHER',child=q.request['worker'])))
            with patch('within_band_priority_queue.os.kill',side_effect=kill) as signals:
                with self.assertRaisesRegex(RuntimeError,'phase changed'):q.adopt()
            self.assertEqual([c.args[1] for c in signals.call_args_list],[signal.SIGSTOP,signal.SIGCONT])
            self.assertIn(101,ids);self.assertIn(102,ids)

    def stage(self,q):
        run=q.previous/'run';run.mkdir()
        (run/'COMPLETED_EPOCHS.json').write_text(json.dumps(dict(ordered=25,mean=25)))
        (run/'FREEZE.json').write_text('{}')
        for arm in ('ordered','mean'):
            folder=run/arm;folder.mkdir()
            (folder/'SELECTED_025.json').write_text('{}')
            (folder/'SELECTED_025.pt').write_bytes(b'selected-'+arm.encode())
            (folder/'STAGE_025.json').write_text(json.dumps(dict(checkpoint_sha256=sha256(folder/'SELECTED_025.pt'))))
            (folder/'EPOCH_025.json').write_text('{}')
            (folder/'LAST.pt').write_bytes(b'resume-'+arm.encode())
        return dict(epoch=25,updates=1875,freeze_sha256=sha256(run/'FREEZE.json'))

    def test_stage_requires_both_saved_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            q,_,_=self.fixture(Path(d));self.stage(q)
            self.assertTrue(q.stage_ready())
            (q.previous/'run/mean/SELECTED_025.pt').unlink()
            self.assertFalse(q.stage_ready())

    def test_park_archives_resume_files_before_retirement(self):
        with tempfile.TemporaryDirectory() as d:
            q,ids,kill=self.fixture(Path(d));saved=self.stage(q)
            def guarded_kill(pid,sig):
                if pid==102 and sig==signal.SIGTERM:
                    for arm in ('ordered','mean'):
                        self.assertEqual((q.q/'old_resume_checkpoints'/f'{arm}_LAST.pt').read_bytes(),
                                         (q.previous/'run'/arm/'LAST.pt').read_bytes())
                kill(pid,sig)
            with patch('within_band_priority_queue.os.kill',side_effect=guarded_kill),patch('torch.load',return_value=saved):
                q.park_old(q.request['worker'])
            self.assertNotIn(102,ids)
            self.assertTrue((q.q/'OLD_PARKED.json').exists())

    def test_checkpoint_mismatch_leaves_worker_running(self):
        with tempfile.TemporaryDirectory() as d:
            q,ids,kill=self.fixture(Path(d));saved=self.stage(q);saved['epoch']=26
            with patch('within_band_priority_queue.os.kill',side_effect=kill) as signals,patch('torch.load',return_value=saved),patch('within_band_priority_queue.time.sleep'):
                with self.assertRaisesRegex(RuntimeError,'No coherent checkpoint'):q.park_old(q.request['worker'])
            self.assertEqual(ids[102]['state'],'R')
            self.assertTrue(all(c.args[1]!=signal.SIGTERM for c in signals.call_args_list))


if __name__=='__main__':unittest.main()
