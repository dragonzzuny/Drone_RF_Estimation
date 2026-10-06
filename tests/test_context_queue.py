"""Mock process transitions: tests never signal a real process."""
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from context_execution_queue import Queue


def same(actual, expected):
    return bool(actual and actual['state'] not in ('Z', 'X') and
                all(actual.get(k) == expected[k] for k in ('pid', 'start_ticks', 'argv', 'exe')))


class QueueAdoptionChecks(unittest.TestCase):
    def fixture(self, root):
        scheduler = dict(pid=101, start_ticks='10', state='S', argv=['scheduler'], exe='/python')
        worker = dict(pid=102, start_ticks='11', state='R', argv=['magnitude'], exe='/python')
        state = dict(status='FULL_N_MAGNITUDE_RUNNING', controller=scheduler, child=worker)
        (root / 'STATE.json').write_text(json.dumps(state))
        queue = Queue.__new__(Queue)
        queue.root = queue.legacy = root
        queue.common = Mock()
        queue.check = Mock(return_value={'previous_state': state})
        identities = {101: scheduler.copy(), 102: worker.copy()}
        queue.identity = lambda pid: identities.get(pid)
        queue.same = same
        queue.state = Mock()
        return queue, identities

    def test_retire_only_scheduler_and_keep_worker_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            queue, identities = self.fixture(Path(directory))
            def kill(pid, sig):
                self.assertEqual(pid, 101)
                if sig == signal.SIGSTOP:
                    identities[pid]['state'] = 'T'
                if sig == signal.SIGCONT:
                    identities.pop(pid, None)
            with patch('context_execution_queue.os.kill', side_effect=kill) as signals:
                worker = queue.adopt()
            self.assertEqual(worker['pid'], 102)
            self.assertEqual([call.args[1] for call in signals.call_args_list],
                             [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT])
            self.assertEqual(identities[102]['state'], 'R')
            receipt = json.loads((Path(directory) / 'ADOPTION.json').read_text())
            self.assertFalse(receipt['worker_signalled'])
            self.assertEqual(receipt['worker_before'], receipt['worker_after'])

    def test_phase_race_resumes_scheduler_without_retiring_or_touching_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue, identities = self.fixture(root)
            def kill(pid, sig):
                self.assertEqual(pid, 101)
                if sig == signal.SIGSTOP:
                    identities[pid]['state'] = 'T'
                    state = json.loads((root / 'STATE.json').read_text())
                    state['status'] = 'ANOTHER_PHASE'
                    (root / 'STATE.json').write_text(json.dumps(state))
                if sig == signal.SIGCONT:
                    identities[pid]['state'] = 'S'
            with patch('context_execution_queue.os.kill', side_effect=kill) as signals:
                with self.assertRaisesRegex(RuntimeError, 'phase changed'):
                    queue.adopt()
            self.assertEqual([call.args[1] for call in signals.call_args_list],
                             [signal.SIGSTOP, signal.SIGCONT])
            self.assertFalse((root / 'ADOPTION.json').exists())

    def test_pid_reuse_is_rejected_before_any_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            queue, identities = self.fixture(Path(directory))
            identities[102]['start_ticks'] = '9999'
            with patch('context_execution_queue.os.kill') as signals:
                with self.assertRaisesRegex(RuntimeError, 'changed before adoption'):
                    queue.adopt()
            signals.assert_not_called()


if __name__ == '__main__':
    unittest.main()
