"""No host signals or GPU use. Verify transition guards and saved-state checks."""
import json
from pathlib import Path
import signal
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import host_switch as h


class HostSwitchTests(unittest.TestCase):
    def test_pinned_display_does_not_hide_other_gpu_work(self):
        expected = dict(pid=2001, start_ticks='4', state='S',
                        exe='/usr/share/rustdesk/rustdesk',
                        argv=['/usr/share/rustdesk/rustdesk', '--server'])
        policy = dict(identity=expected, maximum_memory_mib=256)
        with patch.object(h, 'gpu_pids', return_value={1002, 2001, 3001}), \
             patch.object(h, 'identity', return_value=expected), \
             patch.object(h.Path, 'stat', return_value=SimpleNamespace(st_uid=h.os.getuid())), \
             patch.object(h.subprocess, 'run', return_value=SimpleNamespace(stdout='2001, 166\n')):
            self.assertEqual(h.unexpected_gpu_pids({1002}, policy), {3001})

    def test_display_reuse_or_excess_memory_is_rejected(self):
        expected = dict(pid=2001, start_ticks='4', state='S',
                        exe='/usr/share/rustdesk/rustdesk',
                        argv=['/usr/share/rustdesk/rustdesk', '--server'])
        policy = dict(identity=expected, maximum_memory_mib=256)
        with patch.object(h, 'gpu_pids', return_value={2001}), \
             patch.object(h, 'identity', return_value=dict(expected, start_ticks='5')):
            with self.assertRaises(RuntimeError):
                h.unexpected_gpu_pids(set(), policy)
        with patch.object(h, 'gpu_pids', return_value={2001}), \
             patch.object(h, 'identity', return_value=expected), \
             patch.object(h.Path, 'stat', return_value=SimpleNamespace(st_uid=h.os.getuid())), \
             patch.object(h.subprocess, 'run', return_value=SimpleNamespace(stdout='2001, 257\n')):
            with self.assertRaises(RuntimeError):
                h.unexpected_gpu_pids(set(), policy)

    def test_rejects_other_queue_phase_before_process_access(self):
        with patch.object(h, 'identity', side_effect=AssertionError('Must not inspect unrelated process')):
            with self.assertRaises(RuntimeError):
                h.check_queue({'status': 'NATIVE_RF_RESUME_RUNNING'})

    def test_refuses_signal_to_reused_pid(self):
        expected = dict(pid=999999, start_ticks='123', argv=['python', 'known.py'], exe='/python', state='S')
        changed = dict(expected, start_ticks='124')
        with patch.object(h, 'identity', return_value=changed), patch.object(h, 'pidfd_open') as opener:
            with self.assertRaises(RuntimeError):
                h.checked_signal(expected, signal.SIGSTOP)
            opener.assert_not_called()

    def test_pidfd_permission_probe_on_self_only(self):
        fd = h.pidfd_open(h.os.getpid())
        try:
            h.pidfd_signal(fd, 0)
        finally:
            h.os.close(fd)

    def test_checkpoint_backup_and_incoherent_epoch_rejection(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            old, backup = root / 'old', root / 'backup'
            run = old / 'run'
            run.mkdir(parents=True)
            backup.mkdir()
            (run / 'FREEZE.json').write_text('{}')
            (run / 'PROGRESS.json').write_text('{}')
            (run / 'COMPLETED_EPOCHS.json').write_text(json.dumps({'ordered': 3, 'mean': 3}))
            for arm in ('ordered', 'mean'):
                folder = run / arm
                folder.mkdir()
                (folder / 'EPOCH_003.json').write_text('{}')
                torch.save(dict(epoch=3, updates=225, freeze_sha256=h.digest(run / 'FREEZE.json'),
                                optimizer={}, torch_rng=torch.get_rng_state(), cuda_rng=[]), folder / 'LAST.pt')
            with patch.object(h, 'OLD', old):
                records = h.preserve_checkpoints(backup)
                self.assertEqual(set(records), {'ordered', 'mean'})
                for item in records.values():
                    self.assertEqual(h.digest(Path(item['backup'])), h.digest(Path(item['original'])))
                (run / 'COMPLETED_EPOCHS.json').write_text(json.dumps({'ordered': 4, 'mean': 3}))
                with self.assertRaises(RuntimeError):
                    h.preserve_checkpoints(backup)

    def test_failed_backup_resumes_paused_processes(self):
        controller, worker = {'pid': 1001}, {'pid': 1002}
        with TemporaryDirectory() as temp, \
             patch.object(h, 'RUN', Path(temp)), \
             patch.object(h, 'read', return_value={}), \
             patch.object(h, 'check_queue', return_value=(controller, worker)), \
             patch.object(h, 'gpu_pids', return_value={1002}), \
             patch.object(h, 'stop') as stop, \
             patch.object(h, 'preserve_checkpoints', side_effect=RuntimeError('bad receipt')), \
             patch.object(h, 'retire') as retire, \
             patch.object(h, 'identity', return_value={}), \
             patch.object(h, 'matches', return_value=True), \
             patch.object(h, 'checked_signal') as send, \
             patch.object(h, 'status'):
            with self.assertRaises(RuntimeError):
                h.transition()
            self.assertEqual(stop.call_count, 2)
            retire.assert_not_called()
            self.assertEqual(send.call_args_list[0].args, (worker, signal.SIGCONT))
            self.assertEqual(send.call_args_list[1].args, (controller, signal.SIGCONT))

    def test_stop_timeout_also_restores_controller(self):
        controller, worker = {'pid': 1001}, {'pid': 1002}
        with TemporaryDirectory() as temp, \
             patch.object(h, 'RUN', Path(temp)), \
             patch.object(h, 'read', return_value={}), \
             patch.object(h, 'check_queue', return_value=(controller, worker)), \
             patch.object(h, 'gpu_pids', return_value={1002}), \
             patch.object(h, 'stop', side_effect=RuntimeError('stop timeout')), \
             patch.object(h, 'identity', return_value={}), \
             patch.object(h, 'matches', return_value=True), \
             patch.object(h, 'checked_signal') as send, \
             patch.object(h, 'status'):
            with self.assertRaises(RuntimeError):
                h.transition()
            send.assert_called_once_with(controller, signal.SIGCONT)


if __name__ == '__main__':
    unittest.main(verbosity=2)
