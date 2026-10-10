"""Test order, label and trainer resolution without creating a GPU worker."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('amendment', Path(__file__).with_name('serial_amendment.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExecutionOrderTests(unittest.TestCase):
    def test_only_pending_second_and_third_are_swapped(self):
        self.assertEqual(module.ORDER, ('septda', 'ordered_context', 'frozen_tf', 'balanced_head', 'wave_guard'))
        self.assertEqual(set(module.ORDER), set(module.ORIGINAL_ORDER))
        self.assertEqual(module.LABEL_BY_NAME['ordered_context'], '시간 순서 게이트')
        self.assertEqual(module.LABEL_BY_NAME['frozen_tf'], '본체 고정 시간·주파수 보강')

    def test_worker_is_frozen_trainer(self):
        self.assertEqual(Path(module.runner.__file__).with_name('train.py'), module.LEGACY/'train.py')
        self.assertTrue((module.LEGACY/'train.py').is_file())

    def test_frozen_runner_launches_in_amended_order(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            amendment = root/'amendment.json'
            amendment.write_text('{}')
            module.BASE_WRITE(root/'PROTOCOL.json', {'septda_run': str(root/'sep')})
            module.BASE_WRITE(root/'READY.json', {
                'protocol_sha256': module.runner.digest(root/'PROTOCOL.json'),
                'preflight_sha256': {}})
            for name in module.ORDER[1:]:
                (root/name).mkdir()
            finished = {'septda'}
            calls = []

            def launch(args, **kwargs):
                self.assertEqual(args[6], '/fake/python')
                self.assertEqual(args[7], str(module.LEGACY/'train.py'))
                self.assertEqual(args[8:10], ['--run', str(root)])
                name = args[-1]
                calls.append(name)
                finished.add(name)

                class FinishedProcess:
                    pid = 123
                    returncode = 0

                    def poll(self):
                        return 0
                return FinishedProcess()

            with patch.object(module, 'validate_amendment', return_value={}):
                module.configure(root, amendment)
            with patch.object(module.runner, 'verify'), \
                 patch.object(module.runner, 'check_complete', side_effect=lambda root, name, p: name in finished), \
                 patch.object(module.runner, 'progress'), \
                 patch.object(module.runner.subprocess, 'Popen', side_effect=launch):
                module.runner.run(root, '/fake/python')
            self.assertEqual(calls, list(module.ORDER[1:]))
            self.assertEqual(module.runner.read(root/'COMPLETE.json')['execution_order'], list(module.ORDER))

    def test_state_decorator_keeps_original_provenance(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            amendment = root/'amendment.json'
            amendment.write_text('{}')
            # No pinned model files or live state are touched by this unit test.
            with patch.object(module, 'validate_amendment', return_value={}):
                module.configure(root, amendment)
                module.runner.write(root/'QUEUE_STATE.json', {'status': 'WAITING_SEPTDA_50', 'pid': 123})
            state = module.runner.read(root/'QUEUE_STATE.json')
            self.assertEqual(state['next_candidate'], 'ordered_context')
            self.assertEqual(state['original_order'], list(module.ORIGINAL_ORDER))
            self.assertEqual(state['execution_order'], list(module.ORDER))
            self.assertEqual(state['pid'], 123)
            self.assertEqual(state['execution_amendment_sha256'], module.runner.digest(amendment))


if __name__ == '__main__':
    unittest.main()
