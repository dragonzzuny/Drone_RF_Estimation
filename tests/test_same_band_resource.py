import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from resume_same_band_resource import admit_display


class DisplayResourceChecks(unittest.TestCase):
    def setUp(self):
        self.expected=dict(pid=10,start_ticks='1',argv=['/display','--server'],exe='/display',state='S')
        self.result=dict(allowed=False,reasons=['UNRECOGNIZED_OR_OVERSIZED_GPU_PROCESS'],free_mib=7200,
            blocking_processes=[dict(identity=self.expected.copy(),memory_mib=166)])

    def test_only_pinned_small_display_is_admitted(self):
        result=admit_display(self.result,self.expected,256)
        self.assertTrue(result['allowed']);self.assertFalse(self.result['allowed'])

    def test_pid_reuse_is_not_admitted(self):
        self.result['blocking_processes'][0]['identity']['start_ticks']='2'
        self.assertFalse(admit_display(self.result,self.expected,256)['allowed'])

    def test_large_or_nonfinite_memory_is_not_admitted(self):
        for memory in [257,float('nan'),float('inf'),-1]:
            with self.subTest(memory=memory):
                self.result['blocking_processes'][0]['memory_mib']=memory
                self.assertFalse(admit_display(self.result,self.expected,256)['allowed'])

    def test_other_resource_failures_are_preserved(self):
        for reason in ['INSUFFICIENT_FREE_MEMORY','GPU_IDENTITY_CHANGED','RESOURCE_QUERY_UNAVAILABLE']:
            result=copy.deepcopy(self.result);result['reasons'].append(reason)
            self.assertFalse(admit_display(result,self.expected,256)['allowed'])

    def test_other_compute_process_is_not_admitted(self):
        other=copy.deepcopy(self.result['blocking_processes'][0]);other['identity']['pid']=20
        self.result['blocking_processes'].append(other)
        self.assertFalse(admit_display(self.result,self.expected,256)['allowed'])


if __name__=='__main__':unittest.main()
