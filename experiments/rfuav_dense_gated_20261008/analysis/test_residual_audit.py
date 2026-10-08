"""Numerical diagnostic checks, including correlated sources and slot switches."""
import unittest
import numpy as np

from fit_residual_audit import decompose, chunk_assignment_diagnostic


class ResidualAuditTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.refs = self.rng.normal(size=(3,4096))+1j*self.rng.normal(size=(3,4096))

    def test_nonorthogonal_closure_keeps_cross_term(self):
        self.refs[1] += .4*self.refs[0]
        estimate=.7*self.refs[0]+(.2+.3j)*self.refs[1]
        value=decompose(estimate,self.refs,0)
        self.assertLess(abs(value['closure_error']),1e-10)
        self.assertGreater(abs(value['cross_term_nmse']),.01)
        self.assertLess(value['outside_reference_span_nmse'],1e-20)

    def test_complex_oracle_does_not_claim_phase_gain_is_positive_gain(self):
        value=decompose(1.5j*self.refs[0],self.refs,0)
        self.assertLess(value['oracle_complex_gain_nmse'],1e-20)
        self.assertAlmostEqual(value['oracle_positive_gain_nmse'],1.)

    def test_zero_estimate(self):
        value=decompose(np.zeros_like(self.refs[0]),self.refs,0)
        self.assertAlmostEqual(value['nmse'],1.)
        self.assertAlmostEqual(value['oracle_complex_gain_nmse'],1.)

    def test_oracle_chunk_assignment_detects_constructed_switch(self):
        estimates=self.refs.copy()
        estimates[:,2048:]=self.refs[[1,2,0],2048:]
        value=chunk_assignment_diagnostic(estimates,self.refs,1024)
        self.assertEqual(value['changed_chunks'],2)
        self.assertGreater(value['fixed_nmse'],0)
        self.assertEqual(value['oracle_chunk_assignment_nmse'],0)

    def test_correct_assignment_has_no_oracle_improvement(self):
        value=chunk_assignment_diagnostic(self.refs,self.refs,1024)
        self.assertEqual(value['oracle_relative_reduction'],0)
        self.assertEqual(value['changed_chunks'],0)


if __name__=='__main__':
    unittest.main()
