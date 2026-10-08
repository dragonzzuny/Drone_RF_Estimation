import unittest
import numpy as np
from scipy.optimize import nnls
from frame_spectral import frame_nnls


class FrameNNLSTests(unittest.TestCase):
    def test_matches_independent_scipy_solver(self):
        rng=np.random.default_rng(814)
        for k in (2,3):
            matrix=rng.uniform(size=(64,k));power=rng.uniform(size=(64,40))
            result=frame_nnls(matrix,power)
            expected=np.stack([nnls(matrix,power[:,i])[0] for i in range(40)],axis=1)
            np.testing.assert_allclose(result,expected,rtol=1e-8,atol=1e-10)

    def test_zero_input_and_single_active_source(self):
        matrix=np.eye(3)
        power=np.array([[0.,1.,0.],[0.,0.,0.],[0.,0.,4.]])
        np.testing.assert_allclose(frame_nnls(matrix,power),power,atol=1e-12)


if __name__=='__main__':unittest.main()
