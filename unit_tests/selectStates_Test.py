import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from helpers.alignmentHelpers import select_states

class TestSelectStates(unittest.TestCase):
    def test_select_states(self):
        startingState = np.array([[0.1, 0.2], [0.3, 0.4]])
        prior = np.array([0.5, 0.6, 0.7, 0.8])
        trans = np.array([[0.9, 0.1], [0.2, 0.8]])
        meansFull = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        covarsFull = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                              [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]])
        mixmat = np.array([[0.4, 0.6], [0.7, 0.3]])
        obs = np.array([[0.2, 0.5, 0.8], [0.3, 0.6, 0.9]])
        stateO = np.array([1, 0, 1, 0])
        noteNum = 2
        sr = 44100

        like2, histvals2, cumsumvals2 = select_states(startingState, prior, trans, meansFull, covarsFull, mixmat, obs, stateO, noteNum, sr)

        # Add your assertions here based on expected results
        self.assertTrue(np.allclose(cumsumvals2, [0.0, 0.09319728, 0.18639456]))

if __name__ == '__main__':
    unittest.main()
