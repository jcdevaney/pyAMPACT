import numpy as np
import unittest

import sys
import os
sys.path.append(os.pardir)
from helpers.audioHelpers import viterbi_path


class TestViterbiPath(unittest.TestCase):

    def test_example_path(self):
        prior = np.array([0.5, 0.5])
        transmat = np.array([[0.7, 0.3], [0.4, 0.6]])
        obslik = np.array([[0.8, 0.2], [0.2, 0.8]])
        expected_path = np.array([0, 1])  # Expected Viterbi path

        path = viterbi_path(prior, transmat, obslik)

        self.assertTrue(np.array_equal(path, expected_path))              

if __name__ == '__main__':
    unittest.main()
