# FAIL
import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from audioHelpers import fill_trans_mat

class TestFillTransMat(unittest.TestCase):
    
    # The issue here is that the expected result fails, but the MATLAB code fails with the same inputs:
    # Unable to perform assignment because the size of the left side is 1-by-1 and the size of the right side is 1-by-4.
    # Line 35 in MATLAB, Line 12 here

    def test_fill_trans_mat(self):
        # Test case 1: Ensure the function works with a small trans_seed and notes
        trans_seed = np.array([[1, 2], [3, 4]])
        notes = 2
        result = fill_trans_mat(trans_seed, notes)        
        expected_result = np.array([[4, 0, 2, 0],
                                     [0, 4, 0, 2],
                                     [3, 0, 1, 0],
                                     [0, 3, 0, 1]])
        
        self.assertTrue(np.array_equal(result, expected_result))

        # Test case 2: Ensure the function works with a larger trans_seed and notes
        trans_seed = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        notes = 3
        result = fill_trans_mat(trans_seed, notes)
        expected_result = np.array([[5, 6, 0, 0, 0, 2, 3, 0, 0],
                                     [0, 5, 6, 0, 0, 0, 2, 3, 0],
                                     [0, 0, 5, 6, 0, 0, 0, 2, 3],
                                     [8, 9, 0, 1, 2, 0, 0, 0, 7],
                                     [0, 8, 9, 0, 1, 2, 0, 0, 0],
                                     [0, 0, 8, 9, 0, 1, 2, 0, 0],
                                     [0, 0, 0, 7, 8, 0, 1, 2, 0],
                                     [3, 0, 0, 0, 7, 8, 0, 1, 2],
                                     [0, 3, 0, 0, 0, 7, 8, 0, 1]])
        self.assertTrue(np.array_equal(result, expected_result))

if __name__ == '__main__':
    unittest.main()