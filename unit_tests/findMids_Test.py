# PASS

import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from findMids import find_mids

class TestFindMids(unittest.TestCase):
    
    def test_find_mids(self):
        # Define some test data
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mins = np.array([0, 3])
        maxes = np.array([2])
        windowLength_ms = 20  # 1 second
        sr = 44100  # Sample rate

        # Call the function to be tested
        x_mids, y_mids = find_mids(x, mins, maxes, windowLength_ms, sr)

        # Define expected results based on the test data (you may need to adjust these)
        expected_x_mids = [2, 2]  # Adjust based on your actual implementation
        expected_y_mids = [3.0, 3.0]  # Adjust based on your actual implementation

        # Assert that the actual results match the expected results
        self.assertEqual(list(x_mids), expected_x_mids)
        self.assertEqual(list(y_mids), expected_y_mids)

if __name__ == '__main__':
    unittest.main()
