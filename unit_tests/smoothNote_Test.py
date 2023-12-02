# PASS, matches MATLAB
import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from smoothNote import smooth_note

class TestSmoothNoteFunction(unittest.TestCase):

    def test_smooth_note(self):
        # Define sample input data for testing
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        x_mid = np.array([2, 5, 8])  # Sample x_mid values
        y_mid = np.array([0.5, 1.0, 0.5])  # Sample y_mid values

        # Call the function with the sample input
        smoothed = smooth_note(x, x_mid, y_mid)
        

        # Define the expected output based on your knowledge
        expected_output = np.array([0.0, 0.0, 0.5, 0.666667, 0.833333, 1.0, 0.833333, 0.666667, 0.5, 0.0])

        # Use assertion methods to check if the function output matches the expected output
        np.testing.assert_allclose(smoothed, expected_output, rtol=1e-5)
        

if __name__ == '__main__':
    unittest.main()
