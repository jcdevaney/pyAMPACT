#PASS, not compared against MATLAB
import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from findSteady import find_steady


class TestFindSteady(unittest.TestCase):
    
    def test_findSteady(self):
        x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        mins = np.array([0, 5])
        maxes = np.array([2, 7])
        x_mids = np.array([1.5, 3.5, 6.0, 8.5, 9.0, 9.1])
        y_mids = np.array([15, 35, 55])
        thresh_cents = 5

        steady = find_steady(x, mins, maxes, x_mids, y_mids, thresh_cents)
        
        # Define the expected steady portion based on the example usage
        # expected_steady = np.array([15, 35, 55, 75])
        expected_steady = [1.5]

        # Check if the result matches the expected steady portion
        self.assertTrue(np.array_equal(steady, expected_steady))
        

        # Test case 2: More complex input
        x2 = np.array([0, -10, -20, -30, -40, -30, -20, -10, 0, 10, 20, 30, 40, 30, 20, 10, 0])
        mins2 = np.array([2, 5])
        maxes2 = np.array([4, 8])
        x_mids2 = np.array([1.0, 3.0, 4.5, 6.5, 7.5, 9.5, 11.0])
        y_mids2 = np.array([15, 35, 55])
        thresh_cents2 = 50

        steady2 = find_steady(x2, mins2, maxes2, x_mids2, y_mids2, thresh_cents2)        
        
        # Define the expected steady portion based on the example usage
        expected_steady2 = [4.5]

        self.assertTrue(np.array_equal(steady2, expected_steady2))


if __name__ == '__main__':
    unittest.main()
