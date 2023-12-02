import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from f0EstWeightedSum import f0_est_weighted_sum

class TestF0EstWeightedSum(unittest.TestCase):
    def test_case1(self):
        x = np.array([1, 2, 3])
        f = np.array([2, 3, 4])
        f0i = 1
        fMax = 4
        fThresh = 0.5

        f0, pow, strips = f0_est_weighted_sum(x, f, f0i, fMax, fThresh)        

        
        # Add your assertions here to check the correctness of the output
        # self.assertAlmostEqual(f0, expected_f0_value, places=decimal_places)
        # self.assertAlmostEqual(pow, expected_pow_value, places=decimal_places)
        # Add more assertions for strips if needed
    

if __name__ == '__main__':
    unittest.main()
