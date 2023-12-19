# PASS, same in MATLAB
import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from helpers.alignmentHelpers import get_ons_offs

class TestGetOnsOffs(unittest.TestCase):

    def test_get_ons_offs(self):
        # Sample input matrix
        onsoffs = np.array([[1, 2, 3, 1, 2, 3],
                            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                            [0, 1, 2, 0, 1, 2]])

        expected_result = {'ons': [1.0, 2.5], 'offs': [1.5, 3.0]}

        result = get_ons_offs(onsoffs)

        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
