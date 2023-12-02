import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from getCentVals import get_cent_vals

class TestGetCentVals(unittest.TestCase):

    def test_get_cent_vals(self):
        # Test input data
        fixed_labels = np.array([[0.981770, 4.344300],
                                [4.416100, 4.884900],
                                [4.932800, 5.588000],
                                [5.637400, 9.148700],
                                [9.148700, 9.724100],
                                [9.724100, 11.732525]
                                ])

        times_ons = fixed_labels[:, 0]
        times_offs = fixed_labels[:, 1]
        

        print("times_ons:", times_ons)
        print("times_offs:", times_offs)
        times = {'ons': times_ons, 'offs': times_offs}
        yinres = {'sr': 44100, 'f0': [440.0, 880.0, 220.0, 660.0]}
        
        

        # Expected output
        expected_cents = [
            [0.0, 1200.0],
            [0.0, 1200.0],
            [0.0, 1200.0]
        ]

        # Call the function to get the actual result
        actual_cents = get_cent_vals(times, yinres)
        print(expected_cents)

        # Assert that the actual result matches the expected result
        self.assertEqual(actual_cents, expected_cents)

if __name__ == '__main__':
    unittest.main()
