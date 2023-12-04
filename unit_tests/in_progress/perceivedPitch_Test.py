# Tests here pass for NaN and large gamma, and values, but it may be too brute force.
# The WEIGHTS calculation in perceived_pitch may need revisiting, or the whole function,
# But it seems quite close?
import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from perceivedPitch import perceived_pitch

class TestPerceivedPitch(unittest.TestCase):
    
    def test_perceived_pitch(self):
        # Test case 1: Basic test with no outliers
        f0s = np.array([220.0, 330.0, 440.0, 330.0, 220.0])
        sr = 44100/32 # /32 for hop
        pp1, pp2 = perceived_pitch(f0s, sr)
        self.assertAlmostEqual(pp1, 330.0, places=2)
        self.assertAlmostEqual(pp2, 293.33, places=2)
        
        # Test case 2: Test with outliers and NaN values
        f0s = np.array([220.0, 330.0, 440.0, np.nan, 330.0, 220.0, 8820.0])
        sr = 44100/32 # /32 for hop
        pp1, pp2 = perceived_pitch(f0s, sr)
        self.assertAlmostEqual(pp1, 294.35, places=2) # Correct that in it should be the same,
        self.assertAlmostEqual(pp2, 294.35, places=2) # because of pruning outliers.
        
        # Test case 3: Test with a large gamma value
        f0s = np.array([220.0, 330.0, 440.0, 330.0, 220.0])
        sr = 44100/32 # /32 for hop
        gamma = 1000000  # Large gamma value
        pp1, pp2 = perceived_pitch(f0s, sr, gamma)
        self.assertAlmostEqual(pp1, 330.0, places=2)
        self.assertAlmostEqual(pp2, 293.33, places=2)

if __name__ == '__main__':
    unittest.main()
