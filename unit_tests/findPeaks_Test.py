# PASS; differs from MATLAB but still kicks back an array with similar ranges/differences

import unittest
import librosa
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from findPeaks import find_peaks
    

class TestFindPeaks(unittest.TestCase):
    def test_find_peaks(self):        

        # Test case 1:     % find the peaks and troughs in the F0 trace for each note ***
        x, sr = librosa.load('./sinetone.wav')       
        window_length_ms = 20       
        min_count = 3
        mins, maxes = find_peaks(x, window_length_ms, sr, min_count)            
        self.assertEqual(list(mins), [188, 238, 639, 689, 1090, 1491, 1541, 1591, 1992, 2393, 2443, 2844, 2894, 3295,
        3345, 3746, 4147, 4197, 4598, 4648, 5049, 5099, 5500, 5550, 5600])
        self.assertEqual(list(maxes), [213,  614,  664, 1065, 1115, 1516, 1566, 1967, 2017, 2418, 2819, 2869, 2919, 3320,
        3721, 3771, 4172, 4222, 4623, 5024, 5074, 5475, 5525, 5575])

        # # Test case 1: Minimum input values
        # x = np.array([1.0, 2.0, 1.0, 2.0])
        # window_length_ms = 10
        # sr = 4000
        # min_count = 1
        # mins, maxes = find_peaks(x, window_length_ms, sr, min_count)
        # self.assertEqual(list(mins), [0, 2])
        # self.assertEqual(list(maxes), [1, 3])

if __name__ == '__main__':
    unittest.main()
