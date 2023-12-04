import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from f0EstWeightedSumSpec import f0_est_weighted_sum_spec

class TestF0EstWeightedSumSpec(unittest.TestCase):

    def test_f0_est_weighted_sum_spec_with_if(self):
        # Test the function with useIf=True (using IF features)
        fileName = "../audio_files/exampleOneNote.wav"
        noteStart_s = 0.5
        noteEnd_s = 1.5
        f0i = 440.0
        useIf = True
        
        # Call the function to get the results
        f0, p, t, M, xf = f0_est_weighted_sum_spec(fileName, noteStart_s, noteEnd_s, f0i, useIf)
        
        # Perform your assertions here to validate the results
        self.assertTrue(isinstance(f0, np.ndarray))
        self.assertTrue(isinstance(p, np.ndarray))
        self.assertTrue(isinstance(t, np.ndarray))
        self.assertTrue(isinstance(M, np.ndarray))
        self.assertTrue(isinstance(xf, np.ndarray))

        
    # def test_f0_est_weighted_sum_spec_without_if(self):
    #     # Test the function with useIf=False (not using IF features)
    #     fileName = "../audio_files/exampleOneNote.wav"
    #     noteStart_s = 0.5
    #     noteEnd_s = 1.5
    #     f0i = 440.0
    #     useIf = False
        
    #     # Call the function to get the results
    #     f0, p, t, M, xf = f0_est_weighted_sum_spec(fileName, noteStart_s, noteEnd_s, f0i, useIf)
        
    #     # Perform your assertions here to validate the results
    #     self.assertTrue(isinstance(f0, np.ndarray))
    #     self.assertTrue(isinstance(p, np.ndarray))
    #     self.assertTrue(isinstance(t, np.ndarray))
    #     self.assertTrue(isinstance(M, np.ndarray))
    #     self.assertTrue(isinstance(xf, np.ndarray))
    

if __name__ == '__main__':
    unittest.main()
