# CONDITIONAL PASS (see note below)
import unittest
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from noteDct import note_dct

class TestNoteDCT(unittest.TestCase):

    def test_note_dct(self):
        

        # Test case 1: Basic functionality
        sr = 44100  # Sample rate
        duration = 5.0  # Duration of the audio signal in seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)  # Time array
        x = 0.5 * np.sin(2 * np.pi * 440.0 * t)  # Generate a sine wave as the input signal
        Ndct = 10  # Number of DCT coefficients                        
        coefs, approx = note_dct(x, Ndct, sr)


        # Check if the length of coefs is Ndct/approx is size of x 
        self.assertEqual(len(coefs), Ndct)
        self.assertEqual(len(approx), x.size)

        #Need to check the coefs values themselves, and approximations.
        
            
        # Test case 2: Test with a different input signal
        sr = 22050  # Different sample rate
        x = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        Ndct = 2                
        coefs, approx = note_dct(x, Ndct, sr)
        
        # Check if the length of coefs is Ndct/approx is size of x        
        self.assertEqual(len(coefs), Ndct)
        self.assertEqual(len(approx), x.size)

        

if __name__ == '__main__':
    unittest.main()