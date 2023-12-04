import unittest
import numpy as np
import librosa
from scipy.io import wavfile

import sys
import os
sys.path.append(os.pardir)
from calculateVibrato import calculate_vibrato

class TestCalculateVibrato(unittest.TestCase):

    def test_calculate_vibrato(self):        
        note_vals, sr = librosa.load('../audio_files/exampleOneNote.wav', sr=4000)  # Load your audio file
        vibrato_depth, vibrato_rate = calculate_vibrato(note_vals, sr)
        # print(f'Vibrato Depth: {vibrato_depth}')
        # print(f'Vibrato Rate (Hz): {vibrato_rate}')

        expected_vibrato_depth = 0.00718
        expected_vibrato_rate = 465.14857

        self.assertAlmostEqual(vibrato_depth, expected_vibrato_depth, places=5)
        self.assertAlmostEqual(vibrato_rate, expected_vibrato_rate, places=5)


if __name__ == '__main__':
    unittest.main()
