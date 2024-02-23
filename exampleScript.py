import numpy as np
import pandas as pd
import librosa
import sys

# import functions
from pyAMPACT.symbolic import Score
from pyAMPACT.alignment import run_alignment, alignment_visualiser, ifgram, freq_and_mag_matrices
from pyAMPACT.alignmentUtils import calculate_f0_est
from pyAMPACT.dataCompilation import data_compilation


"""
Params:
- audio_file (path)
- midi_file (path)
width = 3
target_sr = 4000
n_harm = 3
win_ms = 100

Outputs:
cdata_file (path)

"""


# # Specify audio and MIDI file NAMES
# audio_file = './test_files/example3note.wav'
# midi_file = './test_files/monophonic3notes.mid'

# # Poly
audio_file = './test_files/polyExample.wav'
midi_file = './test_files/polyExample.mid'

# audio_file = './test_files/TAVERNaudio/B063_00_04.wav'
# midi_file = './test_files/TAVERNaudio/B063_00_04.krn'

# audio_file = './rihanna-files/rihanna-vocal tracks/Close to You vocals.wav'
# midi_file = './rihanna-files/Close to You.mei'


piece = Score(midi_file)
nmat = piece.nmats()

y, original_sr = librosa.load(audio_file)
  

# Load singing means and covariances
means = pd.read_csv('./test_files/SingingMeans.csv', sep=' ').values
covars = pd.read_csv('./test_files/SingingCovars.csv', sep=' ').values


# Run the alignment
width = 3
target_sr = 4000
n_harm = 3
win_ms = 100
hop_length = 32

# print(nmat)
res, dtw, spec, newNmat = run_alignment(
    y, original_sr, piece, means, covars, width, target_sr, n_harm, win_ms, hop_length)

nmat = newNmat
# print(nmat)

# Visualize the alignment
alignment_visualiser(spec, 1)

# Data from IF gram/Reassigned Spec
freqs, times, mags_db, f0_values, sig_pwr, mag_mat = ifgram(audiofile=audio_file, tsr=target_sr, win_ms=win_ms)
# freqs, times, mags_db = ifgram(audiofile=audio_file, tsr=target_sr, win_ms=win_ms)
f0_values, sig_pwr = calculate_f0_est(audio_file, hop_length, win_ms, target_sr)

# Prune NaN and zero values from f0_values and sig_pwr
f0_values = f0_values[~np.isnan(f0_values)]
sig_pwr = sig_pwr[sig_pwr != 0]

print('End at data_compilation')
sys.exit()
# data_compilation(f0_values, sig_pwr, freq_mat, mag_mat, nmat, target_sr, hop_length, audio_file)
