import numpy as np
import pandas as pd
import librosa

# This only necessary to use global package
# import pyampact

# Comment out when building package
from pyampact.alignment import *
from pyampact.alignmentUtils import *
from pyampact.dataCompilation import *
from pyampact.performance import *
from pyampact.symbolic import Score
from pyampact.symbolicUtils import *

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
# audio_file = './test_files/polyExample.wav'
# midi_file = './test_files/polyExample.mid'

audio_file = './test_files/TAVERNaudio/B063_00_02.wav'
midi_file = './test_files/TAVERNaudio/B063_00_02.krn'

# audio_file = './rihanna-files/rihanna-vocal tracks/Close to You vocals.wav'
# midi_file = './rihanna-files/Close to You.mei'

export_path = "./output_files"

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


res, dtw, spec, nmat = run_alignment(
    y, original_sr, piece, means, covars, nmat, width, target_sr, n_harm, win_ms, hop_length)



# Visualize the alignment
times = []
freqs = []
alignment_visualiser(spec, times, freqs, 1)

# Data from IF gram/Reassigned Spec
freqs, times, mags, f0_values, mags_mat = ifgram(audiofile=audio_file, tsr=target_sr, win_ms=win_ms)
mags_db = librosa.amplitude_to_db(mags, ref=np.max)

f0_values, sig_pwr = calculate_f0_est(audio_file, hop_length, win_ms, target_sr)
sig_pwr = mags ** 2 # power of signal, magnitude/amplitude squared

# Prune NaN and zero values from f0_values and sig_pwr
f0_values = f0_values[~np.isnan(f0_values)]
sig_pwr = sig_pwr[sig_pwr != 0]


data_compilation(f0_values, sig_pwr, mags_mat, nmat, target_sr, hop_length, y, audio_file, export_path)
