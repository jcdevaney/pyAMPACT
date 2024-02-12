import numpy as np
import sys

from scipy.io import wavfile
from mido import MidiFile
from performance import estimate_perceptual_parameters
from f0EstWeightedSumSpec import f0_est_weighted_sum_spec
from alignment import run_alignment
from polyFiles_inProgress.runPolyAlignment import runPolyAlignment

# For converting .mat to .csv, remove later
# import scipy.io
# import pandas as pd
# mat = scipy.io.loadmat('./test_files/polySingingMeansCovars.mat')
# mat = {k:v for k, v in mat.items() if k[0] != '_'}
# data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()}) # compatible for both python 2.x and python 3.x
# data.to_csv("./output_files/SingingMeansCovars.csv")
# sys.exit()

# audio file - change to librosa
audiofile = './test_files/polyExample.wav'
sr, sig = wavfile.read(audiofile)

# midifile
midifile = './test_files/polyExample.mid'

# HMM parameters
means_covars_mat = './test_files/polySingingMeansCovars.csv'
means = 0
covars = 0

# voice type for each polyphonic line
voice_type = [2, 1, 1, 1]

# align MIDI to audio
# estimated_ons, estimated_offs, nmat, dtw = runPolyAlignment(audiofile, midifile, means_covars_mat, voice_type)
estimated_ons, estimated_offs, nmat, dtw = run_alignment(audiofile, midifile, means, covars, 3, 4000, 3, 100)

# loop through voices
for v in range(4):
    print(ons)
    sys.exit()
    ons = np.nonzero(estimated_ons[v])[0]
    offs = np.nonzero(estimated_offs[v])[0]
    loc = 1
    n = 1
    # Estimate f0 for a matrix (or vector) of amplitudes and frequencies
    f0, pwr, t, M, xf = f0_est_weighted_sum_spec(audiofile, ons[loc], offs[loc], nmat[v][n, 3])
    # Estimate note-wise perceptual values
    note_vals = estimate_perceptual_parameters([f0], [pwr], [xf], [M], sr, 256, 1, sig)
    loc = loc + 1
