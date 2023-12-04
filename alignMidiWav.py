import numpy as np
import librosa
import pandas as pd

from orio_simmx import orio_simmx
from simmx import simmx
from dp import dp

import os
curr_dir = os.getcwd()
from script import Score


def align_midi_wav(MF, WF, TH, ST, width, tsr, nhar, wms):
    piece = Score(MF)
    pianoRoll = piece.pianoRoll()      

    # Construct N
    sampled_grid = []
    for row in pianoRoll:
        sampled_grid.append(row)

    # Calculate spectrogram
    y, sr = librosa.load(WF)
    fft_len = int(2**np.round(np.log(wms/1000*tsr)/np.log(2)))
    hop_length = int((TH * tsr * 1000))
    D = librosa.feature.melspectrogram(
        y=y, sr=tsr, n_fft=fft_len, hop_length=hop_length, window='hamming')

    # basenote = 21  # Corresponds to MIDI note A0
    # tuning = 1.0
    # M = notes2mask(N, fft_len, tsr, nhar, basenote, width, tuning)  # You need to implement notes2mask function

    N = np.array(sampled_grid)
    # N = N.astype(np.int16)

    mask = piece.mask(wms, tsr, nhar, width, bpm=60, aFreq=440,
                      base_note=0, tuning_factor=1, obs=20)

    # INT or FLOAT?
    M = np.array(mask)  # provided as CSV
    M = M.astype(np.int16)

    # Calculate the peak-structure-distance similarity matrix
    if ST == 1:
        S = orio_simmx(M, D)
    else:
        S = simmx(M, D)

    # Threshold for matching a "silence" state 0..1
    # silstatethresh = 0.4;
    # S[onsetcols-1, :] = silstatethresh * np.max(S)

    # Ensure no NaNs (only going to happen with simmx)
    S[np.isnan(S)] = 0

    # Do the DP search
    p, q, D = dp(1 - S)  # You need to implement dpfast function

    # Map indices into MIDI file that make it line up with the spectrogram
    m = np.zeros(D.shape[1], dtype=int)
    for i in range(D.shape[1]):
        if np.any(q == i):
            m[i] = p[np.min(np.where(q == i))]
        else:
            m[i] = 1

    return m, p, q, S, D, M, N
