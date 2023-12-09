import numpy as np
import librosa
# import pretty_midi
import matplotlib.pyplot as plt
import pandas as pd

from mido import MidiFile, MidiTrack, Message
# import functions
from runAlignment import run_alignment
from alignmentVisualiser import alignment_visualizer
from getOnsOffs import get_ons_offs
from getTimingData import get_timing_data
from nmat2midi import nmat2midi
from getCentVals import get_cent_vals
from unused_scripts.getPitchVibratoData import get_pitch_vibrato_data
from findPeaks import find_peaks
from findMids import find_mids
from smoothNote import smooth_note
from findSteady import find_steady
from noteDct import note_dct

import sys

# # ONE NOTES - GOOD!
# # Specify audio and MIDI file NAMES
# audio_file = './audio_files/exampleOneNote.wav'
# midi_file = './audio_files/monophonic1note.mid'

# # Number of notes to align
# num_notes = 1

# # Define state order and note numbers
# state_ord = np.array([1, 3, 1])
# note_num = np.array([1, 1, 1])


# # THREE NOTES - GOOD!
# Specify audio and MIDI file NAMES
audio_file = './audio_files/example3note.wav'
midi_file = './audio_files/monophonic3notes.mid'

# Number of notes to align
num_notes = 3

# Define state order and note numbers
state_ord = np.array([1, 3, 2, 3, 2, 3])
note_num = np.array([1, 1, 2, 2, 3, 3])


# # # SIX NOTES
# # Specify audio and MIDI file NAMES
# audio_file = './audio_files/example.wav'
# midi_file = './audio_files/monophonic6notes.mid'

# # Number of notes to align
# num_notes = 6

# # Define state order and note numbers
# state_ord = np.array([1, 3, 2, 3, 2, 3, 2, 3, 3, 3, 1])
# note_num = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6])


# Load singing means and covariances
means = pd.read_csv('SingingMeans.csv', sep=' ').values
covars = pd.read_csv('SingingCovars.csv', sep=' ').values


# Specify HMM parameters
learn_params = 0

# Run the alignment
width = 3
target_sr = 4000
n_harm = 3
win_ms = 100


all_state, select_state, spec, yin_res = run_alignment(
    audio_file, midi_file, num_notes, state_ord, note_num, means, covars, learn_params, width, target_sr, n_harm, win_ms)


# Load audio and MIDI data
audio, _ = librosa.load(audio_file, sr=target_sr)
midi_data = MidiFile(midi_file)


# Visualize the alignment
alignment_visualizer(select_state, midi_file, spec, 1)

# Get onset and offset times
times = get_ons_offs(select_state)


times_df = pd.DataFrame({'ons': times['ons'], 'offs': times['offs']})
# MATLAB turns to .txt file, Here is makes CSV, change .csv to .txt to get planned text file
times_df.to_csv('example.csv', sep='\t', index=False, header=False)

# Load data into a Pandas DataFrame
# fixed_labels = pd.read_csv('exampleFixed.txt', delimiter='\t', header=None) # 6 notes
fixed_labels = pd.read_csv('exampleFixed.txt', delimiter='\t', header=None)
# Assign columns to 'ons' and 'offs'
times = pd.DataFrame({'ons': fixed_labels.iloc[:, 0].values, 'offs': fixed_labels.iloc[:, 1].values})


# Map timing information to the quantized MIDI file
# nmat_new = get_timing_data(midi_file, onsets, offsets)
# nmat_new = get_timing_data(midi_file, times)
# print(nmat_new)

# Placeholder
nmat_new = [
    [0, 3.3625, 1.0000, 70.0000, 9.0000, 0, 3.3625],
    [3.4343, 0.4688, 1.0000, 69.0000, 9.0000, 3.4343, 0.4688],
    [3.9510, 0.6552, 1.0000, 70.0000, 9.0000, 3.9510, 0.6552]
]



# Write the new MIDI file
# midi = MidiFile()
# midi.tracks.append(nmat2midi(nmat_new, midi_file))
# midi.save('examplePerformance.mid')

# Get cent values for each note
# THIS NEEDS SOME CALCULATION ON THE YINRES RETURN, AS IT IS GIVING HZ AND NOT ??? FROM MATLAB
cents = get_cent_vals(times, yin_res, target_sr)  # Original

# Calculate intervals size, perceived pitch, vibrato rate, and vibrato depth
# vibrato_depth, vibrato_rate, interval_size, perceived_pitch = get_pitch_vibrato_data(cents, target_sr)

# Get DCT values for each note
mins = []
maxes = []
x_mids = []
y_mids = []
smoothed_f0s = []
steady = []
dct_vals = []
approx = []


# CHECK FOR COMPARISON TO MATLAB.  SPECIFICALLY, FOR EXAMPLE:
# Line 147 is actually: min_idx[i], max_idx[i] = find_peaks(cents[i], win_ms, int(target_sr/32), 60)
for i in range(len(cents)):    
    # Find peaks and troughs in the F0 trace for each note    
    
    min_idx, max_idx = find_peaks(cents[i], win_ms, int(target_sr/32), 60)
    mins.append(min_idx)
    maxes.append(max_idx)


    # Find midpoints between mins and maxes
    x_mid, y_mid = find_mids(cents[i], mins[i], maxes[i], 100, int(target_sr/32))
    x_mids.append(x_mid)
    y_mids.append(y_mid)    


    # Generate a smoothed trajectory of a note by connecting the midpoints,
    # Currently BUGGED - You could leave this if you get hung up!
    smoothed_f0 = smooth_note(cents[i], x_mid, y_mid)
    smoothed_f0s.append(smoothed_f0)        

    # Find the steady-state portion of a note
    steady = find_steady(cents[i], mins[i], maxes[i], x_mid, y_mid, 1)
    print(steady)
    # steady.append([steady_start, steady_end])

    # # Compute the DCT of a signal and approximate it with the first 3 coefficients
    # dct_val, approx_val = note_dct(smoothed_f0[steady_start:steady_end], 3, int(target_sr/32))
    # dct_vals.append(dct_val)
    # approx.append(approx_val)

# Get loudness values for each note (you will need to implement this function)
# loudness_estimates, loudness_structure = getLoudnessEstimates(audio_file, onsets, offsets)
