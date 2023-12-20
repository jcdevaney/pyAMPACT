import os
curr_dir = os.getcwd()
from symbolic import Score

import numpy as np
import pandas as pd


# import functions
from align import run_alignment, alignment_visualizer
from alignmentHelpers import get_ons_offs
from pitch import estimate_perceptual_parameters, get_cent_vals, smooth_note, note_dct
from audioUtils import find_peaks, find_mids, freq_and_mag_matrices, find_steady
from pitch import estimate_perceptual_parameters

import sys

# Specify audio and MIDI file NAMES
audio_file = './audio_files/example3note.wav'
midi_file = './test_files/monophonic3notes.mid'
piece = Score(midi_file)
notes = piece.midiPitches()

# Number of notes to align
num_notes = 0;
for note in notes['Piano']: # Hardcoded. Fix this?
    if note != -1: # Exclude rests
        num_notes += 1        

# Define state order and note numbers
state_ord = np.array([1, 3, 2, 3, 2, 3]) # Placeholder, gets selectState in runAlignment
note_num = np.repeat(np.arange(1, num_notes + 1), 2)


# Load singing means and covariances
means = pd.read_csv('./test_files/SingingMeans.csv', sep=' ').values
covars = pd.read_csv('./test_files/SingingCovars.csv', sep=' ').values


# Specify HMM parameters
learn_params = 0

# Run the alignment
width = 3
target_sr = 4000
n_harm = 3
win_ms = 100

# all_state removed
select_state, spec, yin_res = run_alignment(
    audio_file, midi_file, num_notes, state_ord, note_num, means, covars, learn_params, width, target_sr, n_harm, win_ms)


# Visualize the alignment
alignment_visualizer(select_state, midi_file, spec, 1)

# Get onset and offset times
times = get_ons_offs(select_state)


times_df = pd.DataFrame({'ons': times['ons'], 'offs': times['offs']})
# MATLAB turns to .txt file, Here is makes CSV, change .csv to .txt to get planned text file
times_df.to_csv('./audio_output_files/example.csv', sep='\t', index=False, header=False)

# Load data into a Pandas DataFrame
fixed_labels = pd.read_csv('./test_files/exampleFixed.txt', delimiter='\t', header=None)
# Assign columns to 'ons' and 'offs'
times = pd.DataFrame({'ons': fixed_labels.iloc[:, 0].values, 'offs': fixed_labels.iloc[:, 1].values})



# Build JSON
nmat = piece.nmats()

xmlIds = nmat['Piano'].index

measures = nmat['Piano']['MEASURE'].values,
onsets = nmat['Piano']['ONSET'].values,
durations = nmat['Piano']['DURATION'].values,
parts = "Piano" # Hardcoded
midis = nmat['Piano']['MIDI'].values,
onset_secs = nmat['Piano']['ONSET_SEC'].values,
offset_secs = nmat['Piano']['OFFSET_SEC'].values


# Add -1 to signify ending
# Some values have different lengths and will hit an error when calculating starting idx and end_idx.
# Specifically around the chunking of values per note.  This could be remedied by doing chunk calculations
# separately, but for now adding -1 as placeholders for the end of the piece.
measures = np.append(measures, -1)
onsets = np.append(onsets, -1)
durations = np.append(durations, -1)
midis = np.append(midis, -1) 
onset_secs = np.append(onset_secs, -1)
offset_secs = np.append(offset_secs, -1)

f0_values = yin_res['f0']
pwr_values = yin_res['ap']


# Construct frequency and magnitude matrices
freq_mat, mag_mat = freq_and_mag_matrices(audio_file, target_sr)
res = estimate_perceptual_parameters(f0_values, pwr_vals=pwr_values,F=freq_mat,M=mag_mat,SR=target_sr,hop=32,gt_flag=True, X=audio_file)

times_ons = tuple(times['ons'].values)
times_offs = tuple(times['offs'].values)


# Initialize the audio_params
audio_params = {}

# Iterate over the indices of XML_IDs
for i in range(len(xmlIds)):
    start_idx = int(i * len(f0_values) / len(xmlIds))
    end_idx = int((i + 1) * len(f0_values) / len(xmlIds))

    # Extract values for the current time interval
    f0_chunk = f0_values[start_idx:end_idx]    
    pwr_chunk = res['pwr_vals'][start_idx:end_idx]
    slope_chunk = res['spec_slope'][start_idx:end_idx]
    flux_chunk = res['spec_flux'][start_idx:end_idx]
    flat_chunk = res['spec_flat'][start_idx:end_idx]

    # Create a dictionary for the current time interval
    audio_params[xmlIds[i]] = {
        "startTime": times_ons[i],
        "endTime": times_offs[i],                
        "f0Vals": f0_chunk,
        "ppitch1": res['ppitch'][0],
        "ppitch2": res['ppitch'][1],
        "jitter": res['jitter'],
        "vibratoDepth": res['vibrato_depth'],
        "vibratoRate": res['vibrato_rate'],
        "pwrVals": pwr_chunk,
        "avgPwr": sum(res['pwr_vals']) / len(res['pwr_vals']),
        "shimmer": res['shimmer'],
        # "specCent": res['spec_centroid']
        # "specCentMean": 1370.1594532691213,
        "specSlope": slope_chunk,
        "meanSpecSlope": res['mean_spec_slope'],
        "spec_flux": flux_chunk,
        "mean_spec_flux": res['mean_spec_flux'],
        "spec_flat": flat_chunk,
        "mean_spec_flat": res['mean_spec_flat'],
        "MEASURE": measures[i],
        "ONSET": onsets[i],
        "DURATION": durations[i],
        "PART":"Piano",
        "MIDI": midis[i],
        "ONSET_SEC": onset_secs[i],
        "OFFSET_SEC": offset_secs[i]
        # Add other parameters and their corresponding chunks here
    }


# Create DataFrames from the dictionary
audio_df = pd.DataFrame([audio_params])

audio_df.to_json("./test_files/cdata_from_audioScript.json", orient="records", indent=4)




# Map timing information to the quantized MIDI file
# nmat_new = get_timing_data(midi_file, times)

# Write the new MIDI file
# midi = MidiFile()
# midi.tracks.append(nmat2midi(nmat_new, midi_file))
# midi.save('./audio_output_files/examplePerformance.mid')

# Get cent values for each note
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

    """
    BUGGED AFTER THIS POINT

    # Generate a smoothed trajectory of a note by connecting the midpoints,    
    smoothed_f0 = smooth_note(cents[i], x_mid, y_mid)
    smoothed_f0s.append(smoothed_f0)        

    # Find the steady-state portion of a note
    steady = find_steady(cents[i], mins[i], maxes[i], x_mid, y_mid, 1)    
    # steady.append([steady_start, steady_end])

    # # Compute the DCT of a signal and approximate it with the first 3 coefficients
    # dct_val, approx_val = note_dct(smoothed_f0[steady_start:steady_end], 3, int(target_sr/32))
    # dct_vals.append(dct_val)
    # approx.append(approx_val)

# Get loudness values for each note (you will need to implement this function)
# loudness_estimates, loudness_structure = getLoudnessEstimates(audio_file, onsets, offsets)
"""