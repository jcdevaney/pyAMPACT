import numpy as np
import pandas as pd

import os
curr_dir = os.getcwd()
from script import Score

# import functions
from runAlignment import run_alignment
from alignmentVisualiser import alignment_visualizer
from getOnsOffs import get_ons_offs
from getCentVals import get_cent_vals
from findPeaks import find_peaks
from findMids import find_mids
# from smoothNote import smooth_note
# from findSteady import find_steady
# from noteDct import note_dct
from freqAndMagMatrices import freq_and_mag_matrices
from estimatePerceptualParameters import estimate_perceptual_parameters

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
times_df.to_csv('example.csv', sep='\t', index=False, header=False)

# Load data into a Pandas DataFrame
fixed_labels = pd.read_csv('exampleFixed.txt', delimiter='\t', header=None)
# Assign columns to 'ons' and 'offs'
times = pd.DataFrame({'ons': fixed_labels.iloc[:, 0].values, 'offs': fixed_labels.iloc[:, 1].values})



# Build JSON
durations = piece.durations()
durations = durations['Piano'].values # Hardcode 'Piano' part?
durations = np.append(durations, -1) # Add -1 to signify ending

f0_values = yin_res['f0']
pwr_values = yin_res['ap']


# Construct frequency and magnitude matrices
freq_mat, mag_mat = freq_and_mag_matrices(audio_file, target_sr)
res = estimate_perceptual_parameters(f0_values, pwr_vals=pwr_values,F=freq_mat,M=mag_mat,SR=target_sr,hop=32,gt_flag=True, X=audio_file)

timesOns = tuple(times['ons'].values)

# Create a dictionary
# print("f0Vals", len(f0_values))
# print("pwrVals", len(res['pwr_vals']))
# print("specSlope", len(res['spec_slope']))
# print("spec_flux", len(res['spec_flux']))
# print("spec_flat", len(res['spec_flat']))

# Initialize the params_dict
params_dict = {}




# Iterate over the indices of timesOns
for i in range(len(timesOns)):
    start_idx = int(i * len(f0_values) / len(timesOns))
    end_idx = int((i + 1) * len(f0_values) / len(timesOns))

    # Extract values for the current time interval
    f0_chunk = f0_values[start_idx:end_idx]    
    pwr_chunk = res['pwr_vals'][start_idx:end_idx]
    slope_chunk = res['spec_slope'][start_idx:end_idx]
    flux_chunk = res['spec_flux'][start_idx:end_idx]
    flat_chunk = res['spec_flat'][start_idx:end_idx]

    # Create a dictionary for the current time interval
    params_dict[timesOns[i]] = {
        "dur": durations[i],
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
        "mean_spec_flat": res['mean_spec_flat']   
        # Add other parameters and their corresponding chunks here
    }
    

# print(timesOns)
# params_dict = {
#     timesOns: {
#     # "dur": durations,
#     "f0Vals": f0_values,
#     "ppitch1": res['ppitch'][0],
#     "ppitch2": res['ppitch'][1],
#     "jitter": res['jitter'],
#     "vibratoDepth": res['vibrato_depth'],
#     "vibratoRate": res['vibrato_rate'],
#     "pwrVals": res['pwr_vals'],
#     "avgPwr": sum(res['pwr_vals']) / len(res['pwr_vals']),
#     "shimmer": res['shimmer'],
#     # "specCent": res['spec_centroid']
#     # "specCentMean": 1370.1594532691213,
#     "specSlope": res['spec_slope'],
#     "meanSpecSlope": res['mean_spec_slope'],
#     "spec_flux": res['spec_flux'],
#     "mean_spec_flux": res['mean_spec_flux'],
#     "spec_flat": res['spec_flat'],
#     "mean_spec_flat": res['mean_spec_flat']    
#     }
# }

# Create a DataFrame from the dictionary
df = pd.DataFrame([params_dict])

# Save the DataFrame as JSON
df.to_json("./test_files/cdata_from_audioScript.json", orient="records", indent=4)



# Map timing information to the quantized MIDI file
# nmat_new = get_timing_data(midi_file, onsets, offsets)
# nmat_new = get_timing_data(midi_file, times)
# print(nmat_new)

# Placeholder
# nmat_new = [
#     [0, 3.3625, 1.0000, 70.0000, 9.0000, 0, 3.3625],
#     [3.4343, 0.4688, 1.0000, 69.0000, 9.0000, 3.4343, 0.4688],
#     [3.9510, 0.6552, 1.0000, 70.0000, 9.0000, 3.9510, 0.6552]
# ]



# Write the new MIDI file
# midi = MidiFile()
# midi.tracks.append(nmat2midi(nmat_new, midi_file))
# midi.save('examplePerformance.mid')

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