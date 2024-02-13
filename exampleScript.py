import numpy as np
import pandas as pd


# import functions
from symbolic import Score
from alignment import run_alignment, alignment_visualiser, ifgram, freq_and_mag_matrices, find_peaks, find_mids
from alignmentUtils import calculate_f0_est

from performance import estimate_perceptual_parameters, get_cent_vals

import os
import sys

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


# Specify audio and MIDI file NAMES
# audio_file = './test_files/avemaria_full.wav'
# midi_file = './test_files/avemaria_full.mid'
# audio_file = './test_files/example3note.wav'
# midi_file = './test_files/monophonic3notes.mid'

# # Poly
audio_file = './test_files/polyExample.wav'
midi_file = './test_files/polyExample.mid'


piece = Score(midi_file)
notes = piece.midiPitches()
nmat = piece.nmats()
  


# Load singing means and covariances
means = pd.read_csv('./test_files/SingingMeans.csv', sep=' ').values
covars = pd.read_csv('./test_files/SingingCovars.csv', sep=' ').values


# Run the alignment
width = 3
target_sr = 4000
n_harm = 3
win_ms = 100
hop_length = 32


res, dtw, spec = run_alignment(
    audio_file, midi_file, means, covars, width, target_sr, n_harm, win_ms, hop_length)


# Visualize the alignment
alignment_visualiser(midi_file, spec, 1)

# Data from IF gram/Reassigned Spec
i_freqs, times, sig_pwr = ifgram(audiofile=audio_file, tsr=target_sr, win_ms=win_ms)

# Construct frequency and magnitude matrices
freq_mat, mag_mat = freq_and_mag_matrices(audio_file, target_sr)


# loop through voices
midi_columns = []
ons = []
offs = []
for key, df in nmat.items():        
    onset_sec = df['ONSET_SEC']
    offset_sec = df['OFFSET_SEC']
    midi_notes = df['MIDI'] 
    ons.append(onset_sec)
    offs.append(offset_sec)
    midi_columns.append(midi_notes)


# Instead passing in ifgram values in advance and doing the math on the spec
f0_values, pwr = calculate_f0_est(audio_file, hop_length, win_ms, target_sr)

perceptual_params = estimate_perceptual_parameters(f0_values, F=freq_mat, pwr_vals=pwr, M=mag_mat, SR=target_sr, hop=hop_length, gt_flag=True, X=audio_file)


# Old way
# for key, df in nmat.items():               
#     measures = df['MEASURE'].values
#     onsets = df['ONSET'].values
#     durations = df['DURATION'].values
#     parts = key
#     midis = df['MIDI'].values
#     onset_secs = df['ONSET_SEC'].values
#     offset_secs = df['OFFSET_SEC'].values            
    
# # Add -1 to signify ending
# measures = np.append(measures, -1)
# onsets = np.append(onsets, -1)
# durations = np.append(durations, -1)
# midis = np.append(midis, -1) 
# onset_secs = np.append(onset_secs, -1)
# offset_secs = np.append(offset_secs, -1)





# Get onset and offset times
times = res # From run_alignment

combined_times = np.column_stack((np.repeat(times['on'], 1), np.repeat(times['off'], 1)))
combined_times = np.append(combined_times, -1)

# Initialize the audio_params
audio_params = {}

note = 0
prev_key = None
# Iterate over the indices of XML_IDs
for key, df in nmat.items():
    if key != prev_key:        
        # Update the previous key to the current key
        note = 0
        prev_key = key
    measures = df['MEASURE'].values    
    onsets = df['ONSET'].values
    durations = df['DURATION'].values
    parts = key
    midis = df['MIDI'].values
    onset_secs = df['ONSET_SEC'].values
    offset_secs = df['OFFSET_SEC'].values 
    
    # Add -1 to signify ending
    measures = np.append(measures, -1)
    onsets = np.append(onsets, -1)
    durations = np.append(durations, -1)
    midis = np.append(midis, -1) 
    onset_secs = np.append(onset_secs, -1)
    offset_secs = np.append(offset_secs, -1)

    for i in df.index:              
        note += 1          
        start_idx = int(note * len(f0_values) / len(df))
        end_idx = int((note + 1) * len(f0_values) / len(df))

        # Extract values for the current time interval
        f0_chunk = f0_values[start_idx:end_idx]    
        pwr_chunk = perceptual_params['pwr_vals'][start_idx:end_idx]
        slope_chunk = perceptual_params['spec_slope'][start_idx:end_idx]
        flux_chunk = perceptual_params['spec_flux'][start_idx:end_idx]
        flat_chunk = perceptual_params['spec_flat'][start_idx:end_idx]    

        # Create a dictionary for the current time interval
        audio_params[i] = {            
            "startTime": combined_times[note], # i is on
            "endTime": combined_times[note+1], # every other i is off                
            "f0Vals": f0_chunk,
            "ppitch1": perceptual_params['ppitch'][0],
            "ppitch2": perceptual_params['ppitch'][1],
            "jitter": perceptual_params['jitter'],
            "vibratoDepth": perceptual_params['vibrato_depth'],
            "vibratoRate": perceptual_params['vibrato_rate'],
            "pwrVals": pwr_chunk,
            "avgPwr": sum(perceptual_params['pwr_vals']) / len(perceptual_params['pwr_vals']),
            "shimmer": perceptual_params['shimmer'],
            # "specCent": perceptual_params['spec_centroid'] # can get this via librosa
            # "specCentMean": 1370.1594532691213,
            "specSlope": slope_chunk,
            "meanSpecSlope": perceptual_params['mean_spec_slope'],
            "spec_flux": flux_chunk,
            "mean_spec_flux": perceptual_params['mean_spec_flux'],
            "spec_flat": flat_chunk,
            "mean_spec_flat": perceptual_params['mean_spec_flat'],
            "MEASURE": measures[note],
            "ONSET": onsets[note],
            "DURATION": durations[note],
            "PART":"Piano",
            "MIDI": midis[note],
            "ONSET_SEC": onset_secs[note],
            "OFFSET_SEC": offset_secs[note]
            # Add other parameters and their corresponding chunks here
        }


# Create DataFrames from the dictionary
audio_df = pd.DataFrame([audio_params])

audio_file_name = os.path.splitext(os.path.basename(audio_file))[0]

output_file = f"./output_files/alignment_cdata_{audio_file_name}.json"


audio_df.to_json(output_file, orient="records", indent=4)


# Everything beyond here is not yet live...
sys.exit()


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