import numpy as np
import pandas as pd


# import functions
from symbolic import Score
from alignment import run_alignment, alignment_visualiser, ifgram, freq_and_mag_matrices, find_peaks, find_mids
from f0EstWeightedSum import f0_est_weighted_sum
from f0EstWeightedSumSpec import f0_est_weighted_sum_spec

from performance import estimate_perceptual_parameters, get_cent_vals



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
audio_file = './test_files/example3note.wav'
midi_file = './test_files/monophonic3notes.mid'

# # Poly
# audio_file = './test_files/polyExample.wav'
# midi_file = './test_files/polyExample.mid'

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


res, dtw, spec, estimated_pitch, onset_times, offset_times = run_alignment(
    audio_file, midi_file, means, covars, width, target_sr, n_harm, win_ms)


# Visualize the alignment
alignment_visualiser(midi_file, spec, 1)

# Data from IF gram/Reassigned Spec
i_freqs, times, sig_pwr = ifgram(audiofile=audio_file, tsr=target_sr, win_ms=win_ms)

# Construct frequency and magnitude matrices
freq_mat, mag_mat = freq_and_mag_matrices(audio_file, target_sr)




# loop through voices
for v in range(4):        
    part_name = list(nmat.keys())[v]            
    desired_part = nmat[part_name]    
    ons = np.nonzero(onset_times[v])[0]
    offs = np.nonzero(offset_times[v])[0]
    loc = 0
    n = 0    
    # Estimate f0 for a matrix (or vector) of amplitudes and frequencies

    # START HERE
    f0, pwr, t, M, xf = f0_est_weighted_sum_spec(audio_file, ons[loc], offs[loc], desired_part['MIDI'].values[loc])
    # Estimate note-wise perceptual values
    note_vals = estimate_perceptual_parameters([f0], [pwr], [xf], [M], 4000, 256, 1, 1)
    loc = loc + 1





# Build JSON
nmat = piece.nmats()

instrumentList = list(nmat.keys())

xmlIds = nmat[instrumentList[0]].index
# xmlIds = nmat['Piano'].index

measures = nmat[instrumentList[0]]['MEASURE'].values,
onsets = nmat[instrumentList[0]]['ONSET'].values,
durations = nmat[instrumentList[0]]['DURATION'].values,
parts = nmat[instrumentList[0]]
midis = nmat[instrumentList[0]]['MIDI'].values,
onset_secs = nmat[instrumentList[0]]['ONSET_SEC'].values,
offset_secs = nmat[instrumentList[0]]['OFFSET_SEC'].values


# Add -1 to signify ending
measures = np.append(measures, -1)
onsets = np.append(onsets, -1)
durations = np.append(durations, -1)
midis = np.append(midis, -1) 
onset_secs = np.append(onset_secs, -1)
offset_secs = np.append(offset_secs, -1)





sys.exit()


perceptual_params = estimate_perceptual_parameters(f0_values,F=freq_mat,pwr_vals=sig_pwr,M=mag_mat,SR=target_sr,hop=32,gt_flag=True, X=audio_file)

# Get onset and offset times
times = res # From run_alignment

combined_times = np.column_stack((np.repeat(times['on'], 1), np.repeat(times['off'], 1)))
combined_times = np.append(combined_times, -1)

# Initialize the audio_params
audio_params = {}

# Iterate over the indices of XML_IDs
for i in range(len(xmlIds)):
    start_idx = int(i * len(f0_values) / len(xmlIds))
    end_idx = int((i + 1) * len(f0_values) / len(xmlIds))

    # Extract values for the current time interval
    f0_chunk = f0_values[start_idx:end_idx]    
    pwr_chunk = perceptual_params['pwr_vals'][start_idx:end_idx]
    slope_chunk = perceptual_params['spec_slope'][start_idx:end_idx]
    flux_chunk = perceptual_params['spec_flux'][start_idx:end_idx]
    flat_chunk = perceptual_params['spec_flat'][start_idx:end_idx]    

    # Create a dictionary for the current time interval
    audio_params[xmlIds[i]] = {
        "startTime": combined_times[i], # i is on
        "endTime": combined_times[i+1], # every other i is off                
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

audio_df.to_json("./output_files/cdata_from_alignment_script.json", orient="records", indent=4)


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