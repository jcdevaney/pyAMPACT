import numpy as np
import librosa
import librosa.display

from getVals import get_vals
# from runHMMAlignment import run_HMM_alignment
# from selectStates import select_states

import os
curr_dir = os.getcwd()
from script import Score

import sys


def run_alignment(filename, midiname, num_notes, state_ord2, note_num, means, covars, learn_params, width, target_sr, nharm, win_ms):
    """
    Description: Calls the DTW alignment function and refines the results with the HMM
    alignment algorithm, with both a basic and modified state spaces (based on the lyrics).
    This function returns the results of both the state spaces as well as the YIN analysis
    of the specified audio file.

    Inputs:
    filename - name of the audio file
    midiname - name of the MIDI file
    num_notes - number of notes in the MIDI file to be aligned
    state_ord2 - vector of state sequence
    note_num - vector of note numbers corresponding to state sequence
    means - mean values for each state
    covars - covariance values for each state
    learn_params - flag as to whether to learn means and covars in the HMM
    width - width parameter (you need to specify this value)
    target_sr - target sample rate (you need to specify this value)
    nharm - number of harmonics (you need to specify this value)
    win_ms - window size in milliseconds (you need to specify this value)

    Outputs:
    allstate - ending times for each state
    selectstate - ending times for each state
    spec - spectrogram of the audio file
    yinres - structure of results of running the YIN algorithm on the audio signal indicated by the input variable filename
    """

    if learn_params is None:
        learn_params = 0

    # Refine state_ord2 to correspond to the number of states specified in num_notes
    note_num = np.array(note_num)
    num_states = max(np.where(note_num <= num_notes)[0])
    # state_ord2 = state_ord2[:num_states] # Original
    # note_num = note_num[:num_states] # Original

    # Read audio file and perform DTW alignment and YIN analysis
    hop = 32
    audiofile, sr = librosa.load(filename, sr=4000)

    # Normalize audio file
    audiofile = audiofile / np.sqrt(np.mean(audiofile ** 2)) * 0.6

    align, yinres, spec, dtw = get_vals(
        filename, midiname, audiofile, sr, hop, width, target_sr, nharm, win_ms)

        
    audiofile = None  # Clear the audiofile variable


    """
    CLEARED. TO BE FIXED LATER

    # Run HMM alignment with the full state sequence
    vpath, starting_state, prior, trans, means_full, covars_full, mixmat, obs, state_ord = run_HMM_alignment(
        num_notes, means, covars, align, yinres, sr, learn_params)

    # Tally of the number of frames in each state
    histvals = np.histogram(vpath, bins=np.arange(1, max(vpath) + 2))[0]

    # Ending time of each state in seconds
    cumsumvals = np.cumsum(histvals * hop / sr)

    # cumsumvals2 = select_states(starting_state, prior, trans,
    #                             means_full, covars_full, mixmat, obs, state_ord2, note_num, sr)
    # vpath2 = select_states(starting_state, prior, trans,
    #                        means_full, covars_full, mixmat, obs, state_ord2, note_num, sr)

    # Create 3*N matrices of the alignments, where the first row is the current states,
    # the second row is the time at which the state ends, and the third row is the state index,
    # and N is the total number of states
    allstate = np.vstack([state_ord, np.zeros_like(state_ord)])
    allstate[1, :len(cumsumvals)] = cumsumvals

    # selectstate = state_ord2
    # selectstate[1, :len(cumsumvals2)] = cumsumvals2  # Original
    # selectstate[:len(cumsumvals2)] = cumsumvals2 # Original
    # selectstate[:len(cumsumvals)] = cumsumvals # Original
    # selectstate[2][:] = note_num

    # # One Note - The middle row needs attention from select_state, but is reliant on data from HMM alignment.
    # The x-axis seems to be correct but the start/stop is not

    # selectstate = np.zeros((len(state_ord2), len(note_num)))
    # selectstate[0, :] = state_ord2  # Good
    # selectstate[1, :] = cumsumvals2  # This needs to be populated somehow
    # selectstate[2, :] = note_num

    """

    # selectstate is from the HMM alignment, which is not in place.  This runs through
    # get_ons_offs in the exampleScript and then plots the lines accordingly...
    
    # selectstate construction, in progress
    # piece = Score(midiname)
    # # end_times = piece.durations()
    # nmat_from_script = piece.nmats() 
    # end_times = nmat_from_script['Piano'].values        
    # end_times = [row[1] for row in end_times]
    # selectstate = np.empty((3, len(note_num)))
    # # end_times = end_times['Piano'].keys()        
    # selectstate[0, :] = state_ord[:len(note_num)]
    # selectstate[1, :] = note_num
    # selectstate[2, :] = np.pad(end_times, (0, len(note_num) - len(end_times)), mode='constant')
    # print(selectstate)
    # print(yinres['time'])
            
    selectstate = np.array([[1.0000, 3.0000, 2.0000, 3.0000, 2.0000, 3.0000],
                            [0.9818, 4.1941, 4.1941, 4.8929, 4.9205, 6.6859],
                            [1.0000, 1.0000, 2.0000, 2.0000, 3.0000, 3.0000]])
    
    return selectstate, spec, yinres
