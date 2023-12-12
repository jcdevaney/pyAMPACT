import numpy as np
import librosa
import librosa.display
import pretty_midi

import sys
from runDTWAlignment import runDTWAlignment
from pytimbre.yin import yin

import os
curr_dir = os.getcwd()
from script import Score

def get_vals(filename, midi_file, audiofile, sr, hop, width, target_sr, nharm, win_ms):

    # Run DTW alignment
    res, spec, dtw = runDTWAlignment(
        filename, midi_file, 0.025, width, target_sr, nharm, win_ms)

    # Normalize audiofile
    audiofile = audiofile / np.sqrt(np.mean(audiofile ** 2))    
    
    piece = Score(midi_file)
    notes = piece.midiPitches()

   
    # Number of notes to align
    pitch_list = [];
    for note in notes['Piano']: # Hardcoded. Fix this?
        if note != -1: # Exclude rests
            pitch_list.append(note)
    # Define parameters for YIN analysis    
        

    P = {
        'thresh': 1,  # originally 0.01 in MATLAB, no difference?
        'sr': sr,
        'hop': hop,
        # Broadened range from 2 (added 2)
        'maxf0': np.max(librosa.midi_to_hz(np.array(pitch_list) + 4)),
        'minf0': np.min(librosa.midi_to_hz(np.array(pitch_list) - 1)),
    }        

    

    # Parameters
    #     ----------
    #     x : ndarray [shape=(L, )], real - valued
    #         Audio signal
    #     Fs : int
    #         Sampling frequency
    #     N : int
    #         Window size
    #     H : int
    #         Hop size
    #     F_min : float
    #         Minimal frequency
    #     F_max : float
    #         Maximal frequency
    #     threshold : float
    #         Threshold for cumulative mean normalized difference function
    #     verbose : bool
    #         Switch to activate/deactivate status bar

    f0, t, ap = yin(x=audiofile, Fs=P['sr'], N=win_ms, H=P['hop'], F_max=P['maxf0'], F_min=P['minf0'], threshold=P['thresh'])


    yinres = {
        'f0': f0, 
        'time': t,  
        'ap': ap  
    }    
    
    return res, yinres, spec, dtw
