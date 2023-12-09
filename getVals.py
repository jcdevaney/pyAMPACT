import numpy as np
import librosa
import librosa.display
import pretty_midi

import sys
from runDTWAlignment import runDTWAlignment
from pytimbre.yin import yin
from pwr import pwr



def get_vals(filename, midi_file, audiofile, sr, hop, width, target_sr, nharm, win_ms):

    # Run DTW alignment
    res, spec, dtw = runDTWAlignment(
        filename, midi_file, 0.025, width, target_sr, nharm, win_ms)

    # Normalize audiofile
    audiofile = audiofile / np.sqrt(np.mean(audiofile ** 2))

    # Read MIDI file
    # nmat = midi2nmat(midi_file)[0]
    # midi_pitch = nmat[:, 4]

    # Use this in substitute for midi2nmat
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    midi_pitch = []

    # Iterate through the instruments in the MIDI file
    for instrument in midi_data.instruments:
        # Skip non-pitched instruments (e.g., drums)
        if not instrument.is_drum:
            # Iterate through the notes in the instrument
            for note in instrument.notes:
                # Append the pitch (MIDI note number) to the list
                midi_pitch.append(note.pitch)

    # Define parameters for YIN analysis
    P = {
        'thresh': 1,  # originally 0.01 in MATLAB, no difference?
        'sr': sr,
        'hop': hop,
        # Broadened range from 2 (added 2)
        'maxf0': np.max(librosa.midi_to_hz(np.array(midi_pitch) + 4)),
        'minf0': np.min(librosa.midi_to_hz(np.array(midi_pitch) - 1)),
    }


    # # Run YIN on audiofile
    # f0, voiced_flag, voiced_probs = librosa.pyin(
    #     audiofile, sr=P['sr'], hop_length=P['hop'], fmax=P['maxf0'], fmin=P['minf0'], n_thresholds=P['thresh'])

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

    f02, t, ap = yin(x=audiofile, Fs=P['sr'], N=win_ms, H=P['hop'], F_max=P['maxf0'], F_min=P['minf0'], threshold=P['thresh'])

    x, t, power = pwr(x=audiofile, Fs=P['sr'], N=win_ms, H=P['hop'], F_max=P['maxf0'], F_min=P['minf0'], threshold=P['thresh'])
        
    # yinres = {
    #     'f0': f0,  # good
    #     'ap': voiced_flag,  # not the same
    #     'pwr': voiced_probs  # not the same
    # }

    yinres = {
        'f0': f02, 
        'ap': ap,  # Good?
        'pwr': power  # Placeholder
    }    
    
    return res, yinres, spec, dtw
