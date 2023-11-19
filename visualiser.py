# NEEDS FIXING

import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import mido
from mido import MidiFile
import os

def visualiser(trace, mid, spec):
    
    # Plots a gross DTW alignment overlaid with the fine alignment
    # resulting from the HMM aligner on the output of YIN. Trace[0, :]
    # is the list of states in the HMM (currently ignored, assumed to
    # be 1,2,3,2,1,2,3,2...), and trace[1, :] is the number of YIN
    # frames for which that state is occupied.

    # Args:
    # trace (np.ndarray): 3-D matrix of a list of states (trace[0, :]),
    #     the times they end at (trace[1, :]), and the state indices (trace[2, :])
    # mid (str): Path to the MIDI file
    # spec (np.ndarray): Spectrogram of audio file

    # Dependencies:
    #     Toiviainen, P. and T. Eerola. 2006. MIDI Toolbox. Available from:
    #     https://www.jyu.fi/hum/laitokset/musiikki/en/research/coe/materials
    #     /miditoolbox/

    # Please note that I've left the implementation of the plotFineAlign function 
    # as a placeholder. You should implement this function to create the fine 
    # alignment plot as per your requirements.
    

    
    # Fix for ending zeros that mess up the plot
    if trace[1, -1] == 0:
        trace = trace[:, :-1]
    if trace[1, -2] == 0:
        trace[1, -2] = trace[1, -3]

    # Hop size between frames
    stftHop = 0.025

    # Read MIDI file
    midi = MidiFile(mid)
    notes = []
    for msg in midi.play():
        if msg.type == 'note_on':
            notes.append(msg.note)

    # Plot spectrogram of audio file
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(20 * np.log10(spec), sr=44100, x_axis='time', y_axis='cqt_note', cmap='gray_r')
    plt.title('Spectrogram with Aligned MIDI Notes Overlaid')
    plt.xlabel('Time (s)')
    plt.ylabel('MIDI Note')
    plt.colorbar(format='%+2.0f dB')

    # Plot alignment
    plot_fine_align(trace[0, :], trace[1, :], notes, stftHop)
    if trace.shape[0] >= 3:
        notenums = trace[2, 1:]
    else:
        nlim = len(notes)
        notenums = list(np.arange(1, nlim + 1)) * 4 + [nlim]

    plt.show()

def plot_fine_align(trace_states, trace_times, notes, stft_hop):
    # Your implementation for plotFineAlign function here
    pass

# # Example usage
# if __name__ == "__main__":
#     trace = np.array([[1, 2, 3, 2, 1, 2, 3, 2],
#                      [10, 15, 20, 25, 30, 35, 40, 45],
#                      [0, 1, 2, 1, 0, 1, 2, 1]])
#     mid_file = "your_midi_file.mid"
#     spec = np.random.rand(513, 100)  # Example spectrogram, replace with your data
#     visualiser(trace, mid_file, spec)
