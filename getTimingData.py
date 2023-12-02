import pretty_midi
import pandas as pd

import numpy as np

from midi2nmat import midi2nmat
import os
curr_dir = os.getcwd()
from script import Score

import json
import sys

def get_timing_data(midifile, times):    
    # Read quantized MIDI file
    midi_data = Score(midifile)            
    nmat_from_script = midi_data.nmats() 

    nmat_vals = nmat_from_script['Piano'].values
    new_nmat_from_script = nmat_vals[nmat_vals[:, 4] != -1]
    print(new_nmat_from_script)


    nmat_from_func = midi2nmat(midifile) # This instead?
    print(nmat_from_func[1::2])
    

    # START HERE!!!
    # Then build the proper nmat as from MATLAB.  DOES THIS NEED TO BE BUILT???
    nmat_old = np.empty((0,7))    
    
    nmat_old[:, :2] /= 2 # Problem???    
    
    # Replace timing information in MIDI file with performance timings
    nmat_new = nmat_old.copy()
    
    nmat_new[:, 5:7] = [times['ons'].values, (times['offs'] - times['ons']).values]
    offset = nmat_new[0, 5] - nmat_old[0, 0]
    nmat_new[:, 5] -= offset
    nmat_new[:, :2] = nmat_new[:, 5:7]

    return nmat_new





# # FORMER
# import pretty_midi

# def get_timing_data(midifile, times):
#     # Read quantized MIDI file
#     midi_data = pretty_midi.PrettyMIDI(midifile)
    
#     # Convert note onset and offset times to seconds
#     # onset_seconds = [midi_data.time_to_tick(time) / midi_data.tick_to_time(1) for time in times['ons']] # Original
#     onset_seconds = [midi_data.time_to_tick(time) / midi_data.tick_to_time(1) for time in times]
    
#     return onset_seconds
   
