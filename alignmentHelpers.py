from symbolic import Score

import numpy as np
import mido

from sklearn.covariance import ShrunkCovariance
from hmmlearn import hmm



def get_ons_offs(onsoffs):
    """
    Extracts a list of onset and offset from an inputted 
             3*N matrix of states and corresponding ending times 
             from AMPACT's HMM-based alignment algorithm
    
    :param onsoffs: A 3*N alignment matrix, the first row is a list of N states
        the second row is the time which the state ends, and the
        third row is the state index
    :returns: 
        - res.ons: List of onset times
        - res.offs: List of offset times
    """
    
    # Find indices where the first row is equal to 3
    stopping = np.where(onsoffs[0] == 3)[0]    
    # Calculate starting indices by subtracting 1 from stopping indices
    starting = stopping - 1

    res = {'ons': [], 'offs': []}     
    for i in range(len(starting)):
        res['ons'].append(onsoffs[1, starting[i]])
        res['offs'].append(onsoffs[1, stopping[i]])
    
    return res



def midi2nmat(filename):
    """
    Read midi file FILENAME into Matlab variable NMAT (Beta)
    Based on Ken Schutte's m-files (readmidi, midiInfo, getTempoChanges)
    This beta might replace the mex-files used in the previous version of the toolbox as 
    newer versions of Matlab (7.4+) and various OS's need new compilations 
    of the mex files. Using the C sources and the compiled mex files provides
    faster reading of midi files but because the compatibility is limited, this 
    simple workaround is offered. This beta version is very primitive,
    though. - Tuomas Eerola

    KNOWN PROBLEMS: - Tempo changes are handled in a simple way
                    - Extra messages are not retained  
                    - Channels may not be handled correctly    

    For more information on Ken Schutte's functions, see 
    http://www.kenschutte.com/software

    CREATED ON 31.12.2007 BY TE (MATLAB 7.4 MacOSX 10.4)
    """
    mid = mido.MidiFile(filename)
    nmat = []

    # Convert ticks per beat to seconds per tick
    ticks_per_beat = mid.ticks_per_beat    
    seconds_per_tick = 60 / (5000000 / ticks_per_beat)

    current_tempo = 500000  # Default tempo

    for track in mid.tracks:
        cum_time = 0
        starttime = 0

        for msg in track:
            cum_time += msg.time

            if msg.type == 'set_tempo':
                tempo = msg.tempo
                current_tempo = tempo

            if msg.type == 'note_on' and msg.velocity > 0:
                note = msg.note
                velocity = msg.velocity
                start_time = cum_time * seconds_per_tick
                nmat.append([note, velocity, start_time, 0, 0, 0])

            if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                for event in reversed(nmat):
                    if event[2] is not None and event[2] <= cum_time * seconds_per_tick:
                        end_time = cum_time * seconds_per_tick
                        duration = end_time - event[2]
                        event[3] = end_time
                        event[4] = duration
                        event[5] = 1  # Mark the note as processed
                        break

            if msg.type == 'end_of_track':
                if len(nmat) > 0:
                    last_event = nmat[-1]
                    last_event[5] = 1  # Mark the note as processed

    # Filter out unprocessed notes
    nmat = [event for event in nmat if event[5] == 1]

    return np.array(nmat)