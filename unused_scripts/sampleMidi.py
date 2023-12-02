import numpy as np
import mido

def samplemidi(MF, T=0.05, C=None, MX=5.0):
    # Default channel selection (all except drums)
    if C is None:
        C = list(range(1, 10)) + list(range(11, 128))
    
    # Read MIDI file
    mid = mido.MidiFile(MF)
    
    # Initialize an empty piano roll matrix
    nr = 128
    nc = int(mid.length / T) + 1
    D = np.zeros((nr, nc), dtype=int)
    
    # Iterate over MIDI messages and fill the piano roll
    current_time = 0
    for msg in mid:
        current_time += msg.time
        
        if current_time >= T:
            current_time -= T
            if isinstance(msg, mido.Message) and msg.type == 'note_on' and msg.note in C:
                D[msg.note, int(current_time / T)] = 1
    
    # Cap note durations
    if MX > 0:
        for i in range(nr):
            for j in range(nc):
                if D[i, j] == 1:
                    duration = 1
                    while j + duration < nc and D[i, j + duration] == 1:
                        duration += 1
                    if duration > int(MX / T):
                        D[i, j + int(MX / T) : j + duration] = 0
    
    return D
