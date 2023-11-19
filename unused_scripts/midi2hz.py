import math

def midi2hz(m): 
# Conversion of MIDI note number to frequency (Hz)
# f=midi2hz(m)
# Convert MIDI note numbers to frequencies in hertz (Hz). The A3 
# (Midi number 69) is 440Hz.
#
# Input arguments: M = pitches in MIDI numbers
#
# Output: F = pitches in hertz
#
# Example: midi2hz(pitch(createnmat));
#
#  Author		Date
#  T. Eerola	1.2.2003
#© Part of the MIDI Toolbox, Copyright © 2004, University of Jyvaskyla, Finland
# See License.txt

    if m.length == 0:
        return
    else: 
        f = 440 * math.exp((m-69) * math.log(2)/12);
    return f