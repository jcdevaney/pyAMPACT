# NOT FROM GPT, FIX THIS
import librosa
import numpy as np
import pandas as pd

# Load the 'SingingMeansCovars.mat' file into Python if you haven't already

means = sqrtmeans
covars = sqrtcovars

width = 2
target_sr = 2000

# Audio file to be aligned
audiofile = 'exampleOneNote.wav'

# MIDI file to be aligned
midifile = 'exampleOneNote.mid'

# Number of notes to align
numNotes1 = 1

# Vector of order of states (according to lyrics) in stateOrd and corresponding note numbers in noteNum
stateOrd1 = [1, 3, 1]
noteNum1 = [1, 1, 1]


dtw1note = exampleGenerator(midifile, audiofile, numNotes1, stateOrd1, noteNum1, means, covars, width)

# Save dtw1note to an Excel file
filename = 'example1noteWidth2TEST.xlsx'
dtw1note.to_excel(filename, index=False)

# Continue with the rest of your code for other examples (audiofile6, audiofile3, audiofilePoly)
