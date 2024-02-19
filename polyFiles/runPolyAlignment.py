# import numpy as np
# import pandas as pd
# import librosa


# import librosa.display
# import matplotlib.pyplot as plt

# import sys # Remove after debugging

# from scipy.signal import resample

# from symbolic import Score
# from alignment import midi2nmat, runDTWAlignment, ifgram

# from genMeansCovars import genMeansCovars
# from genPolyTrans import genPolyTrans  # Assuming you have a function for generating polyphonic transition matrix

# # ONSET function is in the nmat function, can be pulled from there

# def runPolyAlignment(audiofile, midifile, meansCovarsMat='polySingingMeansCovars.csv', voiceType=None):
#     # If voiceType is not provided, set a default value
#     if voiceType is None:
#         voiceType = [2, 1, 1, 1]
    


#     piece = Score(midifile)
#     nmat = piece.nmats()
    

#     # # Initial DTW alignment stuff
#     # # Read MIDI file
#     # nmatAll = midi2nmat(midifile)  # You need to replace this with the actual conversion function

#     # # Adjust note values if minimum velocity is 0
#     # if np.min(nmatAll[:, 2]) == 0:
#     #     nmatAll[:, 2] += 1

#     # nmat = {}
#     # maxNotes = int(np.max(nmatAll[:, 2]))

    

#     # Find the maximum note index
#     maxNotes = {part: df['MIDI'].max() for part, df in nmat.items()}    

#     # In MATLAB, the maxNotes variable is the size of the array of notes, in this case 4.
#     # I don't know if this is a mistake, but this should be confirmed.  To keep consistent,
#     # reinitializing the maxNotes variable here:
#     numOfMaxNotes = 0
#     for index, (part, value) in enumerate(maxNotes.items()):
#         numOfMaxNotes += 1
#         # print(f"The index of '{value}' in the list is: {index}")

#     # Initialize HMM variables
#     startingState = {}  
    
#     for i in range(1, numOfMaxNotes + 1):
#         startingState[i] = np.concatenate(([1], np.zeros(3**i - 1)))    

#     # Get transition matrix for HMM
#     notes, trans = genPolyTrans(50, 0, 5)
    
#     notesInd = {}  
    
#     for i in range(1, numOfMaxNotes + 1):
#         notesInd[i] = np.concatenate([np.array(note).flatten() for note in notes[i]]).reshape(-1, 1)
 

#     # Run DTW alignment using composite midifile
    
#     align, spec, dtw = runDTWAlignment(audiofile, midifile, 0.025, 3, 4000, 3, 100)

    

#     # Load the audio file
    
#     y, sr = librosa.load(audiofile)

#     # Harmonic-Percussive Source Separation
#     harmonic, percussive = librosa.effects.hpss(y)

#     # Use the harmonic component for pitch estimation
#     pitch, magnitudes = librosa.core.piptrack(y=harmonic, sr=sr)

#     # Get the index of the maximum magnitude for each frame
#     max_magnitude_index = np.argmax(magnitudes, axis=0)

#     # Convert the index to frequency    
#     pitch_frequencies = librosa.fft_frequencies(sr=sr, n_fft=len(y))
#     estimated_pitch = pitch_frequencies[max_magnitude_index]

#     # Plot the pitch track
#     plt.figure(figsize=(15, 5))
#     librosa.display.specshow(librosa.amplitude_to_db(magnitudes, ref=np.max),
#                              y_axis='log', x_axis='time', sr=sr)
    
#     plt.title('Estimated Pitch (Hz)')
#     plt.yticks([])
#     # plt.show()

#     sys.exit()

    





























    

#     # Calculate how many voices change at each transition
#     # nmatAll[:, 0] = np.floor(nmatAll[:, 0] * 1000) / 1000
    
#     # unique_beats, idx1, idx2 = np.unique(nmat[1], return_index=True, return_inverse=True) # ORIGINAL
#     unique_beats = pd.concat([nmat[part]['ONSET'] for part in nmat.keys()]).unique()    


#     # uniqueAlignOns = align.nmat[idx1, 0]
#     unique_align_ons = np.unique(align['on'])    
#     onsetMap = np.zeros((len(unique_beats), numOfMaxNotes))
#     onsMap2 = np.zeros_like(onsetMap)

    

#     # Populate onsetMap based on unique_beats and nmat values
#     for i, beat in enumerate(unique_beats):
#         for j in range(numOfMaxNotes):
#             # Get part name by index
#             part_name = list(nmat.keys())[j]            
#             desired_part = nmat[part_name]
#             if np.sum(desired_part.values == beat):
#                 onsetMap[i, j] = 1
    
#     # Populate onsMap2 based on unique_align_ons and onsetMap values
#     for i, beat in enumerate(unique_beats):
#         for j in range(numOfMaxNotes):
#             if onsetMap[i, j] == 1 and i < len(unique_align_ons):
#                 onsMap2[i, j] = unique_align_ons[i]

        

#     # Set parameters for audio analysis
#     offset1 = 0.125
#     offset2 = 0.125
#     y, sr = librosa.load(audiofile)    
#     # y = resample(y, 1, 2) # resample.py has the translation from MATLAB
#     tuning = librosa.estimate_tuning(y=y, sr=sr) # Works but MATLAB returns 2 while this is 0.5    

#     # TBD    
#     parameter_win_len_STMSP = 441
#     parameter_shift_FB = tuning

    

#     # 1/27 START HERE

#     # create a matrix of the notes in the audio in midi note numbers for each
#     # transition, as defined by onsetMap
#     pitches = []

#     for i in range(numOfMaxNotes):
#         idxCell = 1
#         part_name = list(nmat.keys())[i]            
#         desired_part = nmat[part_name]
#         pitches.append(np.array([desired_part['MIDI'] + tuning]))

#     for i in range(1, onsetMap.shape[0]):
#         pitches_i = np.zeros((numOfMaxNotes, 3))  # Initialize a new 2D array for each iteration
#         for j in range(numOfMaxNotes):
#             if onsetMap[i, j] == 1:
#                 part_name = list(nmat.keys())[j]            
#                 desired_part = nmat[part_name]
#                 pitches_i[j, 0] = desired_part['MIDI'].iloc[0] + tuning
#                 pitches_i[j, 1] = 0
#                 try:
#                     pitches_i[j, 2] = desired_part.iloc[idxCell + 1, 4] + tuning
#                 except IndexError:
#                     pass
#                 idxCell += 1
#             else:                
#                 size_pitches_prev = pitches[i-1].shape[1]
#                 print(pitches_i)
#                 pitches_i[j, 0] = pitches[i-1][j, size_pitches_prev-1] + tuning
#                 pitches_i[j, 1] = pitches[i-1][j, size_pitches_prev-1] + tuning
#                 try:
#                     pitches_i[j, 2] = pitches[i-1][j, 2] + tuning
#                 except IndexError:
#                     pass
#         pitches.append(pitches_i)  # Append the new 2D array to the pitches list


#     print(pitches)
#     sys.exit()
#     # Get means and covars for the singing voice
#     # Differentiate for different voices
    
#     meansCovarsMat_path = './test_files/polySingingMeansCovars.csv'
#     vals = pd.read_csv(meansCovarsMat_path)

#     means_seed = []
#     covars_seed = []
#     versions = []

#     for i in range(len(nmat)):
#         means, covars, version = genMeansCovars(notes, vals[i], voiceType)
#         means_seed.append(means)
#         covars_seed.append(covars)
#         versions.append(version)
    
#     print(means_seed)
#     print(covars_seed)
#     print(versions)

#     # Set the harmonics that are going to be considered
#     harmonics = np.array([-1, 0, 1])
#     harmonics2 = np.array([-1, 0, 1, 12, 19, 24, 28, 31, 36])

#     # Run audio analysis
#     fpitch_all = audio_to_pitch_via_FB(audio, parameter)
#     hop = len(audio) / fpitch_all.shape[1]

#     # Start at Line 175 NAME
