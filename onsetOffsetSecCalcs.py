# The midi2nmat function was taken from the MATLAB code and populates the nmat with much of the same data as
# the current symbolic.py functions.  It also provides the ONSET_SEC and DURATION_SEC in the last two columns,
# which are key to calculating the DTW Alignment.  I had this function below as a placeholder to do this, by 
# cycling through the MIDI data, finding the note on/off events, and adding the cumulative time based on seconds
# per tick of the tempo.  This could possibly work, or something similar, but I don't know how accurate it is
# across the board.
# It seems the best approach to get all of this data into the Score.nmats() to reference and mutate whenever
# necessary.  We can also populate data back into empty columns of the nmat, but to overwrite columns, as 
# done below, seems unnecessary on the outset.

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
    seconds_per_tick = 60 / (50000000 / ticks_per_beat)
    
    current_tempo = 500000  # Default tempo

    for track in mid.tracks:
        cum_time = 0
        start_time = 0

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


# This code below is from MATLAB to update the first two columns of the nmat, after running maptimes to get the
# ONSET and OFFSET times in seconds.  I don't know how useful this is now given the robustness of the symbolic.py
# matrix, but what we do need is ONSET_SEC and OFFSET_SEC to be populated into the nmat, whether or not it runs
# this update, is up for discussion.  I could not really follow why it was there, other than the first two columns
# were referenced later on in DTW.  This could be adjusted downstream.  In either case, here it is:

    # Columns 6 and 7 in the MATLAB version of the nmat, produced by the midi2nmat function, have values that
    # I can't identify source-wise but are reliant on calculating the onset/offset times.
    # I have this hardcoded to ease with debugging for now.  Trying to use:
        
    # # The original translation from MATLAB.  Col numbers should be adjusted for 0 index    
    # # Modifying column 7 of align.nmat
    # align_nmat[:, 6] = align_nmat[:, 5] + align_nmat[:, 6]
    
    # # Modifying columns 1 and 2 of align.nmat using maptimes
    # align_nmat[:, :2] = maptimes(align_nmat[:, 5:7], (dtw.MA - 1) * tres, (dtw.RA - 1) * tres)



# This was an alternate approach I was exploring drawing directly from symbolic.py and doing this calculation.
# I don't feel that this is entirely useful on its own, but including as it could somehow be used with the midi2nmat
# piece to update columns as needed.
    #
    # all_onset_values = []
    # for part_name, part_df in nmat.items():
    #     onset_values = part_df['ONSET'].values
    #     all_onset_values.extend(onset_values)
    # print(all_onset_values)

    # all_duration_values = []
    # for part_name, part_df in nmat.items():
    #     duration_values = part_df['DURATION'].values
    #     all_duration_values.extend(duration_values)
    # print(all_duration_values)

    # selected_columns_values = {}
    # for part_name, part_df in nmat.items():
    #     selected_columns_values[part_name] = part_df.iloc[:,1:3].values        
    # # print(selected_columns_values)

    # all_float_pairs = []
    # for part_name, values_array in selected_columns_values.items():
    #     float_pairs = [tuple(row) for row in values_array]
    #     all_float_pairs.extend(float_pairs)

    # # Now, all_float_pairs is a list of tuples, each containing a pair of float values from the two columns
    # print(all_float_pairs)

    # # THE MAPTIMES FUNCTION NEEDS TO TAKE IN THE FLOAT PAIRS AND RETURN THE ONSET AND OFFSET TIMES!
    # u = maptimes(all_float_pairs)



# Finally this is what is running to get the DTW object that is run in the runDTWAlignment script.  The
# previous explanations above construct the input params to get the audio and midi aligned, specifically
# with onset and offsets in seconds.  Also durations.  There is a bit about rests in the MATLAB code,
# in that this may need adjusting should rests be accounted for, which they are in symbolic.py.  This
# is specifically around Line 111 in runPolyAlignment in MATLAB.  This may work itself out once we get 
# this data in order in symbolic.py though!
def align_midi_wav(MF, WF, TH, ST, width, tsr, nhar, wms):    
    """
    Align a midi file to a wav file using the "peak structure
    distance" of Orio et al. that use the MIDI notes to build 
    a mask that is compared against harmonics in the audio.
        
    :param MF: is the name of the MIDI file, 
    :param WF: is the name of the wav file.
    :param TH: is the time step resolution (default 0.050).
    :param ST: is the similarity type: 0 (default) is triangle inequality;

    :returns:
        - m: Is the map s.t. M(:,m).            
        - [p,q]: Are the path from DP.
        - S: The similarity matrix.
        - D: Is the spectrogram. 
        - M: Is the midi-note-derived mask.
        - N: Is Orio-style "peak structure distance".    
    """
    
    # Is this correct re: alignMidiWav in MATLAB?
    # Should the pianoRoll be used to construct N
    piece = Score(MF)
    pianoRoll = piece.pianoRoll()      

    # Construct N
    sampled_grid = []
    for row in pianoRoll:
        sampled_grid.append(row)
    
    N = np.array(sampled_grid)    

    d, sr = librosa.load(WF, sr=None, mono=False)

        
    # Calculate spectrogram
    # Had other methods to do this, but melspectrogram was most concise.  Does this work?
    y, sr = librosa.load(WF)
    fft_len = int(2**np.round(np.log(wms/1000*tsr)/np.log(2)))
    hop_length = int((TH * tsr * 1000))
    D = librosa.feature.melspectrogram(
        y=y, sr=tsr, n_fft=fft_len, hop_length=hop_length, window='hamming')
    

    
    # First mask declaration here follows the MATLAB params, but not sure
    # these are necessary at this point.
    # mask = piece.mask(wms, tsr, nhar, width, bpm=60, aFreq=440,
    #                   base_note=0, tuning_factor=1, obs=20)     
    mask = piece.mask()

    
    M = np.array(mask)
    # M = M.astype(np.int16)
    
    

    # Calculate the peak-structure-distance similarity matrix
    if ST == 1:
        S = orio_simmx(M, D)
    else:
        S = simmx(M, D) # Throws errors, not currently implemented
    

    # Ensure no NaNs (only going to happen with simmx)
    S[np.isnan(S)] = 0
    
    # Do the DP search
    p, q, D = dp(1 - S)  # Used dp for the sake of simplicity/not writing Cython methods as required by dpfast. Is this okay?
    

    # Map indices into MIDI file that make it line up with the spectrogram
    # Not sure if this is working as all other params are questionable!
    m = np.zeros(D.shape[1], dtype=int)
    for i in range(D.shape[1]):
        if np.any(q == i):
            m[i] = p[np.min(np.where(q == i))]
        else:
            m[i] = 1    
    return m, p, q, S, D, M, N
