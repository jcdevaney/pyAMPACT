import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import mido

import sys
import os
sys.path.append(os.pardir)

from pytimbre.yin import yin
from symbolic import Score
from alignmentUtils import orio_simmx, simmx, dp


    
def run_alignment(filename, midiname, num_notes, state_ord2, note_num, means, covars, learn_params, width, target_sr, nharm, win_ms):
    """    
    Calls the DTW alignment function and refines the results with the HMM
    alignment algorithm, with both a basic and modified state spaces (based on the lyrics).
    This function returns the results of both the state spaces as well as the YIN analysis
    of the specified audio file.

    :param filename: Name of the audio file.
    :param midiname: Name of the MIDI file.
    :param num_notes: Number of notes in the MIDI file to be aligned.
    :param state_ord2: Vector of state sequence.
    :param note_num: Vector of note numbers corresponding to state sequence.
    :param means: Mean values for each state.
    :param covars: Covariance values for each state.
    :param learn_params: Flag as to whether to learn means and covars in the HMM.
    :param width: Width parameter (you need to specify this value).
    :param target_sr: Target sample rate (you need to specify this value).
    :param win_ms: Window size in milliseconds (you need to specify this value).
    :param nharm: Number of harmonics (you need to specify this value).

    :returns: 
        - allstate: Ending times for each state.
        - selectstate: Ending times for each state.
        - spec: Spectrogram of the audio file.
        - yinres: Structure of results of running the YIN algorithm on the audio signal indicated by the input variable filename.
    """

    if learn_params is None:
        learn_params = 0

    # Refine state_ord2 to correspond to the number of states specified in num_notes
    note_num = np.array(note_num)    
    
    num_states = max(np.where(note_num <= num_notes)[0])    
    

    # state_ord2 = state_ord2[:num_states] # Original    
    note_num = note_num[:num_states] # Original
    

    # Read audio file and perform DTW alignment and YIN analysis
    hop = 32
    audiofile, sr = librosa.load(filename, sr=4000)

    # Normalize audio file
    audiofile = audiofile / np.sqrt(np.mean(audiofile ** 2)) * 0.6

    align, yinres, spec, dtw = get_vals(
        filename, midiname, audiofile, sr, hop, width, target_sr, nharm, win_ms)
        
    

    audiofile = None  # Clear the audiofile variable
    
    # print(align['on'])
    # print(align['off'])
    
    # # selectstate construction, in progress   
    lengthOfNotes = note_num + 1
    
    selectstate = np.empty((3, len(lengthOfNotes)))            
    interleaved = [val for pair in zip(align['on'], align['off']) for val in pair]
    # print(interleaved)
    # interleaved = [val / 2 for val in interleaved]                
    
    selectstate[0, :] = state_ord2[:len(lengthOfNotes)]
    selectstate[1, :] = interleaved[:-1][:selectstate[1, :].shape[0]]
    selectstate[2, :] = note_num
    # print(selectstate)

    # Output from above
    # selectstate = np.array([[1, 3, 2, 3, 2],
    #                         [0, 4.5936, 4.8384, 5.49216, 5.5296],
    #                         [1, 1, 2, 2, 3]])

    # selectstate = np.array([[1.0000, 3.0000, 2.0000, 3.0000, 2.0000, 3.0000],
    #                         [0.9818, 4.1941, 4.1941, 4.8929, 4.9205, 6.6859],
    #                         [1.0000, 1.0000, 2.0000, 2.0000, 3.0000, 3.0000]])
    
    return selectstate, spec, yinres, align



def get_vals(filename, midi_file, audiofile, sr, hop, width, target_sr, nharm, win_ms):
    """    
    Gets values for DTW alignment and YIN analysis of specified audio 
    signal and MIDI file
        
    :param filename: Name of the audio file.
    :param midiname: Name of the MIDI file.
    :param audiofile: Name of teh audio file.
    :param sr: Sample rate.
    :param hop: Hop size.        
        
    :returns:
        - res.on list of DTW predicted onset times in seconds
        - res.off list of DTW predicted offset times in seconds
        - yinres.ap aperiodicty estimates for each frame
        - yinres.pwr power estimates for each frame        
    """

    # Run DTW alignment
    res, spec, dtw = runDTWAlignment(
        filename, midi_file, 0.025, width, target_sr, nharm, win_ms)
    
    # Normalize audiofile
    audiofile = audiofile / np.sqrt(np.mean(audiofile ** 2))    
    
    piece = Score(midi_file)
    notes = piece.midiPitches()

   # Get a list of column names
    column_names = notes.columns.tolist()

    # Use the first column name as the reference_column, build a for loop to pull all references for polyphonic
    # Can be done later...
    reference_column = column_names[0]

    # Number of notes to align
    pitch_list = [];    
    for note in notes[reference_column].values: # Hardcoded. Fix this?
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

    f0, t, ap = yin(x=audiofile, Fs=P['sr'], N=win_ms, H=P['hop'], F_max=P['maxf0'], F_min=P['minf0'], threshold=P['thresh'])

    yinres = {
        'f0': f0, 
        'time': t,  
        'ap': ap  
    }    

        
    
    return res, yinres, spec, dtw



def runDTWAlignment(audiofile, midorig, tres, width, targetsr, nharm, winms):    
    """
    Perform a dynamic time warping alignment between specified audio and MIDI files.

    Returns a matrix with the aligned onset and offset times (with corresponding MIDI
    note numbers) and a spectrogram of the audio.

    :param sig: Audio file.
    :param sr: Sample rate
    :param midorig: MIDI file.
    :param tres: Time resolution for MIDI to spectrum information conversion.
    :param plot: Boolean, whether to plot the spectrogram.

    :returns:
        - align: dynamic time warping MIDI-audio alignment structure
            - align.on: onset times
            - align.off: offset times
            - align.midiNote: MIDI note numbers
        - spec: spectrogram    
    """
    
    # midorig is the path string, not the file
    midi_notes = []

    # Now done in alignMidiWav
    y, sr = librosa.load(audiofile)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)

    m, p, q, S, D, M, N = align_midi_wav(
        MF=midorig, WF=audiofile, TH=tres, ST=0, width=width, tsr=targetsr, nhar=nharm, wms=winms)

    dtw = {
        'M': m,
        'MA': p,
        'RA': q,
        'S': S,
        'D': D,
        'notemask': M,
        'pianoroll': N
    }
    

    # THIS IS INCOMPLETE
    # The alignment needs to happen against the nmat values...    
    nmat = midi2nmat(midorig)
    
    
    # # Load the audio file    
    # y, sr = librosa.load(audiofile)

    # # Calculate the onset times
    # onset_frames = librosa.onset.onset_detect(y, sr=sr, units='time')

    # # Convert onset frames to time
    # onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    

    # # Print the onset times
    # print("Onset Times:", onset_times)
    
    

    # Assuming you want data for the first instrument
    align = {
        'nmat': nmat,
        'on': nmat[:,2],
        'off': nmat[:,3],
        'midiNote': midi_notes
    }    
    

    return align, spec, dtw




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

    piece = Score(MF)
    pianoRoll = piece.pianoRoll()      

    # Construct N
    sampled_grid = []
    for row in pianoRoll:
        sampled_grid.append(row)

    # Calculate spectrogram
    y, sr = librosa.load(WF)
    fft_len = int(2**np.round(np.log(wms/1000*tsr)/np.log(2)))
    hop_length = int((TH * tsr * 1000))
    D = librosa.feature.melspectrogram(
        y=y, sr=tsr, n_fft=fft_len, hop_length=hop_length, window='hamming')

    # basenote = 21  # Corresponds to MIDI note A0
    # tuning = 1.0

    # M = notes2mask(N, fft_len, tsr, nhar, basenote, width, tuning)  # You need to implement notes2mask function

    N = np.array(sampled_grid)
    # N = N.astype(np.int16)

    mask = piece.mask(wms, tsr, nhar, width, bpm=60, aFreq=440,
                      base_note=0, tuning_factor=1, obs=20)    

    # INT or FLOAT?
    M = np.array(mask)  # provided as CSV
    M = M.astype(np.int16)

    # Calculate the peak-structure-distance similarity matrix
    if ST == 1:
        S = orio_simmx(M, D)
    else:
        S = simmx(M, D)

    # Threshold for matching a "silence" state 0..1
    # silstatethresh = 0.4;
    # S[onsetcols-1, :] = silstatethresh * np.max(S)

    # Ensure no NaNs (only going to happen with simmx)
    S[np.isnan(S)] = 0

    # Do the DP search
    p, q, D = dp(1 - S)  # You need to implement dpfast function    

    # Map indices into MIDI file that make it line up with the spectrogram
    m = np.zeros(D.shape[1], dtype=int)
    for i in range(D.shape[1]):
        if np.any(q == i):
            m[i] = p[np.min(np.where(q == i))]
        else:
            m[i] = 1

    return m, p, q, S, D, M, N


    
def alignment_visualiser(trace, mid, spec, fig=1):    
    """    
    Plots a gross DTW alignment overlaid with the fine alignment
    resulting from the HMM aligner on the output of YIN.  Trace(1,:)
    is the list of states in the HMM, and trace(2,:) is the number of YIN
    frames for which that state is occupied.  Highlight is a list of 
    notes for which the steady state will be highlighted.
    
    
    :param trace: 3-D matrix of a list of states (trace(1,:)), the times
        they end at (trace(2,:)), and the state indices (trace(3,:))
    :param mid: Midi file.
    :param spec: Spectrogram of audio file (from align_midi_wav). 
    
    :return: Visualized spectrogram            

    """

    if fig is None:
        fig = 1

    # hop size between frames
    stft_hop = 0.023  # Adjusted from 0.025

    # Read MIDI file
    # This used to take in whole nmat, but now just pitches.
    # note, vel, start time, end time, duration, note is processed    
    piece = Score(mid)
    notes = piece.midiPitches()
    
    # Get a list of column names
    column_names = notes.columns.tolist()

    # Use the first column name as the reference_column, build a for loop to pull all references for polyphonic
    # Can be done later...
    reference_column = column_names[0]
    
    
    # ADJUST CONTRAST AND CHECK HARMONICS...
    # Plot spectrogram of the audio file
    fig = plt.figure(fig)
    plt.imshow(20 * np.log10(spec), aspect='auto', origin='lower', cmap='gray')
    plt.title('Spectrogram with Aligned MIDI Notes Overlaid')
    plt.xlabel('Time (.05s)')
    plt.ylabel('Midinote')
    plt.clim([plt.gci().get_clim()[0], plt.gci().get_clim()[1] - 50])
    plt.colorbar()
    
    
    # plt.show() # Uncomment to show

    # Zoom in on fundamental frequencies
    # notes = nmat[:, 0]  # Note
    
    notes = notes[notes[reference_column].values != -1]
    # notes = notes[reference_column].values
    notes = (2 ** ((notes - 105) / 12)) * 440
    # notes = np.append(notes, notes[-1])
    nlim = len(notes)
    notes = notes[reference_column].values

    # # Normalize the second row based on the first value being adjusted to 0
    # normalized_row = trace[1, :] - trace[1, 0]

    # # Update the second row in the trace array with the normalized values
    # trace[1, :] = normalized_row

    # Hardcoded as taken from MATLAB
    trace = np.zeros((3, 6)) # Size adjustment for sake of example...
    notes = [58.27047019, 55.0, 58.27047019, 58.27047019]    
    trace[0,:] = [1, 3, 2, 3, 2, 3]
    trace[1,:] = [0.9818, 4.1941, 4.1941, 4.8929, 4.9205, 6.6859]
    # print(trace[0,:])
    # print(trace[1,:])
    # print(notes)
    

 
    plot_fine_align(trace[0, :], trace[1, :],
                    notes[:nlim], stft_hop)  # Original

    
    
def plot_fine_align(stateType, occupancy, notes, stftHop):        
    """    
    Plot the HMM alignment based on the output of YIN.

    
    :param stateType: List of states in the HMM.
    :param occupancy: List indicating the time (in seconds) at which the states in stateType end.
    :param notes: List of MIDI note numbers that are played.
    :param stftHop: The hop size between frames in the spectrogram.

    :return: Spectrogram plot of HMM alignment
    """

    # Define styles for different states
    styles = [
        # {'color': 'red', 'marker': '+', 'linestyle': '-', 'linewidth': 2},
        {'color': 'none', 'marker': '+', 'linestyle': '-',
            'linewidth': 2},
        {'color': 'green', 'marker': '+', 'linestyle': '-', 'linewidth': 2},
        {'color': 'blue', 'marker': '+', 'linestyle': '-', 'linewidth': 2}]

    # Calculate segment boundaries
    cs = np.array(occupancy) / stftHop
    segments = np.vstack((cs[:-1], cs[1:])).T

    # Create the plot    
    stateNote = (np.maximum(1, np.cumsum(stateType == 3) + 1)) - 1        
    
    for i in range(segments.shape[0]):             
        # style = styles[int(stateNote[i + 1]) - 1] # This could work?   
        style = styles[int(stateType[i + 1]) - 1]
        x = segments[i, :]
        y = np.tile(notes[stateNote[i]], (2, 1))

        plt.plot(x, y, color=style['color'], marker=style['marker'],
                 linestyle=style['linestyle'], linewidth=style['linewidth'])

    plt.show()


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


def find_mids(x, mins, maxes, windowLength_ms, sr):
    """
    Find the midpoints between mins and maxes in a signal x.
    mins and maxes could come from findPeaks.  Finds the y 
    values of peaks and then finds the x values of the signal 
    that are closest to the average between the min and max 
    peak.
    
    :param x: Inputted signal in cents.
    :param mins: Indices of minima of x.
    :param maxes: Indices of maxima of x.
    :param windowLength_ms: Window length in miliseconds.
    :param sr: Sampling rate of x (frame rate of frequency analysis).

    :returns:
        - x_mids - midpoint locations in x axis between peaks and troughs  
        - y_mids - midpoint locations in y axis between peaks and troughs  
    """
    # Convert window length from milliseconds to frames
    windowLength = int(round(windowLength_ms * sr / 1000.0) * 2)

    # Combine minima and maxima indices and sort them
    pks = np.sort(np.concatenate((maxes, mins)))

    # Initialize an array to store neighborhood means
    neighborhoods = np.zeros(len(pks))

    # Compute the average frequency estimate around each peak
    for i in range(len(pks)):
        start_idx = max(pks[i] - windowLength // 2, 0)
        end_idx = min(pks[i] + windowLength // 2, len(x))
        idx = range(start_idx, end_idx)
        neighborhoods[i] = np.mean(x[idx])

    # Find the mid-points in frequency between peaks
    y_mids = (neighborhoods[:-1] + neighborhoods[1:]) / 2

    # Find the index of the point in the signal between each peak
    # with its value closest to the mid-point in frequency
    x_mids = np.zeros(len(y_mids), dtype=int)
    for i in range(len(y_mids)):
        idx = range(pks[i], pks[i + 1] + 1)
        offset = np.argmin(np.abs(y_mids[i] - x[idx]))
        x_mids[i] = pks[i] + offset

    return x_mids, y_mids




def find_peaks(x, window_length_ms, sr, min_count):
    """
    Find peaks and troughs in a waveform
    Finds the max and min in a window of a given size and keeps
    track of how many windows each point is the min or max of.
    Points that are the min or max of more than minCount windows
    are returned.

    
    :param x: Inputted signal.
    :param windowLength_ms: Window length in ms.
    :param sr: Sampling rate.
    :param minCount: Minimum number of windows that a point needs to be the max
        of to be considered a minimum or a maximum.

    :returns:
        - mins: Minimum values in the signal.
        - maxes: Maximum values in the signal.
    """
    # ADDED THIS
    min_count = min_count / 12
    
    # Would this work???
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html


    # Create an array    
    x = np.array(x)    
    
    # Create arrays of zeros for mins and maxes
    mins = np.zeros_like(x)
    maxes = np.zeros_like(x)    

    # Calculate window length in samples
    window_length = int(window_length_ms * sr / 1000)    
    # Calculate the minimum and maximum values
    for i in range(len(x) - window_length):
        w = x[i:i+window_length]
        di_min = np.argmin(w)
        di_max = np.argmax(w)  
        mins[i + di_min] += 1
        maxes[i + di_max] += 1

    # Prune mins and maxes to only those that occur in minCount or more windows
    # START HERE!        
    mins = np.where(mins >= min_count)[0]    
    maxes = np.where(maxes >= min_count)[0]

    return mins, maxes


def find_steady(x, mins, maxes, x_mids, y_mids, thresh_cents):
    """
    Find the steady-state portion of a note.
    Finds the section of the note with steady vibrato where the 
    peaks and troughs are at least thresh_cents cents away from 
    the mid points between them.  mins and maxes come from 
    findPeaks, x_mids and y_mids come from findMids.  Steady is 
    a range of two indices into f0. mins and maxes may come from
    the findPeaks function and x_mids and y_mids may come from
    the findMids function.

    
    :params x: Vector of f0 estimates in cents.
    :params mins: Indices of minima of x.
    :params maxes: Indices of maxima of x.
    :params x_mids: Midpoint locations in x axis between peaks and troughs.
    :params y_mids: Midpoint locations in y axis between peaks and troughs. 
    :params thresh_cents: Minimum distance in cents from midpoint for peaks and
        troughs.

    :returns:
        - steady: Steady-state portion of inputted signal x.
    """
    # Find extrema that are far enough away from the midpoints    
    peaks = np.sort(np.concatenate((mins, maxes)))       
    excursion = y_mids - x[peaks[:-1]]        
    bigEnough = np.abs(excursion) >= thresh_cents    

    # Count how many extrema are big enough in a row
    inARow = np.zeros(len(bigEnough))
    inARow[0] = int(bigEnough[0])
    for i in range(1, len(bigEnough)):
        if bigEnough[i]:
            inARow[i] = inARow[i - 1] + 1
        else:
            inARow[i] = 0

    # Extract the portion of the note corresponding to the longest run of big enough extrema
    pos = np.argmax(inARow)
    times = inARow[pos]
    steadyPeaks = peaks[int(pos - times + 1):int(pos + 1)]    
    steadyPeaks = np.round(steadyPeaks).astype(int)
    
    # Find indices in x_mids that correspond to the steady portion
    start_idx = np.argmax(x_mids > steadyPeaks[0])
    end_idx = len(x_mids) - np.argmax(x_mids[::-1] < steadyPeaks[1]) - 1

    steady = x_mids[start_idx : end_idx + 1]    
    return steady


def freq_and_mag_matrices(audiofile, sr):
    """    
    Calculate the frequency and magnitude matrices

    :param audiofile: Path to audio file.
    :param sr: Target sample rate.
    """
        
    y, sr = librosa.load(audiofile)

    # Calculate the Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)

    # Get the frequencies corresponding to each row of the STFT matrix
    freq_mat = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0])

    # Extract the magnitude and phase information
    mag_mat = np.abs(D)    

    return freq_mat, mag_mat




# Currently not in use
# def run_HMM_alignment(notenum, means, covars, align, yinres, sr, learnparams=False):    
#     """
#       Refines DTW alignment values with a three-state HMM, identifying 
#       silence,transient, and steady state parts of the signal. The HMM  
#       uses the DTW alignment as a prior. 
    
#     Inputs:
#         notenum - number of notes to be aligned
#         means - 3x2 matrix of mean aperiodicy and power values HMM states
#               column - silence, trans, steady state
#               rows - aperiodicity, power
#         covars - 3x2 matrix of covariances for the aperiodicy and power
#             values (as per means)
#         align - 
#         res - structure containing inital DTW aligment
#         yinres - structure containg yin analysis of the signal
#         sr - sampling rate of the signal
    
#     Outputs: 
#         vpath - verterbi path
#         startingState - starting state for the HMM
#         prior - prior matrix from DTW alignment
#         trans - transition matrix
#         meansFull - means matrix
#         covarsFull - covariance matrix
#         mixmat - matrix of priors for GMM for each state
#         obs - two row matrix observations (aperiodicty and power)
#         stateOrd - modified state order sequence
#     """

#     if not learnparams:
#         shift = 0

#     # Create vectors of onsets and offsets times from DTW alignment
#     align['on'] = np.array(align['on'])
#     align['off'] = np.array(align['off'])
#     ons = np.floor(align['on'] * sr / 32).astype(int)
#     offs = np.floor(align['off'] * sr / 32).astype(int)

#     # Create observation matrix
#     # obs = np.zeros((3, offs[notenum] + 50))
#     obs = np.zeros((3, yinres['ap'].size))

#     # - 1 to account for 0 index of Python
#     obs[0, :] = np.sqrt(yinres['ap'][:offs[notenum - 1] + 50])
#     # obs[1, :] = np.sqrt(yinres['pwr'][:offs[notenum - 1] + 50]) # Changed
#     obs[1, :] = np.sqrt(yinres['time'][:offs[notenum - 1] + 50])
#     # obs[2, :] = 69 + 12 * yinres['f0'][:offs[notenum - 1] + 50]  # Convert octave to MIDI note

#     yinres['f0'] = np.ceil(yinres['f0'])
#     midiPitches = librosa.hz_to_midi(yinres['f0'])
#     # Convert octave to MIDI note
#     obs[2, :] = midiPitches[:offs[notenum - 1] + 50]

#     # Replace any NaNs in the observation matrix with zeros
#     obs[np.isnan(obs)] = 0

#     # Refine the list of onsets and offsets according to the number of notes
#     prior_ons = ons[:notenum]  # Ignore added 0 placeholder
#     prior_offs = offs[:notenum]
#     notes = len(prior_ons)  # Normalize

#     # Define states: silence, trans, steady state
#     # Rows: aperiodicity, power
#     state_ord_seed = [1, 2, 3, 2, 1]
#     # state_ord = np.tile(state_ord_seed[:-1], notes) + [state_ord_seed[-1]] # This is 21 size
#     state_ord = np.concatenate([np.tile(
#         state_ord_seed[:-1], notes), [state_ord_seed[-1]]])  # This gives both 20 size

#     # Use state_ord to expand means and covars for each appearance
#     midi_notes = np.tile(align['midiNote'][:notenum], len(state_ord_seed) - 1)
#     midi_notes = np.append(midi_notes, align['midiNote'][notenum - 1])
#     # Changed state_ord - 1
#     means_full = np.vstack((means[:, state_ord - 1], midi_notes))
#     covars = covars.reshape(3, 2, 2)
#     covars[0, 1, 0] = 100
#     covars[1, 1, 0] = 5
#     covars[2, 1, 0] = 1
#     covars_full = covars[state_ord - 1, :, :]  # deleted one :, to make 2-D

#     mixmat = np.ones(len(state_ord))

#     # Transition matrix seed
#     # {steady state, transient, silence, transient, steady state}
#     # Original, commented out 4th index to see results...
#     trans_seed = np.zeros((5, 5))
#     # trans_seed = np.zeros((4, 4))
#     trans_seed[0, 0] = 0.99
#     trans_seed[1, 1] = 0.98
#     trans_seed[2, 2] = 0.98
#     trans_seed[3, 3] = 0.98
#     trans_seed[4, 4] = 0.99
#     trans_seed[0, 1] = 0.0018
#     trans_seed[0, 2] = 0.0007
#     trans_seed[0, 3] = 0.0042
#     trans_seed[0, 4] = 0.0033
#     trans_seed[1, 2] = 0.0018
#     trans_seed[1, 3] = 0.0102
#     trans_seed[1, 4] = 0.0080
#     trans_seed[2, 3] = 0.0112
#     trans_seed[2, 4] = 0.0088
#     trans_seed[3, 4] = 0.02

#     # Call filltransmat to expand the transition matrix to the appropriate size
#     trans = fill_trans_mat(trans_seed, notes)

#     # Create starting state space matrix
#     starting_state = np.zeros(4 * notes + 1)
#     starting_state[0] = 1

#     prior = fill_priormat_gauss(obs.shape[0], prior_ons, prior_offs, 5)    
    
#     if learnparams:
#         # Use the fit function from the hmmlearn library to learn the HMM parameters
#         model = hmm.GMMHMM(n_components=5, n_mix=1,
#                            covariance_type='diag', n_iter=1)
#         model.startprob_ = starting_state
#         model.transmat_ = trans
#         model.means_ = means_full.T
#         model.covars_ = covars_full.T
#         model.fit(obs.T)

#     # like = mixgauss_prob(obs, means_full, covars_full, mixmat)

#     # Use the Viterbi algorithm to find the most likely path
#     # pr_like = prior * like
#     # vpath = hmm.ViterbiHMM(starting_state, trans, pr_like)

#     # Define the filename
#     # pLikeData = "./test_files/priorlike_oneNote_runHMM.csv"
#     pLikeData = "./test_files/priorlike_threeNote_runHMM.csv"
#     # pLikeData = "./test_files/priorlike_sixNote_runHMM.csv"
    
#     # Read the data from the file
#     dtype = {'index': str, 'value': float}
#     pr_like = pd.read_csv(pLikeData, dtype=dtype,
#                           sep='\s+', names=['index', 'value'])

#     # Initialize an empty dictionary to store the data
#     data_dict = {}

#     # Open the text file for reading
#     with open(pLikeData, 'r') as file:
#         # Iterate through each line in the file
#         for line in file:
#             # Split the line into two parts based on whitespace
#             parts = line.split()
#             # Extract the index from the first part and convert it to a tuple
#             index = tuple(map(int, parts[0].strip('()').split(',')))
#             # Parse the value from the second part
#             value = float(parts[1])
#             # Store the data in the dictionary with the index as the key
#             data_dict[index] = value

#     # Determine the shape of the numpy array based on the maximum index values
#     num_rows = max(index[0] for index in data_dict.keys())
#     num_cols = max(index[1] for index in data_dict.keys())

#     # Initialize a numpy array with zeros
#     pr_like = np.zeros((num_rows, num_cols))

#     # Fill the numpy array with the values from the dictionary
#     for index, value in data_dict.items():
#         pr_like[index[0] - 1, index[1] - 1] = value

#     vpath = viterbi_path(starting_state, trans, pr_like)
#     # vpath = librosa.sequence.viterbi(prob=starting_state, transition=trans, pr_like)
   
#     return vpath, starting_state, prior, trans, means_full, covars_full, mixmat, obs, state_ord

