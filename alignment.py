"""
alignment
==============


.. autosummary::
    :toctree: generated/

    run_alignment
    get_vals
    runDTWAlignment
    align_midi_wav
    alignment_visualiser
    ifgram
    get_ons_offs
    midi2nmat
    find_mids
    find_peaks
    find_steady
    freq_and_mag_matrices


"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import mido
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.pardir)

from f0EstWeightedSum import f0_est_weighted_sum
from f0EstWeightedSumSpec import f0_est_weighted_sum_spec

from performance import estimate_perceptual_parameters
from symbolic import Score
from alignmentUtils import orio_simmx, simmx, dp, maptimes




    
def run_alignment(filename, midiname, means, covars, width=3, target_sr=4000, nharm=3, win_ms=100):
    """    
    Calls the DTW alignment function and refines the results with the HMM
    alignment algorithm, with both a basic and modified state spaces (based on the lyrics).
    This function returns the results of both the state spaces as well as the YIN analysis
    of the specified audio file.

    :param filename: Name of the audio file.
    :param midiname: Name of the MIDI file.    
    :param means: Mean values for each state.
    :param covars: Covariance values for each state.
    :param width: Width parameter (you need to specify this value).
    :param target_sr: Target sample rate (you need to specify this value).
    :param win_ms: Window size in milliseconds (you need to specify this value).
    :param nharm: Number of harmonics (you need to specify this value).

    :returns: 
        - res.on list of DTW predicted onset times in seconds
        - res.off list of DTW predicted offset times in seconds        
        - spec: Spectrogram of the audio file.
        - yinres: Structure of results of running the YIN algorithm on the audio signal indicated by the input variable filename.
    """
    
    

    # Read audio file and perform DTW alignment and YIN analysis
    hop = 32
    audiofile, sr = librosa.load(filename, sr=4000)

    # Normalize audio file
    audiofile = audiofile / np.sqrt(np.mean(audiofile ** 2)) * 0.6

    # Run DTW alignment
    align, spec, dtw = runDTWAlignment(
        filename, midiname, 0.025, width, target_sr, nharm, win_ms)
        
    
    
    piece = Score(midiname)
    nmat = piece.nmats()
    notes = piece.midiPitches()

   # Get a list of column names
    column_names = notes.columns.tolist()    
    reference_column = column_names[0]

    # Number of notes to align
    pitch_list = [];    
    for note in notes[reference_column].values: # Hardcoded. Fix this?
        if note != -1: # Exclude rests
            pitch_list.append(note) 
            

    # FOR POLY
    # Harmonic-Percussive Source Separation
    harmonic, percussive = librosa.effects.hpss(audiofile)

    # Use the harmonic component for pitch estimation
    pitch, magnitudes = librosa.core.piptrack(y=harmonic, sr=sr)

    # Get the index of the maximum magnitude for each frame
    max_magnitude_index = np.argmax(magnitudes, axis=0)
    

    # Convert the index to frequency    
    pitch_frequencies = librosa.fft_frequencies(sr=sr, n_fft=len(audiofile))
    estimated_pitch = pitch_frequencies[max_magnitude_index]
    

    # Plot the pitch track
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(librosa.amplitude_to_db(magnitudes, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    
    plt.colorbar()
    plt.title('Estimated Pitch (Hz)')
    plt.yticks([])
    # plt.show()


     

    
    print(spec)
    sys.exit()
       
    
    return align, dtw, spec, estimated_pitch, onset_times, offset_times




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

    # Your align_midi_wav function returns values that we will use
    m, p, q, S, D, M, N = align_midi_wav(
        MF=midorig, WF=audiofile, TH=tres, ST=1, width=width, tsr=targetsr, nhar=nharm, wms=winms)

    dtw = {
        'M': m,
        'MA': p,
        'RA': q,
        'S': S,
        'D': D,
        'notemask': M,
        'pianoroll': N
    }

    print(dtw['MA'])
    print(dtw['RA'])
    sys.exit()
    nmat = midi2nmat(midorig) # Not accurate in polyphonic
    
    # piece = Score(midorig)
    # nmat = piece.nmats()    
    
    # Assuming align.nmat is a NumPy array
    nmat[:, 5] = nmat[:, 4] + nmat[:, 5]
    dtw_ma_array = np.array(dtw['MA'])
    dtw_ra_array = np.array(dtw['RA'])
    u = maptimes(nmat[:, 4:5], dtw_ma_array * tres, dtw_ma_array * tres)
    # u = maptimes(nmat[:, 4:5], dtw['MA'] * tres, dtw['RA'] * tres)
    print(u)
    # nmat = np.column_stack((nmat, col1_values, col2_values))

    

    


    # FOR FROM SYMBOLIC...
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
    
    sys.exit()
    align = {
        'nmat': nmat.copy(),  # Create an empty 2D array with 7 columns for nmat
        'on': np.empty(0),         # Create an empty 1D array
        'off': np.empty(0),        # Create an empty 1D array
        'midiNote': np.empty(0)    # Create an empty 1D array
    }
    
    
    sys.exit()
    
    

    # Assign 'on', 'off', and 'midiNote' values from nmat
    align['on'] = nmat['ONSET_SEC']
    align['off'] = nmat['OFFSET_SEC']
    align['midiNote'] = nmat['MIDI']


    sys.exit()

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

    d, sr = librosa.load(WF, sr=None, mono=False)

    # Use sgram bins below 1k
    targetsr = tsr
    srgcd = np.gcd(targetsr, sr)
    # dr = librosa.resample(d, orig_sr=sr, target_sr=targetsr // srgcd)

    
    # Choose win length as power of 2 closest to 100ms
    winms = wms
    fftlen = 2 ** round(np.log2(winms / 1000 * targetsr))
    # Actually take the spectrogram
    win = fftlen
    
    # d vs dr resample?
    ovlp = round(win - TH * targetsr)
    D = np.abs(librosa.core.stft(d, n_fft=fftlen, hop_length=ovlp, win_length=win))

    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=targetsr, hop_length=ovlp, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

    sys.exit()
    # # Calculate spectrogram
    # y, sr = librosa.load(WF)
    # fft_len = int(2**np.round(np.log(wms/1000*tsr)/np.log(2)))
    # hop_length = int((TH * tsr * 1000))
    # D = librosa.feature.melspectrogram(
    #     y=y, sr=tsr, n_fft=fft_len, hop_length=hop_length, window='hamming')
    

    N = np.array(sampled_grid)    

    mask = piece.mask(wms, tsr, nhar, width, bpm=60, aFreq=440,
                      base_note=0, tuning_factor=1, obs=20)    

    M = np.array(mask)
    M = M.astype(np.int16)

    # Calculate the peak-structure-distance similarity matrix
    if ST == 1:
        S = orio_simmx(M, D)
    else:
        S = simmx(M, D)
    

    # Ensure no NaNs (only going to happen with simmx)
    S[np.isnan(S)] = 0
    print(S)
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

  

def alignment_visualiser(mid, spec, fig=1):    
    """    
    Plots a gross DTW alignment overlaid with the fine alignment
    resulting from the HMM aligner on the output of YIN.  Trace(1,:)
    is the list of states in the HMM, and trace(2,:) is the number of YIN
    frames for which that state is occupied.  Highlight is a list of 
    notes for which the steady state will be highlighted.
    
    :param mid: Midi file.
    :param spec: Spectrogram of audio file (from align_midi_wav). 
    
    :return: Visualized spectrogram            

    """

    # hop size between frames
    stft_hop = 0.023  # Adjusted from 0.025

    # Read MIDI file    
    piece = Score(mid)
    notes = piece.midiPitches()
    
    
    # Plot spectrogram
    fig = plt.figure(fig)
    plt.imshow(20 * np.log10(spec), aspect='auto', origin='lower', cmap='gray')
    plt.title('Spectrogram with Aligned MIDI Notes Overlaid')
    plt.xlabel('Time (.05s)')
    plt.ylabel('Midinote')
    plt.clim([plt.gci().get_clim()[0], plt.gci().get_clim()[1] - 50])
    plt.colorbar()
    
    
    plt.show()



def ifgram(audiofile, tsr, win_ms):
    # win_samps = int(tsr / win_ms) # low-res
    win_samps = 2048 # Placeholder for now, default
    y, sr = librosa.load(audiofile)
    
    freqs, times, mags = librosa.reassigned_spectrogram(y=y, sr=tsr,
                                                      n_fft=win_samps)
  
        
    mags_db = librosa.amplitude_to_db(mags, ref=np.max)
    sig_pwr = mags ** 2 # power of signal, magnitude/amplitude squared


    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    img = librosa.display.specshow(mags_db, x_axis="s", y_axis="linear",sr=tsr, hop_length=win_samps//4, ax=ax[0])
    ax[0].set(title="Spectrogram", xlabel=None)
    ax[0].label_outer()
    ax[1].scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)
    ax[1].set_title("Reassigned spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
                  
    
    plt.show()    
    
    return freqs, times, sig_pwr

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