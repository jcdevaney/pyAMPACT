"""
alignment
==============


.. autosummary::
    :toctree: generated/

    run_alignment    
    runDTWAlignment
    align_midi_wav
    alignment_visualiser
    ifgram
    get_ons_offs    
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

from scipy.signal import spectrogram

from performance import estimate_perceptual_parameters
from symbolic import Score
from alignmentUtils import orio_simmx, simmx, dp, maptimes




    
def run_alignment(filename, midiname, means, covars, width=3, target_sr=4000, nharm=3, win_ms=100, hop=32):
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
    :param nharm: Number of harmonics (you need to specify this value).
    :param win_ms: Window size in milliseconds (you need to specify this value).

    # :returns: 
    #     - align:
    #     - dtw:
    #     - res.on list of DTW predicted onset times in seconds
    #     - res.off list of DTW predicted offset times in seconds        
    #     - spec: Spectrogram of the audio file.        
    """
    
    

    # Read audio file and perform DTW alignment and YIN analysis    
    audiofile, sr = librosa.load(filename, sr=4000)

    # Normalize audio file
    audiofile = audiofile / np.sqrt(np.mean(audiofile ** 2)) * 0.6

    # Run DTW alignment
    align, spec, dtw = runDTWAlignment(
        filename, midiname, 0.025, width, target_sr, nharm, win_ms)
        
               
    return align, dtw, spec




def runDTWAlignment(audiofile, midifile, tres, width, target_sr, nharm, win_ms):    
    """
    Perform a dynamic time warping alignment between specified audio and MIDI files.

    Returns a matrix with the aligned onset and offset times (with corresponding MIDI
    note numbers) and a spectrogram of the audio.

    :param audiofile: Audio file.
    :param midifile: MIDI file.
    :param tres: Time resolution for MIDI to spectrum information conversion.
    :param width: Width parameter (you need to specify this value).    
    :param target_sr: Target sample rate (you need to specify this value).    
    :param nharm: Number of harmonics (you need to specify this value).
    :param win_ms: Window size in milliseconds (you need to specify this value).
    

    :returns:
        - align: dynamic time warping MIDI-audio alignment structure
            - align.on: onset times
            - align.off: offset times
            - align.midiNote: MIDI note numbers
        - spec: spectrogram
        - dtw: dict of dynamic time warping returns
            - M: map s.t. M(:,m).            
            - MA/RA [p,q]: path from DP
            - S: similarity matrix
            - D: spectrogram 
            - notemask: midi-note-derived mask
            - pianoroll: midi-note-derived pianoroll 
    """
    

    # Now done in alignMidiWav
    # y, sr = librosa.load(audiofile)
    # spec = librosa.feature.melspectrogram(y=y, sr=sr)

    # Your align_midi_wav function returns values that we will use
    m, p, q, S, D, M, N = align_midi_wav(
        MF=midifile, WF=audiofile, TH=tres, ST=1, width=width, tsr=target_sr, nhar=nharm, wms=win_ms)

    dtw = {
        'M': m,
        'MA': p,
        'RA': q,
        'S': S,
        'D': D,
        'notemask': M,
        'pianoroll': N
    }
    
    spec = dtw['D']
    
    
    piece = Score(midifile)

    # Returns a dict, not df
    nmat = piece.nmats() 
      
    
    align = {
        'nmat': nmat.copy(),  # Create an empty 2D array with 7 columns for nmat
        'on': np.empty(0),         # Create an empty 1D array
        'off': np.empty(0),        # Create an empty 1D array
        'midiNote': np.empty(0)    # Create an empty 1D array
    }
    
    # loop through voices
    for key, df in nmat.items():        
        onset_sec = df['ONSET_SEC']
        offset_sec = df['OFFSET_SEC']
        midi_notes = df['MIDI']          
       
    
    # Convert Series to NumPy arrays and reshape
    onset_sec_array = onset_sec.values.reshape(-1, 1)
    offset_sec_array = offset_sec.values.reshape(-1, 1)
    
    combined_slice = np.hstack((onset_sec_array, offset_sec_array))    

    
    dtw['MA'] = np.array(dtw['MA']) * tres
    dtw['RA'] = np.array(dtw['RA']) * tres  
    
    x = maptimes(combined_slice, dtw['MA'], dtw['RA'])

    # # Assign 'on', 'off', and 'midiNote' values from nmat
    align['on'] = x[:,0]
    align['off'] = x[:,1]
    align['midiNote'] = midi_notes
    spec = D # from align_midi_wav


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
    y, sr = librosa.load(WF)
    fft_len = int(2**np.round(np.log(wms/1000*tsr)/np.log(2))) # Good, 512
    
    ovlp = round(fft_len - TH*tsr);    
    
    y = librosa.resample(y, orig_sr=sr, target_sr=tsr)
    # Generate a sample signal (replace this with your own signal)
    
    # srgcd = math.gcd(tsr, sr)
    # dr = librosa.resample(y,tsr/srgcd,sr/srgcd)
    # dr = resample(y,tsr/srgcd,sr/srgcd);
    
    
    # Compute spectrogram
    f, t, Sxx = spectrogram(y, fs=tsr, nperseg=fft_len, noverlap=ovlp, window='hann')
    # Access the magnitude spectrogram (D)
    D = np.abs(Sxx)
    
    # Plot spectrogram
    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(D), shading='auto', vmax=np.max(10 * np.log10(D)))  # Use vmax to set the color scale
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram')    
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    # plt.show()
        

    
    # First mask declaration here follows the MATLAB params, but not sure
    # these are necessary at this point.
    # mask = piece.mask(wms, tsr, nhar, width, bpm=60, aFreq=440,
    #                   base_note=0, tuning_factor=1, obs=20)     
    M = piece.mask(sample_rate=tsr)
    # Save M to a CSV file
    # np.savetxt('./output_files/output.csv', M, delimiter=',')
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
    
    
    # plt.show()



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
                  
    
    # plt.show()
    
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