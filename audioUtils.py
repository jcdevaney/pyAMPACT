# Put audioUtils into audioHelpers

import numpy as np
import librosa
import matplotlib.pyplot as plt


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


