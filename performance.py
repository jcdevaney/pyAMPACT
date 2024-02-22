"""
performance
==============


.. autosummary::
    :toctree: generated/

    estimate_perceptual_parameters
    calculate_vibrato
    perceived_pitch
    get_cent_vals
    smooth_note
    note_dct
    find_mids
    find_peaks
    find_steady


"""

import numpy as np
from scipy.interpolate import interp1d
# from synthtrax import synthtrax
from scipy.fftpack import dct, idct

__all__ = [
    "estimate_perceptual_parameters",
    "calculate_vibrato",
    "perceived_pitch",
    "get_cent_vals",
    "smooth_note",
    "note_dct",
    "find_mids",
    "find_peaks",
    "find_steady"
]

def estimate_perceptual_parameters(f0_vals, pwr_vals, M, SR, hop, gt_flag, X=1):        
    """
    NEEDS INFO UPDATED
    """

    # Perceived pitch
    res_ppitch = perceived_pitch(f0_vals, SR)    
    # Jitter
    tmp_jitter = np.abs(np.diff(f0_vals))
    res_jitter = np.mean(tmp_jitter)

    # Vibrato rate and depth
    mean_f0_vals = np.mean(f0_vals)
    detrended_f0_vals = f0_vals - mean_f0_vals
    # res_vibrato_depth, res_vibrato_rate = calculate_vibrato(detrended_f0_vals, SR / hop)

    
    # Shimmer
    tmp_shimmer = 10 * np.log10(pwr_vals[1:] / pwr_vals[0])
    res_shimmer = np.mean(np.abs(tmp_shimmer))
    res_pwr_vals = 10 * np.log10(pwr_vals)
    res_f0_vals = f0_vals

    if gt_flag:
        M = np.abs(M) ** 2
    
    # res_spec_centroid = np.sum(F * M) / np.sum(M)

    # Spectral Slope                
    mu_x = np.mean(M, axis=0)        
    kmu = np.arange(0, M.shape[0]) - M.shape[0] / 2    
    M_sqrt = np.sqrt(M)
    M_slope = M_sqrt - np.tile(mu_x, (M.shape[0], 1))    
    res_spec_slope = np.dot(kmu, M_slope) / np.dot(kmu, kmu)    
    res_mean_spec_slope = np.mean(res_spec_slope)

    
    # Spectral Flux    
    afDeltaX = np.diff(np.hstack((M[:, 0:1], M)), axis=1)
    res_spec_flux = np.sqrt(np.sum(afDeltaX**2, axis=0)) / M.shape[0]
    res_mean_spec_flux = np.mean(res_spec_flux)
    

    # Spectral Flatness
    XLog = np.log(M + 1e-20)
    res_spec_flat = np.exp(np.mean(XLog, axis=0)) / np.mean(M, axis=0)
    res_spec_flat[np.sum(M, axis=0) == 0] = 0
    res_mean_spec_flat = np.mean(res_spec_flat)


    res = {
        "ppitch": res_ppitch,
        "jitter": res_jitter,
        # "vibrato_depth": res_vibrato_depth,
        # "vibrato_rate": res_vibrato_rate,
        "shimmer": res_shimmer,
        "pwr_vals": res_pwr_vals,
        "f0_vals": res_f0_vals,        
        # "spec_centroid": res_spec_centroid, # can get from librosa
        "spec_slope": res_spec_slope,
        "mean_spec_slope": res_mean_spec_slope,
        "spec_flux": res_spec_flux,
        "mean_spec_flux": res_mean_spec_flux,
        "spec_flat": res_spec_flat,
        "mean_spec_flat": res_mean_spec_flat,
    }

    return res


def calculate_vibrato(note_vals, sr):    
    L = len(note_vals)  # Length of signal
    Y = np.fft.fft(note_vals) / L  # Run FFT on normalized note vals
    w = np.arange(0, L) * sr / L  # Set FFT frequency grid    
    
    vibrato_depth_tmp, noteVibratoPos = max(abs(Y)), np.argmax(abs(Y))  # Find the max value and its position
    vibrato_depth = vibrato_depth_tmp * 2  # Multiply the max by 2 to find depth (above and below zero)
    vibrato_rate = w[noteVibratoPos]  # Index into FFT frequency grid to find position in Hz

    return vibrato_depth, vibrato_rate


def perceived_pitch(f0s, sr, gamma=100000):
    """
    Calculate the perceived pitch of a note based on 
    Gockel, H., B.J.C. Moore,and R.P. Carlyon. 2001. 
    Influence of rate of change of frequency on the overall 
    pitch of frequency-modulated Tones. Journal of the 
    Acoustical Society of America. 109(2):701?12.

    :param f0s: Vector of fundamental frequency estimates
    :param sr: 1/sample rate of the f0 estimates (e.g. the hop rate in Hz of yin)
    :param gamma: Sets the relative weighting of quickly changing vs slowly 
        changing portions of  notes. - a high gamma (e.g., 1000000)  
        gives more weight to slowly changing portions.

    :returns:
        - pp1: perceived pitch using the entire vector of f0 estimates
        - pp2: perceived pitch using the central 80% of f0 estimates
    """

    # Remove all NaNs in the f0 vector
    f0s = f0s[~np.isnan(f0s)]

    # Create an index into the f0 vector to remove outliers by
    # only using the central 80% of the sorted vector
    ord = np.argsort(f0s)
    ind = ord[int(np.ceil(len(f0s) * 0.1)):int(np.floor(len(f0s) * 0.9))]

    # Calculate the rate of change
    deriv = np.diff(f0s) * sr
    deriv = np.append(deriv, -100)  # Append a value to match MATLAB behavior

    # Set weights for the quickly changing vs slowly changing portions
    weights = np.exp(-gamma * np.abs(deriv))

    # Calculate two versions of the perceived pitch    
    pp1 = np.dot(f0s, weights) / np.sum(weights)
    pp2 = np.dot(f0s[ind], weights[ind]) / np.sum(weights[ind])

    return pp1, pp2


def get_cent_vals(times, yinres, sr):
    """
    Get cent values (in relation to A, 440 Hz) for each note.
    
    :param times: Dictionary-like object with 'ons' and 'offs' representing onset and offset times.
    :param yinres: Dictionary-like object with 'f0' and 'sr'.

    :return: List of NumPy arrays containing cent values for each note.
    """

    cents = []

    # Index into f0 estimates in YIN structure with onset and offset times
    for i in range(len(times['ons'])):
        # Define the reference pitch (A4 = 440 Hz)
        ref_pitch = 440.0

        onset_index = round(times['ons'][i] / 32 * sr)
        offset_index = round(times['offs'][i] / 32 * sr)                        

        # Example array of frequencies
        frequencies = yinres['f0'][onset_index:offset_index]

        # Calculate the cent values for each frequency
        cent = 1200 * np.log2(frequencies / ref_pitch)
        
        # average_cent = np.mean(cent)        
        
        # WITHOUT AVERAGING        
        # cents.append(cent)

        # Check if the array is not empty and contains finite values
        if np.any(np.isfinite(cent)):
        # Calculate the mean only if the array is not empty and has finite values
            # average_cent = np.nanmean(cent)
            # cents.append(average_cent)
            cents.append(cent)            
        else:
            break

    return cents  


def smooth_note(x, x_mid, y_mid):
    """
    Generate a smoothed trajectory of a note by connecting the
    midpoints between peaks and troughs.

    :param x: Inputted signal.
    :param x_mid: Midpoint locations in x axis between peaks and troughs. 
    :param y_mid: Midpoint locations in y axis between peaks and troughs.

    :return: Smoothed version of inputted signal x.
    """
    # Make a note the same size as x
    smoothed = np.zeros_like(x)

    # But only populate it with non-zero elements between the x_mid values
    x_range = np.arange(min(x_mid), max(x_mid) + 1)

    # Interpolate the mid points at all of the sample points in the signal
    interp_func = interp1d(x_mid, y_mid, kind='linear', fill_value='extrapolate')    
    smoothed[x_range] = interp_func(x_range)        
    return smoothed


def note_dct(x, Ndct, sr):
    """
    Compute the DCT of a signal and approximate it with the 
    first Ndct coefficients  x is the signal  Ndct is the number 
    of DCT coefficients to be calculated sr is the sampling rate 
    of the signal

    :param x: Signal to be analyzed.
    :param Ndct: Number of DCT coefficients to be calculated.
    :param sr: Sampling rate.

    :returns:    
        - coefs: DCT coefficients.
        - approx: Reconstruction of X using the Ndct number of DCT coefficients.
    """
    # Calculate DCT coefficients using librosa's dct function    
    coefsTmp = dct(x)
    coefsTmp[min(len(coefsTmp), Ndct):] = 0

    # Divide by square root of N so that everything is divided by N instead of
    # the square root of N, because it is already divided by the sqrt of N
    coefs = coefsTmp[:min(Ndct, len(coefsTmp))] / np.sqrt(len(coefsTmp))

    # The sampling rate divided by the length of the signal is the lowest
    # frequency represented by the DCT.  Multiplying by it makes the 1st
    # coefficient into cents/second. For curves of constant slope, this makes
    # the 1st coefficient approximately independent of the length of the
    # signal. Multiplying by that frequency squared makes the 2nd coefficient into
    # cents/second^2. For curves of constant 2nd derivative, this makes the 2nd
    # coefficient approximately independent of the length of the signal, etc.
    #
    # For 2nd coefficient, multiple by -1 so that it represents positive slope
    coefs[1:] = coefs[1:] * (sr / len(x)) ** np.arange(1, len(coefs))
    coefs[1] = -coefs[1]  

    # reconstruct X using the DCT coefficients
    approx = idct(coefsTmp, type=2, norm='ortho')    

    
    return coefs, approx


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

