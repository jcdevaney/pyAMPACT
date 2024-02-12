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


"""

import numpy as np
from scipy.interpolate import interp1d
# from synthtrax import synthtrax
from scipy.fftpack import dct, idct

def estimate_perceptual_parameters(f0_vals, pwr_vals, F, M, SR, hop, gt_flag, X=1):        
    """
    NEEDS INFO UPDATED
    """

    # Perceived pitch
    res_ppitch = perceived_pitch(f0_vals, SR / hop, 1)    
    # Jitter
    tmp_jitter = np.abs(np.diff(f0_vals))
    res_jitter = np.mean(tmp_jitter)

    # Vibrato rate and depth
    mean_f0_vals = np.mean(f0_vals)
    detrended_f0_vals = f0_vals - mean_f0_vals
    res_vibrato_depth, res_vibrato_rate = calculate_vibrato(detrended_f0_vals, SR / hop)

    
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
        "vibrato_depth": res_vibrato_depth,
        "vibrato_rate": res_vibrato_rate,
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
    # print(Y)
    
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
        - res.ons: List of onset times
        - res.offs: List of offset times
    """

    # Remove NaN values from f0s
    f0s = f0s[~np.isnan(f0s)]
    
    # Create an index to remove outliers by using the central 80% of the sorted vector
    ord = np.argsort(f0s)
    ind = ord[int(len(ord)*0.1):int(len(ord)*0.9)]

    # Calculate the rate of change
    deriv = np.append(np.diff(f0s) * sr, -100)        
            
    # Set weights for the quickly changing vs slowly changing portions
    # WEIGHTS ARE 0., incorrect!!
    weights = np.exp(-gamma * np.abs(deriv))

    # But is this?
    # weights = np.exp(-gamma / np.abs(deriv))    
    
    # Calculate two versions of the perceived pitch
    pp1 = np.sum(f0s * weights) / np.sum(weights)
    pp2 = np.sum(f0s[ind] * weights[ind]) / np.sum(weights[ind])
    
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
