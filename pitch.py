# COMPLETE/TESTED
import numpy as np
from scipy.interpolate import interp1d
# from synthtrax import synthtrax
from scipy.fftpack import dct, idct

def estimate_perceptual_parameters(f0_vals, pwr_vals, F, M, SR, hop, gt_flag, X=1):
    
    # # Return to this later...if necessary
    # # This constructs X based on matrices, don't need?
    # if X is None: # nargs < 8
    #     win_s = 0.064
    #     WIN = int(win_s * SR)
    #     nHOP = int(WIN / 4)

    #     # Filter out rows with zero magnitude sum
    #     M2 = M[np.sum(M, axis=1) != 0, :]
    #     F2 = F[np.sum(M, axis=1) != 0, :]

        
    #     X = synthtrax(F2, M2, SR, WIN, nHOP)

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
        # "spec_centroid": res_spec_centroid,
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


import numpy as np

def get_cent_vals(times, yinres, sr):
    # CENTS IN BETWEEN SEQUENTIAL NOTES
    # Check the reference of yin in MATLAB vs yin in librosa
    """
    Get cent values (in relation to A, 440 Hz) for each note.
    

    Parameters:
    - times: Dictionary-like object with 'ons' and 'offs' representing onset and offset times.
    - yinres: Dictionary-like object with 'f0' and 'sr'.

    Returns:
    - cents: List of NumPy arrays containing cent values for each note.
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
    # Make a note the same size as x
    smoothed = np.zeros_like(x)

    # But only populate it with non-zero elements between the x_mid values
    x_range = np.arange(min(x_mid), max(x_mid) + 1)

    # Interpolate the mid points at all of the sample points in the signal
    interp_func = interp1d(x_mid, y_mid, kind='linear', fill_value='extrapolate')    
    smoothed[x_range] = interp_func(x_range)        
    return smoothed




#########################################################################
# [coefs approx] = noteDct(x, Ndct, sr)
#
# Description: Compute the DCT of a signal and approximate it with the 
#              first Ndct coefficients  x is the signal  Ndct is the number 
#              of DCT coefficients to be calculated sr is the sampling rate 
#              of the signal
#
# Inputs:
#  x - signal to be analyzed
#  Ndct - number of DCT coefficients to be calculated
#  sr - sampling rate
#
# Outputs:
#  coefs - DCT coefficients
#  approx - reconstruction of X using the Ndct number of DCT coefficients
#
# Automatic Music Performance Analysis and Analysis Toolkit (AMPACT) 
# http://www.ampact.org
# (c) copyright 2011 Johanna Devaney (j@devaney.ca) and Michael Mandel
#                    (mim@mr-pc.org), all rights reserved
#########################################################################

def note_dct(x, Ndct, sr):
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