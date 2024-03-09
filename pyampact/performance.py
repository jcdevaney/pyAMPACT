"""
performance
==============


.. autosummary::
    :toctree: generated/

    estimate_perceptual_parameters
    calculate_vibrato
    perceived_pitch    
"""

import numpy as np
import librosa
import warnings

__all__ = [
    "estimate_perceptual_parameters",
    "calculate_vibrato",
    "perceived_pitch"
]

np.seterr(all='ignore')
warnings.filterwarnings("ignore", message="Mean of empty slice.", category=RuntimeWarning)


def estimate_perceptual_parameters(f0_vals, pwr_vals, M, SR, hop, gt_flag, y):        
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
    res_vibrato_depth, res_vibrato_rate = calculate_vibrato(detrended_f0_vals, SR / hop)

    
    # Shimmer
    tmp_shimmer = 10 * np.log10(pwr_vals[1:] / pwr_vals[0])
    res_shimmer = np.mean(np.abs(tmp_shimmer))
    res_pwr_vals = 10 * np.log10(pwr_vals)
    res_f0_vals = f0_vals

    if gt_flag:
        M = np.abs(M) ** 2
    
    S, phase = librosa.magphase(librosa.stft(y=y))
    res_spec_centroid = librosa.feature.spectral_centroid(S=S)

    # Spectral Slope                
    mu_x = np.mean(M, axis=0)        
    kmu = np.arange(0, M.shape[0]) - M.shape[0] / 2    
    M_sqrt = np.sqrt(M)
    M_slope = M_sqrt - np.tile(mu_x, (M.shape[0], 1))

    # Check if M is empty
    if M.size == 0:
        # Handle the case where M is empty
        # print("Warning: M is empty. Cannot calculate mean or perform operations.")
        # You may choose to handle this situation differently based on your requirements
        # For example, you could assign default values to res_spec_flux and res_mean_spec_flux
        res_spec_flux = np.nan  # or any other appropriate value or action
        res_mean_spec_flux = np.nan  # or any other appropriate value or action
        res_spec_slope = np.nan  # or any other appropriate value or action
        res_mean_spec_slope = np.nan  # or any other appropriate value or action
    else:
        # Calculate res_spec_flux
        afDeltaX = np.diff(np.hstack((M[:, 0:1], M)), axis=1)
        res_spec_flux = np.sqrt(np.sum(afDeltaX**2, axis=0)) / M.shape[0]
        # Calculate res_mean_spec_flux
        res_mean_spec_flux = np.mean(res_spec_flux)
         # Calculate res_spec_slope
        res_spec_slope = np.dot(kmu, M_slope) / np.dot(kmu, kmu)
        # Calculate res_mean_spec_slope
        res_mean_spec_slope = np.mean(res_spec_slope)

    # res_spec_slope = np.dot(kmu, M_slope) / np.dot(kmu, kmu)
    # res_mean_spec_slope = np.mean(res_spec_slope)

    
    # # Spectral Flux    
    # afDeltaX = np.diff(np.hstack((M[:, 0:1], M)), axis=1)
    # res_spec_flux = np.sqrt(np.sum(afDeltaX**2, axis=0)) / M.shape[0]
    # res_mean_spec_flux = np.mean(res_spec_flux)
    

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
        "spec_centroid": res_spec_centroid,
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