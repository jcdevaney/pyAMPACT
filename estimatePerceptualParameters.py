# COMPLETE/TESTED
import numpy as np
# from synthtrax import synthtrax
from perceivedPitch import perceived_pitch
from calculateVibrato import calculate_vibrato

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