# PASS, matches MATLAB
import numpy as np
from scipy.interpolate import interp1d

def smooth_note(x, x_mid, y_mid):
    # Make a note the same size as x
    smoothed = np.zeros_like(x)

    # But only populate it with non-zero elements between the x_mid values
    x_range = np.arange(min(x_mid), max(x_mid) + 1)

    # Interpolate the mid points at all of the sample points in the signal
    interp_func = interp1d(x_mid, y_mid, kind='linear', fill_value='extrapolate')    
    smoothed[x_range] = interp_func(x_range)        
    return smoothed
