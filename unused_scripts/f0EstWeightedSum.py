# NEEDS TESTING
import numpy as np

def f0_est_weighted_sum(x, f, f0i, fMax, fThresh):
    # Set default values if not provided
    if fMax is None:
        fMax = 5000
    if fThresh is None:
        fThresh = 2 * np.median(np.diff(f[:, 0]))

    x2 = np.abs(x) ** 2
    wNum = np.zeros_like(x2)
    wDen = np.zeros_like(x2)
    strips = {}

    maxI = int(np.max(fMax / f0i))
    for i in range(1, maxI + 1):
        strip = (np.abs(f - f0i * i) < fThresh) * x2        
        strips[i] = strip
        np.add(wNum, (1 / i) * strip)
        np.add(wDen, strip)

    
    np.multiply(wNum, (f < fMax))
    np.multiply(wDen, (f < fMax))

    
    f0 = np.sum(wNum * f, axis=0) / np.sum(wDen, axis=0)
    pow = np.sum(wDen, axis=0)
    

    return f0, pow, strips

# Example usage:
# Replace x, f, f0i with your input data
# f0, pow, strips = f0EstWeightedSum(x, f, f0i)
