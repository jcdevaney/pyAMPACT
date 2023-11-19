import numpy as np
import sys





def yin2(x, p=None):
    """
    YIN2: a simple implementation of the YIN period-estimation algorithm
    
    Args:
    x (array-like): Input signal.
    p (dict): Parameters with the following optional keys:
        - 'maxf0' (int): Maximum search range in samples (default 100).
        - 'minf0' (int): Minimum search range in samples (default 2).
        - 'wsize' (int): Window size in samples (default maxf0).
        - 'hop' (int): Frame period in samples (default wsize).
        - 'thresh' (float): Threshold for period minimum (default 0.1).
        - 'smooth' (int): Size of low-pass smoothing window (default minf0/2).
    
    Returns:
    r (dict): Dictionary containing period and aperiodicity information.
        - 'prd' (array): Estimated periods.
        - 'ap' (array): Aperiodicity measures.
    """
    if p is None:
        p = {}
    
    
    # Defaults
    p.setdefault('maxf0', 100)
    p.setdefault('minf0', 2)
    p.setdefault('wsize', p['maxf0'])
    p.setdefault('hop', p['wsize'])
    p.setdefault('thresh', 0.1)
    p.setdefault('smooth', p['minf0'] // 2)
    
    
    x = np.asarray(x)
    
    if x.ndim > 1:
        raise ValueError('Data should be 1D')

    nsamples = len(x)    
    nframes = (nsamples - p['maxf0'] - p['wsize']) // p['hop']    
    
    pwr = np.zeros(int(nframes))
    prd = np.zeros(int(nframes))
    ap = np.zeros(int(nframes))
    
    # Shifted data
    # x = np.convolve(x, np.ones(p['maxf0'] + 1), mode='same')[p['maxf0']: -p['maxf0']]
    x = np.convolve(x, np.ones(int(p['maxf0']) + 1), mode='full')
    x = x[int(p['maxf0']):(-int(p['maxf0']))]
    
    for k in range(int(nframes)):        
        start = k * p['hop']
        xx = x[start : start + int(p['wsize'])]        
        xxTiled = np.tile(xx[0], (1, int(p['maxf0']) + 1))               
        
        xx = np.append(xx, 1) # BRUTE FORCE MAY NOT WORK, WRITE A BETTER COMPARATOR!
        xx = xx.reshape(xxTiled.shape)        

        d = np.mean((xx - xxTiled)**2, axis=0) / 2  # Squared difference function
        

        dd = d[1:] / (np.cumsum(d[1:]) / np.arange(1, p['maxf0']))
        
        # Parabolic interpolation of all triplets to refine local minima
        min_pos = np.arange(len(dd))  # Nominal position of each sample
        x1 = dd[:-2]
        x2 = dd[1:-1]
        x3 = dd[2:]
        a = (x1 + x3 - 2 * x2) / 2
        b = (x3 - x1) / 2
        shift = -b / (2 * a)  # Offset of interpolated minimum relative to the current sample
        val = x2 - b**2 / (4 * a)  # Value of interpolated minimum
        
        # Replace all local minima by their interpolated value
        idx = 1 + np.where((x2 < x1) & (x2 < x3))[0]
        dd[idx] = val[idx - 1]
        min_pos[idx] = min_pos[idx - 1] + shift[idx - 1]
        
        # Find the index of the first minimum below the threshold
        a = dd < p['thresh']
        if not np.any(a):
            prd0 = np.argmin(dd)  # None below the threshold, take the global minimum instead
        else:
            b = np.min(np.where(a))  # Left edge
            c = min(b * 2, len(a))
            prd0 = b + np.argmin(dd[b: c - 1])
        
        prd = min_pos[prd0] + 1
        
        if prd > 2 and prd < len(dd) and d[prd0] < d[prd0 - 1] and d[prd0] < d[prd0 + 1]:
            # Refine by parabolic interpolation of the raw difference function
            x1 = d[prd - 1]
            x2 = d[prd]
            x3 = d[prd + 1]
            a = (x1 + x3 - 2 * x2) / 2
            b = (x3 - x1) / 2
            shift = -b / (2 * a)  # Offset of interpolated minimum relative to the current sample
            val = x2 - b**2 / (4 * a)  # Value of interpolated minimum
            prd = prd + shift - 1
        
        # Aperiodicity         
        frac = prd - int(prd)
        if frac == 0:
            yy = xx[:,int(prd)]
        else:
            yy = (1 - frac) * xx[int(np.floor(prd) + 1)] + frac * xx[int(np.floor(prd) + 1) + 1]  # Linear interpolation
        pwr = (np.mean(xx[:, 0]**2) + np.mean(yy**2)) / 2  # Average power over fixed and shifted windows
        res = np.mean((xx[:, 0] - yy)**2) / 2
        print('res', res)
        print('pwr', pwr)
        ap = res / pwr
        
        prd = int(prd)
        r = []
        print('ap', ap)
        # r['prd'][k] = prd
        # r['ap'][k] = ap
        
        return ap, pwr

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    