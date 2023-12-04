# COMPLETE NEEDS TESTING

import numpy as np

def synthtrax(F, M, SR, SUBF=128, DUR=0):
    if DUR == 0:
        opsamps = 1 + ((F.shape[1] - 1) * SUBF)
    else:
        opsamps = int(DUR * SR)

    X = np.zeros(opsamps)

    for row in range(F.shape[0]):
        mm = M[row, :]
        ff = F[row, :]
        
        # Replace NaN values with zeros
        mm[np.isnan(mm)] = 0
        ff[np.isnan(ff)] = 0
        
        nzv = np.where(mm != 0)[0]
        firstcol = np.min(nzv)
        lastcol = np.max(nzv)
        
        zz = np.arange(max(1, firstcol - 1), min(F.shape[1], lastcol + 2))
        if len(zz) > 0:
            mm = mm[zz]
            ff = ff[zz]
            nzcols = len(zz)
            
            mz = (mm == 0)
            mask = mz & (np.roll(mz, shift=-1) == 0)
            ff = ff * (1 - mask) + mask * np.roll(ff, shift=-1)
            
            mask = mz & (np.roll(mz, shift=1) == 0)
            ff = ff * (1 - mask) + mask * np.roll(ff, shift=1)
            
            ff = slinterp(ff, SUBF)
            mm = slinterp(mm, SUBF)
            
            pp = np.cumsum(2 * np.pi * ff / SR)
            xx = mm * np.cos(pp)
            
            base = 1 + SUBF * (zz[0] - 1)
            sizex = len(xx)
            ww = np.arange(base - 1, base - 1 + sizex)
            X[ww] = X[ww] + xx

    return X

def slinterp(X, F):
    sx = len(X)
    X1 = np.roll(X, shift=-1)
    XX = np.zeros((F, sx))

    for i in range(F):
        XX[i, :] = ((F - i) / F) * X + (i / F) * X1

    Y = XX[0:((sx - 1) * F + 1)]
    return Y

# Example usage
if __name__ == "__main__":
    # Example input data (replace with your own data)
    F = np.random.rand(2, 100)
    M = np.random.rand(2, 100)
    SR = 44100
    SUBF = 128
    DUR = 5.0

    X = synthtrax(F, M, SR, SUBF, DUR)
    # You now have the synthesized audio signal in X