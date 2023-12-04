# PASS
import numpy as np
from scipy.fftpack import dct, idct

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