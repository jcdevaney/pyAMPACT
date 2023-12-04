# COMPLETE

import numpy as np

def mag_sq(x):
# Compute magnitude squared of complex matrix x.
#
# x2 = magSq(x)
# https://en.moonbooks.org/Articles/How-to-create-a-matrix-of-complex-numbers-in-python-using-numpy-/
# In order of speed (fastest first):
    # x2 = np.power(x.real,2) + np.power(x.imag,2)
    #GPT way
    x2 = np.real(x)**2 + np.imag(x)**2
    return x2
# x2 = x .* conj(x);
# x2 = x .* (x'.');
# x2 = abs(x).^2;
