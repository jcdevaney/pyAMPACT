# CONDITIONAL PASS - see test
import numpy as np
from scipy.signal import gaussian

#########################################################################
# prior = fillpriormat_gauss(Nobs,ons,offs,Nstates)
#
# Description:
#  Creates a prior matrix based on the DTW alignment (supplied by the input
#  variables ons and offs. A rectangular window with half a Gaussian on
#  each side over the onsets and offsets estimated by the DTW alignment.
#
# Inputs:
#  Nobs - number of observations
#  ons - vector of onset times predicted by DTW alignment
#  offs - vector of offset times predicted by DTW alignment
#  Nstates - number of states in the hidden Markov model
#
# Outputs: 
#  prior - prior matrix based on DTW alignment
#
# Automatic Music Performance Analysis and Analysis Toolkit (AMPACT) 
# http://www.ampact.org - Johanna Devaney, 2011
# (c) copyright 2011 Johanna Devaney (j@devaney.ca) and Michael Mandel
#                    (mim@mr-pc.org), all rights reserved.
#########################################################################

def fill_priormat_gauss(Nobs, ons, offs, Nstates):    
    if Nstates is None:
        Nstates = 5

    Nnotes = len(ons)
    prior = np.zeros((Nnotes * (Nstates - 1) + 1, Nobs))
    frames = np.arange(1, Nobs + 1)

    for i in range(Nnotes):
        row = (i - 1) * (Nstates - 1)
        insert = Nstates - 5

        # Silence
        prior[row + 1, :] = flatTopGaussian(frames, gh(ons, i - 1, offs, i - 1, frames, 0.5),
                                            g(offs, i - 1, frames), g(ons, i, frames), gh(ons, i, offs, i, frames, 0.5))
    
        # Throws Value Error for negative values, both here and in MATLAB
        prior[row + 2:row + 2 + insert - 1, :] = np.tile(prior[row + 1, :], (insert, 1))

        # Transient, steady state, transient
        prior[row + 2 + insert, :] = flatTopGaussian(frames, g(offs, i - 1, frames),
                                                     gh(offs, i - 1, ons, i, frames, 0.75),
                                                     gh(ons, i, offs, i, frames, 0.25), g(offs, i, frames))
        prior[row + 3 + insert, :] = flatTopGaussian(frames, g(offs, i - 1, frames),
                                                     g(ons, i, frames), g(offs, i, frames), g(ons, i + 1, frames))
        prior[row + 4 + insert, :] = flatTopGaussian(frames, g(ons, i, frames),
                                                     gh(ons, i, offs, i, frames, 0.75),
                                                     gh(offs, i, ons, i + 1, frames, 0.25), g(ons, i + 1, frames))

    # The last silence
    i += 1
    prior[row + 5 + insert, :] = flatTopGaussIdx(frames, ons, i - 1, offs, i - 1, offs, i, ons, i + 1)

    return prior


def gh(v1, i1, v2, i2, domain, frac=0.5):
    x1 = g(v1, i1, domain)
    x2 = g(v2, i2, domain)
    return int(frac * x1 + (1 - frac) * x2)

#########################################################################
# x = gh(v1, i1, v2, i2, domain, frac) 
#
# Description:
#   Get an element that is frac fraction of the way between v1(i1) and
#   v2(i2), but check bounds on both vectors.  Frac of 0 returns v1(i1), 
#   frac of 1 returns v2(i2), frac of 1/2 (the default) returns half way 
#   between them.
#
# Automatic Music Performance Analysis and Analysis Toolkit (AMPACT) 
# http://www.ampact.org
# (c) copyright 2011 Johanna Devaney (j@devaney.ca) and Michael Mandel
#                    (mim@mr-pc.org), all rights reserved.
#########################################################################
    


def flatTopGaussIdx(x, b1, bi1, t1, ti1, t2, ti2, b2, bi2):
    b1 = g(b1, bi1, x)
    t1 = g(t1, ti1, x)
    t2 = g(t2, ti2, x)
    b2 = g(b2, bi2, x)
    return flatTopGaussian(x, b1, t1, t2, b2)

#########################################################################
# flatTopGaussIdx(x, b1,bi1, t1,ti1, t2,ti2, b2,bi2)
# 
# Description:
#   Create a window function that is zeros, going up to 1s with the left
#   half of a gaussian, then ones, then going back down to zeros with
#   the right half of another gaussian.  b1(bi1) is the x coordinate 2
#   stddevs out from the mean, which is at t1(ti1).  t2(ti2) is the x
#   coordinate of the mean of the second gaussian and b2(bi2) is 2
#   stddevs out from that.  The points should be in that order.  Vectors
#   are indexed intelligently, so you don't have to worry about
#   overflows or underflows.  X is the set of points over which this is
#   to be calculated.
#
# Automatic Music Performance Analysis and Analysis Toolkit (AMPACT) 
# http://www.ampact.org
# (c) copyright 2011 Johanna Devaney (j@devaney.ca) and Michael Mandel
#                    (mim@mr-pc.org), all rights reserved.
#########################################################################


def g(vec, idx, domain):
    if idx < 1:
        return domain[0]
    elif idx > len(vec):
        return domain[-1]
    else:
        return vec[idx - 1]

#########################################################################
# x = g(vec, idx, domain)
# 
# Description:
#   Get an element from vec, checking bounds.  Domain is the set of points
#   that vec is a subset of.
#
# Automatic Music Performance Analysis and Analysis Toolkit (AMPACT) 
# http://www.ampact.org
# (c) copyright 2011 Johanna Devaney (j@devaney.ca) and Michael Mandel
#                    (mim@mr-pc.org), all rights reserved.
#########################################################################


def flatTopGaussian(x, b1, t1, t2, b2):
    if any([b1, t1, t2]) > any([t1, t2, b2]):
        print('Endpoints are not in order: ', b1, t1, t2, b2)

#########################################################################
# flatTopGaussian(x, b1, t1, t2, b2)
# 
# Description:
#   Create a window function that is zeros, going up to 1s with the left 
#   half of a gaussian, then ones, then going back down to zeros with the 
#   right half of another gaussian.  b1 is the x coordinate 2 stddevs out 
#   from the mean, which is at t1.  t2 is the x coordinate of the mean of 
#   the second gaussian and b2 is 2 stddevs out from that.  The points 
#   should be in that order.  X is the set of points over which this is 
#   to be calculated.
#
# Automatic Music Performance Analysis and Analysis Toolkit (AMPACT) 
# http://www.ampact.org
# (c) copyright 2011 Johanna Devaney (j@devaney.ca) and Michael Mandel
#                    (mim@mr-pc.org), all rights reserved.
#########################################################################
    


    def custom_normalize(arr):
        return arr / np.max(np.abs(arr))

    def custom_gaussian(x, std):
        win = gaussian(2 * int(4 * std) + 1, std)
        return np.convolve(x, win, mode='same')

    def custom_gaussian_filter(x, t1, t2, b1, b2):
        left_std = (t1 - b1) / 2 + 1
        middle = np.ones(t2 - t1 - 1)
        right_std = (b2 - t2) / 2 + 1

        left = custom_normalize(custom_gaussian(x, left_std))
        right = custom_normalize(custom_gaussian(x, right_std))

        takeOneOut = t1 == t2
        w = np.concatenate((left[0:t1], middle, right[t2 + takeOneOut:]))
        return w


    # left = librosa.util.normalize(librosa.filters.gaussian(x, std=(t1 - b1) / 2 + 1))
    # middle = np.ones(t2 - t1 - 1)
    # right = librosa.util.normalize(librosa.filters.gaussian(x, std=(b2 - t2) / 2 + 1))

    # takeOneOut = t1 == t2
    # w = np.concatenate((left[0:t1], middle, right[t2 + takeOneOut:]))
    # return w
