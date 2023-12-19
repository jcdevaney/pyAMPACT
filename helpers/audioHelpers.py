import numpy as np
from scipy.signal import gaussian
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


def dp(M):
    r, c = M.shape

    # Initialize cost matrix D - PROBLEM!
    D = np.zeros((r + 1, c + 1))
    # D[0, :] = np.nan
    # D[:, 0] = np.nan
    # D[0, 0] = 0
    D[1:, 1:] = M

    # Initialize traceback matrix phi
    phi = np.zeros((r, c), dtype=int)

    # Dynamic programming loop
    for i in range(r):
        for j in range(c):
            dmax = min(D[i, j], D[i, j + 1], D[i + 1, j])
            tb = 1 if dmax == D[i, j] else (2 if dmax == D[i, j + 1] else 3)

            # dmax, tb = min([D[i, j], D[i, j + 1], D[i + 1, j]])
            D[i + 1, j + 1] = D[i + 1, j + 1] + dmax
            phi[i, j] = tb

    # Traceback from top left
    i = r - 1
    j = c - 1
    p = [i]
    q = [j]
    while i > 0 and j > 0:
        tb = phi[i, j]
        if tb == 1:
            i = i - 1
            j = j - 1
        elif tb == 2:
            i = i - 1
        elif tb == 3:
            j = j - 1
        else:
            raise ValueError("Invalid traceback value")
        p.insert(0, i)
        q.insert(0, j)

    # Strip off the edges of the D matrix before returning
    D = D[1:r + 1, 1:c + 1]        

    return p, q, D



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


""" Gaussian/Viterbi functions"""

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


def viterbi_path(prior, transmat, obslik):    
    T = obslik.shape[1]    
    prior = prior.reshape(-1, 1)
    Q = len(prior)

    

    scaled = False
    delta = np.zeros((Q, T))    
    psi = np.zeros((Q, T), dtype=int)
    path = np.zeros(T, dtype=int)
    scale = np.ones(T)

    t = 0
        
    # # Added this
    # if prior.size != obslik.size:
    #     if prior.size < obslik.size:
    #         # Expand 'prior' to match the size of 'obslik'
    #         prior = np.resize(prior, obslik.shape)
    #     else:
    #         # Expand 'obslik' to match the size of 'prior'
    #         obslik = np.resize(obslik, prior.shape)
        
    
    delta[:, t] = prior.flatten() * obslik[:, t]        


    if scaled:
        delta[:, t] /= np.sum(delta[:, t])
        scale[t] = 1 / np.sum(delta[:, t])
    

    psi[:, t] = 0    
    for t in range(1, T):
        for j in range(Q):            
            delta[j, t] = np.max(delta[:, t - 1] * transmat[:, j])
            delta[j, t] *= obslik[j, t]

        if scaled:
            delta[:, t] /= np.sum(delta[:, t])
            scale[t] = 1 / np.sum(delta[:, t])

    p, path[T - 1] = np.max(delta[:, T - 1]), np.argmax(delta[:, T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]

    return path



def mixgauss_prob(data, means, covariances, weights):    
    # Create a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=2)  # Specify the number of components

    # Fit the GMM to your data
    gmm.fit(data)

    # Calculate the probabilities for each data point
    probs = gmm.predict_proba(data)
    # print('probs', probs)

# 'probs' now contains the conditional probabilities for each data point and each component.
    
    N = len(data)
    K = len(means)    
    

    covariances = [np.eye(5), np.eye(5)]
    # print('data', data.shape)
    # print('means', means.shape)
    # print(covariances)
    # covariances = means + covariances.reshape(1, -1)  # Reshape array2 to (1, 5)
    likelihood_matrix = np.zeros((N, K))    

    
    for i in range(N):
        for j in range(K):
            likelihood = weights[j] * multivariate_normal.pdf(data[i], mean=means[j], cov=covariances[j])
            likelihood_matrix[i][j] = likelihood

    return likelihood_matrix


""" Matrix functions """



def fill_trans_mat(trans_seed, notes):
    # Set up transition matrix
    N = trans_seed.shape[0]    
    trans = np.zeros((notes * (N - 1) + 1, notes * (N - 1) + 1))
    Non2 = int(np.ceil(N / 2 + 1)) # ADDED ONE!
    

    # Fill in the first and last parts of the big matrix with the
    # appropriate fragments of the seed
    trans[0:Non2, 0:Non2] = trans_seed[Non2:, Non2:]
    # trans[1:Non2, 1:Non2] = trans_seed[Non2:, Non2:] # Changed 0 to 1 here
    trans[-Non2:, -Non2:] = trans_seed[0:Non2, 0:Non2]

    # Fill in the middle parts of the big matrix with the whole seed
    for i in range(Non2, (notes - 1) * (N - 1) + 1 - Non2 + 1, N - 1):
        trans[i:i + N, i:i + N] = trans_seed

    return trans


def orio_simmx(M, D):
    # Calculate the similarities
    S = np.zeros((M.shape[1], D.shape[1]))

    # Square the elements of D and M
    D = D**2
    M = M**2

    nDc = np.sqrt(np.sum(D, axis=0))
    # Avoid division by zero
    nDc = nDc + (nDc == 0)

    # Evaluate one row at a time
    for r in range(M.shape[1]):
        S[r, :] = np.sqrt(np.dot(M[:, r], D)) / nDc

    return S

def simmx(A, B):
    """
    Calculate a similarity matrix between feature matrices A and B.

    Args:
        A (numpy.ndarray): The first feature matrix.
        B (numpy.ndarray, optional): The second feature matrix. If not provided, B will be set to A.

    Returns:
        numpy.ndarray: The similarity matrix between A and B.
    """
    if B is None:
        B = A

    # Match array sizes
    size_diff = len(A) - len(B)
    if size_diff > 0:
        A = A[:len(B)]

    EA = np.sqrt(np.sum(A**2, axis=0))
    EB = np.sqrt(np.sum(B**2, axis=0))

    # Avoid division by zero
    EA[EA == 0] = 1
    EB[EB == 0] = 1

    M = (A.T @ B) / (EA[:, np.newaxis] @ EB[np.newaxis, :])

    return M

