"""
alignmentUtils
==============


.. autosummary::
    :toctree: generated/

    dp
    fill_priormat_gauss
    gh
    flatTopGaussIdx
    g
    flatTopGaussian
    viterbi_path
    mixgauss_prob
    fill_trans_mat
    orio_simmx
    simmx
    maptimes
    calculate_f0_est

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

from scipy.signal import gaussian
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from scipy.optimize import linear_sum_assignment

__all__ = [
    "dp",
    "fill_priormat_gauss",
    "gh",
    "flatTopGaussIdx",
    "g",
    "flatTopGaussian",
    "viterbi_path",
    "mixgauss_prob",
    "fill_trans_mat",
    "orio_simmx",
    "simmx",
    "maptimes",
    "calculate_f0_est"
]

def dp(M):
    # neg_M = -M  # Convert to minimization problem
    # row_ind, col_ind = linear_sum_assignment(neg_M)
    # return row_ind.tolist(), col_ind.tolist(), M[row_ind, col_ind]


    """
    Use dynamic programming to find a min-cost path through matrix M.
    Return state sequence in p,q
    """
    r, c = M.shape

    # Initialize cost matrix D
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



# Gaussian/Viterbi functions
def fill_priormat_gauss(Nobs, ons, offs, Nstates):
    """
    Creates a prior matrix based on the DTW alignment (supplied by the input
    variables ons and offs. A rectangular window with half a Gaussian on
    each side over the onsets and offsets estimated by the DTW alignment.
    
    :params Nobs: Number of observations
    :params ons: Vector of onset times predicted by DTW alignment
    :params offs: Vector of offset times predicted by DTW alignment
    :params Nstates: Number of states in the hidden Markov model

    :return prior: Prior matrix based on DTW alignment
    """
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
    """
    Get an element that is frac fraction of the way between v1(i1) and
    v2(i2), but check bounds on both vectors.  Frac of 0 returns v1(i1), 
    frac of 1 returns v2(i2), frac of 1/2 (the default) returns half way 
    between them.
    """
    x1 = g(v1, i1, domain)
    x2 = g(v2, i2, domain)
    return int(frac * x1 + (1 - frac) * x2)
 

def flatTopGaussIdx(x, b1, bi1, t1, ti1, t2, ti2, b2, bi2):
    """
    Create a window function that is zeros, going up to 1s with the left
    half of a gaussian, then ones, then going back down to zeros with
    the right half of another gaussian.  b1(bi1) is the x coordinate 2
    stddevs out from the mean, which is at t1(ti1).  t2(ti2) is the x
    coordinate of the mean of the second gaussian and b2(bi2) is 2
    stddevs out from that.  The points should be in that order.  Vectors
    are indexed intelligently, so you don't have to worry about
    overflows or underflows.  X is the set of points over which this is
    to be calculated.
    """
    b1 = g(b1, bi1, x)
    t1 = g(t1, ti1, x)
    t2 = g(t2, ti2, x)
    b2 = g(b2, bi2, x)
    return flatTopGaussian(x, b1, t1, t2, b2)


def g(vec, idx, domain):
    """
    Get an element from vec, checking bounds.  Domain is the set of points
    that vec is a subset of.
    """
    if idx < 1:
        return domain[0]
    elif idx > len(vec):
        return domain[-1]
    else:
        return vec[idx - 1]


def flatTopGaussian(x, b1, t1, t2, b2):
    """
    Create a window function that is zeros, going up to 1s with the left 
    half of a gaussian, then ones, then going back down to zeros with the 
    right half of another gaussian.  b1 is the x coordinate 2 stddevs out 
    from the mean, which is at t1.  t2 is the x coordinate of the mean of 
    the second gaussian and b2 is 2 stddevs out from that.  The points 
    should be in that order.  X is the set of points over which this is 
    to be calculated.
    """
    if any([b1, t1, t2]) > any([t1, t2, b2]):
        print('Endpoints are not in order: ', b1, t1, t2, b2)
    


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


def viterbi_path(prior, transmat, obslik):
    """
    VITERBI Find the most-probable (Viterbi) path through the HMM state trellis.
    path = viterbi(prior, transmat, obslik)

    
    :param prior(i): Pr(Q(1) = i)
    :param transmat(i,j): Pr(Q(t+1)=j | Q(t)=i)
    :param obslik(i,t): Pr(y(t) | Q(t)=i)

    :returns:
        - path(t): q(t), where q1 ... qT is the argmax of the above expression.
        - delta(j,t) = prob. of the best sequence of length t-1 and then going to state j, and O(1:t)
        - psi(j,t) = the best predecessor state, given that we ended up in state j at t
    """ 
    T = obslik.shape[1]    
    prior = prior.reshape(-1, 1)
    Q = len(prior)

    scaled = False
    delta = np.zeros((Q, T))    
    psi = np.zeros((Q, T), dtype=int)
    path = np.zeros(T, dtype=int)
    scale = np.ones(T)

    t = 0
        
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
    """
    Notation: Y is observation, M is mixture component, and both may be conditioned on Q.
    If Q does not exist, ignore references to Q=j below.
    Alternatively, you may ignore M if this is a conditional Gaussian.
    
    
    :param data(:,t): t'th observation vector     
    :param mu(:,k): E[Y(t) | M(t)=k] 
        or mu(:,j,k) = E[Y(t) | Q(t)=j, M(t)=k]    
    :param Sigma(:,:,j,k): Cov[Y(t) | Q(t)=j, M(t)=k]
        or there are various faster, special cases:
        - Sigma() - scalar, spherical covariance independent of M,Q.
        - Sigma(:,:) diag or full, tied params independent of M,Q. 
        - Sigma(:,:,j) tied params independent of M. 
    
    :param mixmat(k): Pr(M(t)=k) = prior
        or mixmat(j,k) = Pr(M(t)=k | Q(t)=j) 
        Not needed if M is not defined.
    
    :param unit_norm: - optional; if 1, means data(:,i) AND mu(:,i) each have unit norm (slightly faster)
    
    :returns:
        - B(t) = Pr(y(t)) ||
        - B(i,t) = Pr(y(t) | Q(t)=i) 
        - B2(i,k,t) = Pr(y(t) | Q(t)=i, M(t)=k) 
    
    If the number of mixture components differs depending on Q, just set the trailing
    entries of mixmat to 0, e.g., 2 components if Q=1, 3 components if Q=2,
    then set mixmat(1,3)=0. In this case, B2(1,3,:)=1.0.
    """

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
    likelihood_matrix = np.zeros((N, K))    

    
    for i in range(N):
        for j in range(K):
            likelihood = weights[j] * multivariate_normal.pdf(data[i], mean=means[j], cov=covariances[j])
            likelihood_matrix[i][j] = likelihood

    return likelihood_matrix


# Matrix functions
def fill_trans_mat(trans_seed, notes):
    """
    Makes a transition matrix from a seed transition matrix.  The seed
    matrix is composed of the states: steady state, transient, silence,
    transient, steady state, but the full transition matrix starts and
    ends with silence, so the seed with be chopped up on the ends.
    Notes is the number of times to repeat the seed.  Transseed's first
    and last states should be equivalent, as they will be overlapped
    with each other.
    
    :param transseed: Transition matrix seed.
    :param notes: Number of notes being aligned.
    
    :return trans: Transition matrix
    """

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
    """
    Calculate an Orio&Schwartz-style (Peak Structure Distance) similarity matrix

    Parameters:
    - M: Binary mask where each column corresponds to a row in the output matrix S
    - D: Regular spectrogram, where columns of S correspond to columns of D

    Returns:
    - S: Similarity matrix
    """
    # Convert to NumPy arrays if input is DataFrame
    M = M.values if isinstance(M, pd.DataFrame) else M
    D = D.values if isinstance(D, pd.DataFrame) else D

    # Ensure compatibility for matrix multiplication
    
    # if M.shape[1] != D.shape[0]:
    #     M = M.T  # Transpose M if the number of columns in M does not match the number of rows in D

    
    # Calculate the similarities
    S = np.zeros((M.shape[1], D.shape[1]))



    # This way is slow
    # for r in range(M.shape[1]):
    #     for c in range(D.shape[1]):
    #         nDc = np.linalg.norm(D[:, c])
    #         nDc = nDc + (nDc == 0)
    #         S[r, c] = np.linalg.norm(D[:, c] * M[:, r]) / nDc

    # Doing it one row at a time is faster
    D = D**2
    M = M**2

    nDc = np.sqrt(np.sum(D, axis=0))
    # avoid div 0's
    nDc = nDc + (nDc == 0)

    # Evaluate one row at a time
    for r in range(M.shape[1]):
        S[r, :] = np.sqrt(M[:, r] @ D) / nDc

    return S


def simmx(A, B):
    """
    Calculate a similarity matrix between feature matrices A and B.

    :param A: The first feature matrix.
    :param B: The second feature matrix. If not provided, B will be set to A.

    :return: The similarity matrix between A and B.
    """
    A = A.values if isinstance(A, pd.DataFrame) else A
    B = B.values if isinstance(B, pd.DataFrame) else B
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


def maptimes(t, intime, outtime):    
    """
    Map the times in t according to the mapping that each point
    in intime corresponds to that value in outtime.

    Parameters:
    - t: 1D numpy array, input times
    - intime: 1D numpy array, input time points
    - outtime: 1D numpy array, output time points

    Returns:
    - u: 2D numpy array, mapped times
    """

    tr, tc = t.shape
    t = t.flatten()    
    nt = len(t)
    nr = len(intime)
    
    # Decidedly faster than outer-product-array way
    u = t.flatten()
    for i in range(nt):
        idx = np.min([np.argmax(intime > t[i]), len(outtime) - 1])
        u[i] = outtime[idx]        
    
    u = np.reshape(u, (tr, tc))
    return u
    
    # # This is the new way                
    # tr, tc = t.shape
    # t = t.reshape(1, -1)  # make into a row
    # nt = len(t)
    # nr = len(intime)
    
    # # Decidedly faster than outer-product-array way
    # u = t.flatten()
    # for i in range(nt):
    #     # Find the index in intime where the value is greater than t[i]
    #     idx = np.argmax(intime > t[0, i])
        
    #     # Ensure that idx is within the bounds of outtime
    #     idx = min(idx, len(outtime) - 1)        
    #     u[i] = outtime[idx]        
    
    # u = np.reshape(u, (tr, tc))  
    
        
    # return u

    

def calculate_f0_est(filename, hop_length, win_ms, tsr):    
            
    y, sr = librosa.load(filename, sr=tsr)

    # Calculate the maximum absolute amplitude
    max_amplitude = np.max(np.abs(y))

    # Normalize the audio signal
    normalized_y = y / max_amplitude

    # Compute STFT
    stft = librosa.stft(normalized_y, n_fft=win_ms, hop_length=hop_length)

    # Compute magnitude and phase of STFT
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Compute instantaneous frequency
    # Instantaneous frequency = Δ(phase) / Δ(time)
    # Compute phase difference between consecutive frames
    delta_phase = np.diff(phase, axis=1)
    # Compute time difference between consecutive frames
    delta_time = hop_length / tsr
    # Compute instantaneous frequency
    instantaneous_freq = np.diff(phase, axis=1) / delta_time

    # Estimate f0 by finding the dominant frequency bin at each time frame
    f0 = np.argmax(magnitude, axis=0) * tsr / win_ms

    # Estimate power at each time frame (sum of magnitude squared)
    power = np.sum(magnitude**2, axis=0)

    # Now you have f0 and power estimates for each time frame
    # Time axis
    time = np.arange(len(instantaneous_freq[0])) * hop_length / sr
    new_time_value = time[-1] + hop_length / sr
    time = np.append(time, new_time_value)
    
    # # Plot Instantaneous Frequency
    # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 1)
    # plt.imshow(instantaneous_freq, aspect='auto', origin='lower', cmap='viridis')
    # plt.colorbar(label='Instantaneous Frequency (Hz)')
    # plt.title('Instantaneous Frequency')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency Bin')

    # # Plot f0 Estimate
    # plt.subplot(3, 1, 2)
    # plt.plot(time, f0)
    # plt.title('Fundamental Frequency (f0) Estimate')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')

    # # Plot Power Estimate
    # plt.subplot(3, 1, 3)
    # plt.plot(time, power)
    # plt.title('Power Estimate')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Power')
    # plt.tight_layout()
    # plt.show()

    return f0, power


def f0_est_weighted_sum(x, f, f0i):
    # Set default values if not provided    
    fMax = 5000    
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


def f0_est_weighted_sum_spec(F, D, noteStart_s, noteEnd_s, f0i, tsr, useIf=True):
    # Use f0_est_weighted_sum on one note using spectrogram or IF features
    print(noteStart_s)
    print(noteEnd_s)
    
    win_s = 0.064
    nIter = 10
    
    win = int(win_s * tsr)
    hop = int(win / 8)
    
    inds = np.arange(round(noteStart_s * tsr / hop), round(noteEnd_s * tsr / hop) + 1)

    x = np.abs(D[inds]) ** (1/6)
    f = (np.arange(win // 2 + 1) * tsr) / win
    
    if useIf:
        xf = F[:, inds]
    else:
        xf = np.tile(f[:, np.newaxis], (1, x.shape[1]))
    
    f0 = f0_est_weighted_sum(x, xf, f0i)
    
    for _ in range(nIter):
        f0 = f0_est_weighted_sum(x, xf, f0)
    
    _, p, partials = f0_est_weighted_sum(x**6, xf, f0, 22050)
    
    M = partials[0]
    
    for i in range(1, len(partials)):
        M += partials[i]
    
    # Calculate time values
    t = np.arange(len(inds)) * hop / tsr    
    return f0, p, t, M, xf