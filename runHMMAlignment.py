from filltransmat import fill_trans_mat
from viterbiPath import viterbi_path
from mixgaussProb import mixgauss_prob
from fillpriormat_gauss import fill_priormat_gauss

from hmmlearn import hmm
import numpy as np
import pandas as pd
import librosa

import os
import sys
sys.path.append(os.getcwd())


def run_HMM_alignment(notenum, means, covars, align, yinres, sr, learnparams=False):

    if not learnparams:
        shift = 0

    # Create vectors of onsets and offsets times from DTW alignment
    align['on'] = np.array(align['on'])
    align['off'] = np.array(align['off'])
    ons = np.floor(align['on'] * sr / 32).astype(int)
    offs = np.floor(align['off'] * sr / 32).astype(int)

    # Create observation matrix
    # obs = np.zeros((3, offs[notenum] + 50))
    obs = np.zeros((3, yinres['ap'].size))

    # - 1 to account for 0 index of Python
    obs[0, :] = np.sqrt(yinres['ap'][:offs[notenum - 1] + 50])
    obs[1, :] = np.sqrt(yinres['pwr'][:offs[notenum - 1] + 50])
    # obs[2, :] = 69 + 12 * yinres['f0'][:offs[notenum - 1] + 50]  # Convert octave to MIDI note

    yinres['f0'] = np.ceil(yinres['f0'])
    midiPitches = librosa.hz_to_midi(yinres['f0'])
    # Convert octave to MIDI note
    obs[2, :] = midiPitches[:offs[notenum - 1] + 50]

    # Replace any NaNs in the observation matrix with zeros
    obs[np.isnan(obs)] = 0

    # Refine the list of onsets and offsets according to the number of notes
    prior_ons = ons[:notenum]  # Ignore added 0 placeholder
    prior_offs = offs[:notenum]
    notes = len(prior_ons)  # Normalize

    # Define states: silence, trans, steady state
    # Rows: aperiodicity, power
    state_ord_seed = [1, 2, 3, 2, 1]
    # state_ord = np.tile(state_ord_seed[:-1], notes) + [state_ord_seed[-1]] # This is 21 size
    state_ord = np.concatenate([np.tile(
        state_ord_seed[:-1], notes), [state_ord_seed[-1]]])  # This gives both 20 size

    # Use state_ord to expand means and covars for each appearance
    midi_notes = np.tile(align['midiNote'][:notenum], len(state_ord_seed) - 1)
    midi_notes = np.append(midi_notes, align['midiNote'][notenum - 1])
    # Changed state_ord - 1
    means_full = np.vstack((means[:, state_ord - 1], midi_notes))
    covars = covars.reshape(3, 2, 2)
    covars[0, 1, 0] = 100
    covars[1, 1, 0] = 5
    covars[2, 1, 0] = 1
    covars_full = covars[state_ord - 1, :, :]  # deleted one :, to make 2-D

    mixmat = np.ones(len(state_ord))

    # Transition matrix seed
    # {steady state, transient, silence, transient, steady state}
    # Original, commented out 4th index to see results...
    trans_seed = np.zeros((5, 5))
    # trans_seed = np.zeros((4, 4))
    trans_seed[0, 0] = 0.99
    trans_seed[1, 1] = 0.98
    trans_seed[2, 2] = 0.98
    trans_seed[3, 3] = 0.98
    trans_seed[4, 4] = 0.99
    trans_seed[0, 1] = 0.0018
    trans_seed[0, 2] = 0.0007
    trans_seed[0, 3] = 0.0042
    trans_seed[0, 4] = 0.0033
    trans_seed[1, 2] = 0.0018
    trans_seed[1, 3] = 0.0102
    trans_seed[1, 4] = 0.0080
    trans_seed[2, 3] = 0.0112
    trans_seed[2, 4] = 0.0088
    trans_seed[3, 4] = 0.02

    # Call filltransmat to expand the transition matrix to the appropriate size
    trans = fill_trans_mat(trans_seed, notes)

    # Create starting state space matrix
    starting_state = np.zeros(4 * notes + 1)
    starting_state[0] = 1

    prior = fill_priormat_gauss(obs.shape[0], prior_ons, prior_offs, 5)

    if learnparams:
        # Use the fit function from the hmmlearn library to learn the HMM parameters
        model = hmm.GMMHMM(n_components=5, n_mix=1,
                           covariance_type='diag', n_iter=1)
        model.startprob_ = starting_state
        model.transmat_ = trans
        model.means_ = means_full.T
        model.covars_ = covars_full.T
        model.fit(obs.T)

    # like = mixgauss_prob(obs, means_full, covars_full, mixmat)

    # Use the Viterbi algorithm to find the most likely path
    # pr_like = prior * like
    # vpath = hmm.ViterbiHMM(starting_state, trans, pr_like)

    # Define the filename
    # pLikeData = "./placeholders/priorlike_oneNote_runHMM.csv"
    # pLikeData = "./placeholders/priorlike_threeNote_runHMM.csv"
    pLikeData = "./placeholders/priorlike_sixNote_runHMM.csv"

    # Read the data from the file
    dtype = {'index': str, 'value': float}
    pr_like = pd.read_csv(pLikeData, dtype=dtype,
                          sep='\s+', names=['index', 'value'])

    # Initialize an empty dictionary to store the data
    data_dict = {}

    # Open the text file for reading
    with open(pLikeData, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into two parts based on whitespace
            parts = line.split()
            # Extract the index from the first part and convert it to a tuple
            index = tuple(map(int, parts[0].strip('()').split(',')))
            # Parse the value from the second part
            value = float(parts[1])
            # Store the data in the dictionary with the index as the key
            data_dict[index] = value

    # Determine the shape of the numpy array based on the maximum index values
    num_rows = max(index[0] for index in data_dict.keys())
    num_cols = max(index[1] for index in data_dict.keys())

    # Initialize a numpy array with zeros
    pr_like = np.zeros((num_rows, num_cols))

    # Fill the numpy array with the values from the dictionary
    for index, value in data_dict.items():
        pr_like[index[0] - 1, index[1] - 1] = value

    vpath = viterbi_path(starting_state, trans, pr_like)
    # vpath = librosa.sequence.viterbi(prob=starting_state, transition=trans, pr_like)

    return vpath, starting_state, prior, trans, means_full, covars_full, mixmat, obs, state_ord
