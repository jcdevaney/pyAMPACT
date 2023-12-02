# COMPLETE NEEDS TESTING
import numpy as np

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
