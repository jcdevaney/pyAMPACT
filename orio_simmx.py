import numpy as np

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
