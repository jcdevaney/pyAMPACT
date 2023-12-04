import numpy as np

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
