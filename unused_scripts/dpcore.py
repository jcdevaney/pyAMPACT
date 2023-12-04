import numpy as np

def dpcore(S, C=None):
    def dp(S, C=None):
        rows, cols = S.shape
        D = np.zeros((rows, cols))
        P = np.zeros((rows, cols))
        
        if C is None:
            # Default cost matrix
            ncosts = 3
            costs = np.array([1.0, 1.0, 1.0])
            steps = np.array([[1, 1], [1, 0], [0, 1]])
        else:
            crows, ccols = C.shape
            if ccols != 3:
                print("Cost matrix must have 3 columns (i step, j step, cost factor)")
                return None, None
            ncosts = crows
            costs = C[:, 2]
            steps = C[:, :2].astype(int)

        v = 0
        tb = 1  # Value to use for (0, 0)

        for j in range(cols):
            for i in range(rows):
                d1 = S[i, j]
                for k in range(ncosts):
                    if i >= steps[k, 0] and j >= steps[k, 1]:
                        d2 = costs[k] * d1 + D[i - steps[k, 0], j - steps[k, 1]]
                        if d2 < v:
                            v = d2
                            tb = k + 1

                D[i, j] = v
                P[i, j] = tb
                v = float('inf')

        return D, P

    if S.ndim != 2:
        print("Input matrix S must be 2D")
        return None, None

    D, P = dp(S, C)

    return D, P

# Example usage:
S = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
D, P = dpcore(S)
