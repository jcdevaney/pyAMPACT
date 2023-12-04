import numpy as np


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
