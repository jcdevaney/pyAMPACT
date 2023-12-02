# NEEDS TESTING
import librosa

def resample( x, p, q, N, bta ):
    librosa(resample(x, p, q, N, bta))
    
# import librosa
# import numpy as np

# def resample(x, p, q, N=10, bta=5):
#     # Reduce p/q to lowest terms
#     gcd = np.gcd(p, q)
#     p, q = p // gcd, q // gcd

#     if p == 1 and q == 1:
#         return x, 1

#     pqmax = max(p, q)

#     if isinstance(N, list):  # Use input filter
#         L = len(N)
#         h = N
#     else:  # Design filter
#         if N > 0:
#             fc = 1 / (2 * pqmax)
#             L = 2 * N * pqmax + 1
#             h = p * librosa.filters.firls(L-1, [0, 2*fc, 2*fc, 1], [1, 1, 0, 0]) * librosa.filters.kaiser(L, beta=bta)
#         else:
#             L = p
#             h = np.ones(p)

#     Lhalf = (L - 1) / 2

#     if np.isscalar(x):
#         Lx = len(x)
#     else:
#         Lx = x.shape[0]

#     nz = int(q - (Lhalf % q))
#     z = np.zeros(nz)
#     h = np.concatenate((z, h))

#     Lhalf = Lhalf + nz
#     delay = int(np.floor(np.ceil(Lhalf) / q))

#     nz1 = 0
#     while (np.ceil((Lx - 1) * p + len(h) + nz1) / q - delay < np.ceil(Lx * p / q)):
#         nz1 = nz1 + 1

#     h = np.concatenate((h, np.zeros(nz1)))

#     y = librosa.sequence.upfirdn(h, x, up=p, down=q)

#     Ly = int(np.ceil(Lx * p / q))

#     if np.isscalar(x):
#         y = y[delay:delay + Ly]
#     else:
#         y = y[delay:delay + Ly, :]

#     h = h[nz:-nz1]  # Remove leading and trailing zeros from the filter
#     return y, h


# Example usage: COMPARE
# y, h = resample(x, p, q)
# y, h = librosa(resample(x, p, q, N, bta))
