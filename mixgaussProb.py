# import numpy as np
# from scipy.linalg import cholesky
# from scipy.stats import multivariate_normal

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture



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

# # Example usage
# data = ...  # Your dataset
# means = ...  # List of means for each component
# covariances = ...  # List of covariances for each component
# weights = ...  # List of weights for each component

# likelihood_matrix = calculate_likelihood_matrix(data, means, covariances, weights)


# def mixgauss_prob(data, mu, Sigma, mixmat, unit_norm=0):

#     d, T = data.shape    
    
#     # # Square Sigma
#     # # Calculate the number of rows and columns for the square matrix
#     # num_rows, num_cols = Sigma.shape
#     # max_dim = max(num_rows, num_cols)

#     # # Pad with zeros if it's not square - ADDED THIS
#     # if num_rows < max_dim:
#     #     pad_rows = max_dim - num_rows
#     #     Sigma = np.vstack([Sigma, np.zeros((pad_rows, num_cols))])
#     # elif num_cols < max_dim:
#     #     pad_cols = max_dim - num_cols
#     #     Sigma = np.hstack([Sigma, np.zeros((num_rows, pad_cols))])
        
    

#     if mu.ndim == 1:
#         Q, M = 1, 1
#         mu = mu.reshape((d, 1))
#     elif mu.ndim == 2:
#         d, Q = mu.shape
#         M = 1
#     else:
#         d, Q, M = mu.shape
#     if Sigma.ndim == 0:
#         Sigma = np.array(Sigma)
#         mu = mu.reshape((d, Q * M))
#         if unit_norm:
#             D = 2 - 2 * np.dot(mu.T, data)
#         else:
#             D = np.sum((data[:, :, np.newaxis] - mu[:, np.newaxis, :]) ** 2, axis=0)
#         logB2 = -(d / 2) * np.log(2 * np.pi * Sigma) - (1 / (2 * Sigma)) * D
#         B2 = np.exp(logB2).reshape((Q, M, T))
#     elif Sigma.ndim == 2:
#         mu = mu.reshape((d, Q * M))
#         D = np.sum((data[:, :, np.newaxis] - mu[:, np.newaxis, :]) ** 2, axis=0)
#         logB2 = -(d / 2) * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(Sigma)[1] - 0.5 * D
#         B2 = np.exp(logB2).reshape((Q, M, T))
#     elif Sigma.ndim == 3:
#         B2 = np.zeros((Q, M, T))
#         for j in range(Q):
#             for k in range(M):                
#                 if np.all(np.linalg.eigvals(Sigma[j, :, :, k]) > 0):
#                     D = np.sum((data[:, :, np.newaxis] - mu[:, j, k][:, np.newaxis]) ** 2, axis=0)
#                     logB2 = -(d / 2) * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(Sigma[j, :, :, k])[1] - 0.5 * D
#                     B2[j, k, :] = np.exp(logB2)
#                 else:
#                     raise ValueError(f"Sigma[{j}, :, :, {k}] is not positive semidefinite.")
#     else:
#         B2 = np.zeros((Q, M, T))
#         for j in range(Q):
#             for k in range(M):
#                 if mixmat[j, k] > 0:
#                     mvn = multivariate_normal(mu[:, j, k], Sigma[:, :, j, k])
#                     B2[j, k, :] = mvn.pdf(data.T)

#     B = np.zeros((Q, T))      

#     # The first IF block was originally there, added the elif 
#     if Q < T & mixmat.size > 1:
#         for q in range(Q):
#             B[q, :] = mixmat[q, :] @ B2[q, :, :] # Original
#             # B[q, :] = mixmat[q].reshape(1, -1) @ B2[q, :, :]
#     elif Q < T & mixmat.size <= 1:
#         for q in range(Q):
#             B[q, :] = mixmat[q, :] @ B2[q, :, :] # Original
#             # B[q, :] = mixmat[q].reshape(1, -1) @ B2[q, :, :]

#     else:
#         for t in range(T):
#             B[:, t] = np.sum(mixmat * B2[:, :, t], axis=1)

#     return B

# # Example usage:
# # B, B2 = mixgauss_prob(data, mu, Sigma, mixmat, unit_norm)
