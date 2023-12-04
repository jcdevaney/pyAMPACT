# PASS; matches MATLAB
import numpy as np

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
