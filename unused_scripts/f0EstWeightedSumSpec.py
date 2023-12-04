# NEEDS TESTING
import numpy as np
import librosa

from f0EstWeightedSum import f0_est_weighted_sum

def f0_est_weighted_sum_spec(fileName, noteStart_s, noteEnd_s, f0i, useIf=True):
    # Use f0_est_weighted_sum on one note using spectrogram or IF features
    
    # Load audio file
    s, fs = librosa.load(fileName, sr=None)
    
    win_s = 0.064
    nIter = 10
    
    win = int(win_s * fs)
    hop = int(win / 8)
        
    F, D = librosa.reassigned_spectrogram(y=s, sr=fs, win_length=win, hop_length=hop)

    
    
    inds = np.arange(round(noteStart_s * fs / hop), round(noteEnd_s * fs / hop) + 1)
    
    x = np.abs(D[:, inds]) ** (1/6)
    f = (np.arange(win // 2 + 1) * fs) / win
    
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
    t = np.arange(len(inds)) * hop / fs
    
    return f0, p, t, M, xf


# # Example usage:
# f0, p, t, M, xf = f0_est_weighted_sum_spec("./audio_files/exampleOneNote.wav", 1, 32, 22, useIf=True)
