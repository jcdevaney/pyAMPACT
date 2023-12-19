import numpy as np

def notes2mask(N, L, SR, NH, BN, WS, TF):
    # How wide (in semitones) is the window around each harmonic?
    # (divide WS in two since it is applied both above and below ctr freq)
    widthsemifactor = 2 ** ((WS / 2) / 12)
    
    noprows = (L // 2) + 1
    # Convert N to ints
    N = N.astype(int)        
    M = np.zeros((noprows, N.shape[1]))            
    for nrow in range(BN - 1, BN - 1 + N.shape[0]):
        note = BN - 1 + nrow
        # MIDI note to Hz: MIDI 69 = 440 Hz
        freq = TF * (2 ** (note / 12)) * 440 / (2 ** (69 / 12))                
        if np.sum(N[nrow - BN + 1, :]) > 0:
            mcol = np.zeros(noprows)
            
            for harm in range(1, NH + 1):
                minbin = 1 + int(np.floor(harm * freq / widthsemifactor / SR * L))
                maxbin = 1 + int(np.ceil(harm * freq * widthsemifactor / SR * L))
                
                if minbin <= noprows:
                    maxbin = min(maxbin, noprows)
                    mcol[minbin-1:maxbin] = 1
            
            indices = np.where(N[nrow - BN + 1, :])
            for i in range(len(indices[0])):
                row_idx = np.where(mcol)[0]  # Extract the row indices
                col_idx = indices[0][i]  # Extract the column index
            M[row_idx, col_idx] = 1
            

            M[np.where(mcol), indices] = 1
                        
    
    return M
