import numpy as np

def genPolyTrans(selfWeight=5, skip2Weight=1, skip1Weight=1):
    # Initialize voices
    voices = []
    for i in range(1, 7):
        idx = 1
        voice_i = []
        for _ in range(3**i):
            voice_i.append(np.arange(1, i + 1))
            idx += 1
        voices.append(voice_i)

    # Initialize transition matrix
    trans = []
    for t in range(6):
        trans_t = np.zeros((len(voices[t]), len(voices[t])))
        for i in range(1, len(trans_t) + 1):
            for j in range(i, len(trans_t[0]) + 1):
                if np.sum(voices[t][i - 1] == voices[t][j - 1]) >= len(voices[t][j - 1]) - 1:
                    stateChange = max(voices[t][j - 1] - voices[t][i - 1])
                    if stateChange == 2:
                        trans_t[i - 1, j - 1] = skip2Weight
                    elif stateChange == 1:
                        trans_t[i - 1, j - 1] = skip1Weight
                    elif stateChange == 0:
                        trans_t[i - 1, j - 1] = selfWeight

        # Normalize
        trans_t = trans_t / np.sum(trans_t, axis=1)[:, np.newaxis]
        trans.append(trans_t)

    return voices, trans
