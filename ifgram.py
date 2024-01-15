import numpy as np
import librosa
import matplotlib.pyplot as plt

def ifgram(X, N=256, W=None, H=None, SR=1):
    if W is None:
        W = N
    if H is None:
        H = W / 2

    s = len(X)
    X = X.reshape((1, -1))  # Ensure X is a 1-D row vector

    win = 0.5 * (1 - np.cos(np.arange(W) / W * 2 * np.pi))
    dwin = -np.pi / (W / SR) * np.sin(np.arange(W) / W * 2 * np.pi)

    norm = 2 / np.sum(win)

    nhops = 1 + int(np.floor((s - W) / H))

    F = np.zeros((1 + N // 2, nhops))
    D = np.zeros((1 + N // 2, nhops), dtype=complex)

    nmw1 = int(np.floor((N - W) / 2))
    nmw2 = N - W - nmw1

    ww = 2 * np.pi * np.arange(N) * SR / N

    for h in range(nhops):
        u = X[0, np.round((h - 1) * H + np.arange(W)).astype(int)]  # Fix here
        wu = win * u
        du = dwin * u

        if N > W:
            wu = np.concatenate((np.zeros(nmw1), wu, np.zeros(nmw2)))
            du = np.concatenate((np.zeros(nmw1), du, np.zeros(nmw2)))
        elif N < W:
            wu = wu[-nmw1:N - nmw1]
            du = du[-nmw1:N - nmw1]

        t1 = np.fft.fftshift(np.fft.fft(du))
        t2 = np.fft.fftshift(np.fft.fft(wu))
        D[:, h] = t2[:1 + N // 2] * norm

        t = t1 + 1j * (ww * t2)
        a, b = np.real(t2), np.imag(t2)
        da, db = np.real(t), np.imag(t)
        instf = (1 / (2 * np.pi)) * (a * db - b * da) / ((a * a + b * b) + (np.abs(t2) == 0))
        F[:, h] = instf[:1 + N // 2]

    return F, D

# Example usage:
y, sr = librosa.load('./test_files/example3note.wav')
F, D = ifgram(y)

# Plotting the spectrogram
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                         sr=sr, hop_length=len(y)//D.shape[1], x_axis='time', y_axis='linear')

plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# Plotting the reassigned spectrogram
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                         sr=sr, hop_length=len(y)//D.shape[1], x_axis='time', y_axis='linear')

plt.colorbar(format='%+2.0f dB')
plt.title('Reassigned Spectrogram')
plt.show()