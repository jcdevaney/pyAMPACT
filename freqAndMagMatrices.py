import librosa
import numpy as np
import matplotlib.pyplot as plt

def freq_and_mag_matrices(audiofile, sr):
        
    y, sr = librosa.load(audiofile)

    # Calculate the Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)

    # Get the frequencies corresponding to each row of the STFT matrix
    freq_mat = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0])

    # Extract the magnitude and phase information
    mag_mat = np.abs(D)    

    return freq_mat, mag_mat    

    

    
    
