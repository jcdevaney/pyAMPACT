# COMPLETE/TESTED
import numpy as np
import librosa

def calculate_vibrato(note_vals, sr):
    L = len(note_vals)  # Length of signal
    Y = np.fft.fft(note_vals) / L  # Run FFT on normalized note vals
    w = np.arange(0, L) * sr / L  # Set FFT frequency grid

    vibrato_depth_tmp, noteVibratoPos = max(abs(Y)), np.argmax(abs(Y))  # Find the max value and its position
    vibrato_depth = vibrato_depth_tmp * 2  # Multiply the max by 2 to find depth (above and below zero)
    vibrato_rate = w[noteVibratoPos]  # Index into FFT frequency grid to find position in Hz

    return vibrato_depth, vibrato_rate

# # Example usage:
# note_vals, sr = librosa.load('./audio_files/exampleOneNote.wav', sr=4000)  # Load your audio file
# vibrato_depth, vibrato_rate = calculate_vibrato(note_vals, sr)
# print(f'Vibrato Depth: {vibrato_depth}')
# print(f'Vibrato Rate (Hz): {vibrato_rate}')