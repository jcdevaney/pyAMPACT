"""
alignment
==============


.. autosummary::
    :toctree: generated/

    run_alignment    
    runDTWAlignment
    align_midi_wav
    alignment_visualiser
    ifgram
    get_ons_offs            


"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd

import sys
import os
sys.path.append(os.pardir)

from scipy.signal import spectrogram

from pyampact.performance import estimate_perceptual_parameters
from pyampact.alignmentUtils import orio_simmx, simmx, dp, maptimes

__all__ = [
    "run_alignment",
    "runDTWAlignment",
    "align_midi_wav",
    "alignment_visualiser",
    "ifgram",
    "get_ons_offs"
]


    
def run_alignment(y, original_sr, piece, means, covars, width=3, target_sr=4000, nharm=3, win_ms=100, hop=32):
    """    
    Calls the DTW alignment function and refines the results with the HMM
    alignment algorithm, with both a basic and modified state spaces (based on the lyrics).
    This function returns the results of both the state spaces as well as the YIN analysis
    of the specified audio file.

    :param y: Audio time series
    :param original_sr: original sample rate of audio
    :param piece: Score instance of symbolic data
    :param means: Mean values for each state.
    :param covars: Covariance values for each state.
    :param width: Width parameter (you need to specify this value).
    :param target_sr: Target sample rate (you need to specify this value).    
    :param nharm: Number of harmonics (you need to specify this value).
    :param win_ms: Window size in milliseconds (you need to specify this value).

    # :returns: 
    #     - align:
    #     - dtw:
    #     - res.on list of DTW predicted onset times in seconds
    #     - res.off list of DTW predicted offset times in seconds        
    #     - spec: Spectrogram of the audio file.        
    """
        

    # Normalize audio file
    y = y / np.sqrt(np.mean(y ** 2)) * 0.6
    

    # Run DTW alignment
    align, spec, dtw, newNmat = runDTWAlignment(
        y, original_sr, piece, 0.050, width, target_sr, nharm, win_ms)
        
               
    return align, dtw, spec, newNmat




def runDTWAlignment(y, original_sr, piece, tres, width, target_sr, nharm, win_ms):    
    """
    Perform a dynamic time warping alignment between specified audio and MIDI files.

    Returns a matrix with the aligned onset and offset times (with corresponding MIDI
    note numbers) and a spectrogram of the audio.

    :param y: Audio time series of audio.
    :param original_sr: original sample rate of audio
    :param piece: Score instance of symbolic data
    :param tres: Time resolution for MIDI to spectrum information conversion.
    :param width: Width parameter (you need to specify this value).    
    :param target_sr: Target sample rate (you need to specify this value).    
    :param nharm: Number of harmonics (you need to specify this value).
    :param win_ms: Window size in milliseconds (you need to specify this value).
    

    :returns:
        - align: dynamic time warping MIDI-audio alignment structure
            - align.on: onset times
            - align.off: offset times
            - align.midiNote: MIDI note numbers
        - spec: spectrogram
        - dtw: dict of dynamic time warping returns
            - M: map s.t. M(:,m).            
            - MA/RA [p,q]: path from DP
            - S: similarity matrix
            - D: spectrogram 
            - notemask: midi-note-derived mask
            - pianoroll: midi-note-derived pianoroll 
    """
    
    
    m, p, q, S, D, M, N = align_midi_wav(
        piece, WF=y, sr=original_sr, TH=tres, ST=1, width=width, tsr=target_sr, nhar=nharm, wms=win_ms)
    
    dtw = {
        'M': m,
        'MA': p,
        'RA': q,
        'S': S,
        'D': D,
        'notemask': M,
        'pianoroll': N
    }
    
    spec = dtw['D']

    unfiltered_nmat = piece.nmats() 
    
    
    nmat = {}
    # Iterate through each key-value pair (dataframe) in the nmat dictionary
    for key, df in unfiltered_nmat.items():
        # Filter out rows where MIDI column is not equal to -1.0
        filtered_df = df[df['MIDI'] != -1.0]
        # Store the filtered dataframe in the filtered_nmat dictionary with the same key
        nmat[key] = filtered_df
          
    
    align = {
        'nmat': nmat.copy(),  # Create an empty 2D array with 7 columns for nmat
        'on': np.empty(0),         # Create an empty 1D array
        'off': np.empty(0),        # Create an empty 1D array
        'midiNote': np.empty(0)    # Create an empty 1D array
    }
    
    tres = 0.025
    dtw['MA'] = np.array(dtw['MA']) * tres
    dtw['RA'] = np.array(dtw['RA']) * tres
    


    # loop through voices
    onset_sec = []
    offset_sec = []
    midi_notes = []

    
    for key, df in nmat.items():          
        onset_sec = df['ONSET_SEC'].values
        offset_sec = df['OFFSET_SEC'].values
        midi_notes = df['MIDI'].values
        
        # combined_slice = np.column_stack((np.concatenate(onset_sec), np.concatenate(offset_sec)))
        combined_slice = [[on, off] for on, off in zip(onset_sec, offset_sec)]
        combined_slice = np.array(combined_slice)
        
        
    
    
        # Reversed RA and MA
        x = maptimes(combined_slice, dtw['MA'], dtw['RA'])  
        # print('maptimes return, line 183 alignment.py', x)   

        # Assign 'on', 'off', and 'midiNote' values from nmat
        align['on'] = np.append(align['on'], x[:,0])
        align['off'] = np.append(align['off'], x[:,1])
        align['midiNote'] = np.append(align['midiNote'], midi_notes)
        spec = D # from align_midi_wav

        # df.loc[:,'ONSET_SEC'] = x[:,0]
        # df.loc[:,'OFFSET_SEC'] = x[:,1]
        df.at[df.index[0], 'ONSET_SEC'] = 0 # Set first value to 0 always
    
    
    return align, spec, dtw, nmat



def align_midi_wav(piece, WF, sr, TH, ST, width, tsr, nhar, wms):    
    """
    Align a midi file to a wav file using the "peak structure
    distance" of Orio et al. that use the MIDI notes to build 
    a mask that is compared against harmonics in the audio.
        
    :param MF: Score instance of symbolic data
    :param WF: Audio time series of file
    :param TH: is the time step resolution (default 0.050).
    :param ST: is the similarity type: 0 (default) is triangle inequality;

    :returns:
        - m: Is the map s.t. M(:,m).            
        - [p,q]: Are the path from DP.
        - S: The similarity matrix.
        - D: Is the spectrogram. 
        - M: Is the midi-note-derived mask.
        - N: Is Orio-style "peak structure distance".    
    """
    
    
    pianoRoll = piece.pianoRoll()      

    # Construct N
    sampled_grid = []
    for row in pianoRoll:
        sampled_grid.append(row)
    
    N = np.array(sampled_grid)        

        
    # Calculate spectrogram
    
    fft_len = int(2**np.round(np.log(wms/1000*tsr)/np.log(2))) # Good, 512
    
    ovlp = round(fft_len - TH*tsr);    
    
    y = librosa.resample(WF, orig_sr=sr, target_sr=tsr)
    # Generate a sample signal (replace this with your own signal)
    
    # srgcd = math.gcd(tsr, sr)
    # dr = librosa.resample(y,tsr/srgcd,sr/srgcd)
    # dr = resample(y,tsr/srgcd,sr/srgcd);
    
    
    # Compute spectrogram    
    f, t, Sxx = spectrogram(y, fs=tsr, nperseg=fft_len, noverlap=ovlp, window='hann')
    # Access the magnitude spectrogram (D)
    D = np.abs(Sxx)
    
    # Plot spectrogram
    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(D), shading='auto', vmax=np.max(10 * np.log10(D)))  # Use vmax to set the color scale
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram')    
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    # plt.show()
        

    
    # First mask declaration here follows the MATLAB params, but not sure
    # these are necessary at this point.
    # mask = piece.mask(wms, tsr, nhar, width, bpm=60, aFreq=440,
    #                   base_note=0, tuning_factor=1, obs=20)            
    M = piece.mask(sample_rate=tsr, num_harmonics=nhar, width=width, winms=wms)        
    # Save M to a CSV file
    # np.savetxt('./output_files/output.csv', M, delimiter=',')
    # M = M.astype(np.int16)
    
    

    # Calculate the peak-structure-distance similarity matrix    
    print('M/D shapes, line 277 alignment.py')
    print(M.shape)
    print(D.shape)    
    if ST == 0:
        S = orio_simmx(M, D)
    else:
        S = simmx(M, D) # This works, but using orio_simmx
    
    
        
    # Ensure no NaNs (only going to happen with simmx)
    S[np.isnan(S)] = 0
   
    # Do the DP search
    p, q, D = dp(1 - S) 
    
   

    # Map indices into MIDI file that make it line up with the spectrogram
    # Not sure if this is working as all other params are questionable!
    m = np.zeros(D.shape[0], dtype=int)
    for i in range(D.shape[0]):
        if np.any(q == i):
            m[i] = p[np.min(np.where(q == i))]
        else:
            m[i] = 1    
    return m, p, q, S, D, M, N

  

def alignment_visualiser(spec, fig=1):    
    """    
    Plots a gross DTW alignment overlaid with the fine alignment
    resulting from the HMM aligner on the output of YIN.  Trace(1,:)
    is the list of states in the HMM, and trace(2,:) is the number of YIN
    frames for which that state is occupied.  Highlight is a list of 
    notes for which the steady state will be highlighted.
    
    :param mid: Midi file.
    :param spec: Spectrogram of audio file (from align_midi_wav). 
    
    :return: Visualized spectrogram            

    """

    # hop size between frames
    stft_hop = 0.023  # Adjusted from 0.025
    
    
    # Plot spectrogram

    # fig = plt.figure(fig)
    # plt.imshow(20 * np.log10(spec), aspect='auto', origin='lower', cmap='gray')
    # plt.title('Spectrogram with Aligned MIDI Notes Overlaid')    
    # plt.xlabel('Time (.05s)')
    # plt.ylabel('Midinote')    
    # plt.clim([plt.gci().get_clim()[0], plt.gci().get_clim()[1] - 50])
    # plt.colorbar()
    
    # # plt.show()
    


def ifgram(audiofile, tsr, win_ms):    
    # win_samps = int(tsr / win_ms) # low-res
    win_samps = 2048 # Placeholder for now, default
    y, sr = librosa.load(audiofile)
    
    
    freqs, times, mags = librosa.reassigned_spectrogram(y=y, sr=tsr,
                                                      n_fft=win_samps, reassign_frequencies=False)
  
    
    

    # Find the index of the maximum magnitude frequency bin for each time frame
    max_mag_index = np.argmax(mags, axis=0)
    
    # Extract the corresponding frequencies as f0 values
    f0_values = freqs[max_mag_index]
    

    # Calculate the Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)

    # Extract the magnitude and phase information
    mags_mat = np.abs(D)    

    

    # fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    # img = librosa.display.specshow(mags_db, x_axis="s", y_axis="linear",sr=tsr, hop_length=win_samps//4, ax=ax[0])
    # ax[0].set(title="Spectrogram", xlabel=None)
    # ax[0].label_outer()
    # ax[1].scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)
    # ax[1].set_title("Reassigned spectrogram")
    # fig.colorbar(img, ax=ax, format="%+2.f dB")
                  
    # plt.show()
    
    return freqs, times, mags, f0_values, mags_mat

def get_ons_offs(onsoffs):
    """
    Extracts a list of onset and offset from an inputted 
             3*N matrix of states and corresponding ending times 
             from AMPACT's HMM-based alignment algorithm
    
    :param onsoffs: A 3*N alignment matrix, the first row is a list of N states
        the second row is the time which the state ends, and the
        third row is the state index
    :returns: 
        - res.ons: List of onset times
        - res.offs: List of offset times
    """
    
    # Find indices where the first row is equal to 3
    stopping = np.where(onsoffs[0] == 3)[0]    
    # Calculate starting indices by subtracting 1 from stopping indices
    starting = stopping - 1

    res = {'ons': [], 'offs': []}     
    for i in range(len(starting)):
        res['ons'].append(onsoffs[1, starting[i]])
        res['offs'].append(onsoffs[1, stopping[i]])
    
    return res