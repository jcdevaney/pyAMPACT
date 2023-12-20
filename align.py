import os
curr_dir = os.getcwd()

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd


from pytimbre.yin import yin
# from alignmentHelpers import select_states
from mido import MidiFile
from alignmentHelpers import midi2nmat
from audioHelpers import orio_simmx, simmx, dp

# HMM libraries
from audioHelpers import fill_trans_mat, viterbi_path, fill_priormat_gauss, mixgauss_prob
from hmmlearn import hmm

from symbolic import Score
    
def run_alignment(filename, midiname, num_notes, state_ord2, note_num, means, covars, learn_params, width, target_sr, nharm, win_ms):
    """    
    Calls the DTW alignment function and refines the results with the HMM
    alignment algorithm, with both a basic and modified state spaces (based on the lyrics).
    This function returns the results of both the state spaces as well as the YIN analysis
    of the specified audio file.

    Inputs:
    filename - name of the audio file
    midiname - name of the MIDI file
    num_notes - number of notes in the MIDI file to be aligned
    state_ord2 - vector of state sequence
    note_num - vector of note numbers corresponding to state sequence
    means - mean values for each state
    covars - covariance values for each state
    learn_params - flag as to whether to learn means and covars in the HMM
    width - width parameter (you need to specify this value)
    target_sr - target sample rate (you need to specify this value)
    nharm - number of harmonics (you need to specify this value)
    win_ms - window size in milliseconds (you need to specify this value)

    Outputs:
    allstate - ending times for each state
    selectstate - ending times for each state
    spec - spectrogram of the audio file
    yinres - structure of results of running the YIN algorithm on the audio signal indicated by the input variable filename
    """

    if learn_params is None:
        learn_params = 0

    # Refine state_ord2 to correspond to the number of states specified in num_notes
    note_num = np.array(note_num)    
    num_states = max(np.where(note_num <= num_notes)[0])    
    state_ord2 = state_ord2[:num_states] # Original    
    note_num = note_num[:num_states] # Original
    

    # Read audio file and perform DTW alignment and YIN analysis
    hop = 32
    audiofile, sr = librosa.load(filename, sr=4000)

    # Normalize audio file
    audiofile = audiofile / np.sqrt(np.mean(audiofile ** 2)) * 0.6

    align, yinres, spec, dtw = get_vals(
        filename, midiname, audiofile, sr, hop, width, target_sr, nharm, win_ms)
    
    
    audiofile = None  # Clear the audiofile variable


    
    # # CLEARED. TO BE FIXED LATER

    # # Run HMM alignment with the full state sequence
    # vpath, starting_state, prior, trans, means_full, covars_full, mixmat, obs, state_ord = run_HMM_alignment(
    #     num_notes, means, covars, align, yinres, sr, learn_params)

    # # Tally of the number of frames in each state
    # histvals = np.histogram(vpath, bins=np.arange(1, max(vpath) + 2))[0]

    # # Ending time of each state in seconds
    # cumsumvals = np.cumsum(histvals * hop / sr)

    # # cumsumvals2 = select_states(starting_state, prior, trans,
    # #                             means_full, covars_full, mixmat, obs, state_ord2, note_num, sr)
    # # vpath2 = select_states(starting_state, prior, trans,
    # #                        means_full, covars_full, mixmat, obs, state_ord2, note_num, sr)

    # # Create 3*N matrices of the alignments, where the first row is the current states,
    # # the second row is the time at which the state ends, and the third row is the state index,
    # # and N is the total number of states
    # allstate = np.vstack([state_ord, np.zeros_like(state_ord)])
    # allstate[1, :len(cumsumvals)] = cumsumvals

    

    
    # # selectstate construction, in progress   
    lengthOfNotes = note_num + 1    
    selectstate = np.empty((3, len(lengthOfNotes)))            
    interleaved = [val for pair in zip(align['on'], align['off']) for val in pair]
    interleaved = [val / 2 for val in interleaved]    
    selectstate[0, :] = state_ord2[:len(lengthOfNotes)]
    
    selectstate[1, :] = interleaved[:-1]
    selectstate[2, :] = note_num

    
    # Placeholder            
    # selectstate = np.array([[1.0000, 3.0000, 2.0000, 3.0000, 2.0000, 3.0000],
    #                         [0.9818, 4.1941, 4.1941, 4.8929, 4.9205, 6.6859],
    #                         [1.0000, 1.0000, 2.0000, 2.0000, 3.0000, 3.0000]])
    
    return selectstate, spec, yinres



def get_vals(filename, midi_file, audiofile, sr, hop, width, target_sr, nharm, win_ms):
    """    
      Gets values for DTW alignment and YIN analysis of specified audio 
      signal and MIDI file
        Inputs:
      filename
      midifile, audiofile, sr, hop
        Outputs: 
      res
        res.on list of DTW predicted onset times in seconds
        res.off list of DTW predicted offset times in seconds
      yinres (below are the two elements that are used)
        yinres.ap aperiodicty estimates for each frame
        yinres.pwr power estimates for each frame
    
    Dependencies:
      de Cheveigné, A. 2002. YIN MATLAB implementation Available from:
       http://audition.ens.fr/adc/sw/yin.zip
      Toiviainen, P. and T. Eerola. 2006. MIDI Toolbox. Available from:
       https://www.jyu.fi/hum/laitokset/musiikki/en/research/coe/materials...
              /miditoolbox/
        Automatic Music Performance Analysis and Analysis Toolkit (AMPACT) 
    http://www.ampact.org
    (c) copyright 2011 Johanna Devaney (j@devaney.ca), all rights reserved.
    """

    # Run DTW alignment
    res, spec, dtw = runDTWAlignment(
        filename, midi_file, 0.025, width, target_sr, nharm, win_ms)

    # Normalize audiofile
    audiofile = audiofile / np.sqrt(np.mean(audiofile ** 2))    
    
    piece = Score(midi_file)
    notes = piece.midiPitches()

   
    # Number of notes to align
    pitch_list = [];
    for note in notes['Piano']: # Hardcoded. Fix this?
        if note != -1: # Exclude rests
            pitch_list.append(note)
    # Define parameters for YIN analysis    
        

    P = {
        'thresh': 1,  # originally 0.01 in MATLAB, no difference?
        'sr': sr,
        'hop': hop,
        # Broadened range from 2 (added 2)
        'maxf0': np.max(librosa.midi_to_hz(np.array(pitch_list) + 4)),
        'minf0': np.min(librosa.midi_to_hz(np.array(pitch_list) - 1)),
    }        

    # Parameters
    #     ----------
    #     x : ndarray [shape=(L, )], real - valued
    #         Audio signal
    #     Fs : int
    #         Sampling frequency
    #     N : int
    #         Window size
    #     H : int
    #         Hop size
    #     F_min : float
    #         Minimal frequency
    #     F_max : float
    #         Maximal frequency
    #     threshold : float
    #         Threshold for cumulative mean normalized difference function
    #     verbose : bool
    #         Switch to activate/deactivate status bar

    f0, t, ap = yin(x=audiofile, Fs=P['sr'], N=win_ms, H=P['hop'], F_max=P['maxf0'], F_min=P['minf0'], threshold=P['thresh'])


    yinres = {
        'f0': f0, 
        'time': t,  
        'ap': ap  
    }    
    
    return res, yinres, spec, dtw



def runDTWAlignment(audiofile, midorig, tres, width, targetsr, nharm, winms):    
    """
    Perform a dynamic time warping alignment between specified audio and MIDI files.

    Returns a matrix with the aligned onset and offset times (with corresponding MIDI
    note numbers) and a spectrogram of the audio.

    Parameters:
    - sig: audio file
    - sr: sample rate
    - midorig: MIDI file
    - tres: time resolution for MIDI to spectrum information conversion
    - plot: boolean, whether to plot the spectrogram

    Returns:
    - align: dynamic time warping MIDI-audio alignment structure
        - align.on: onset times
        - align.off: offset times
        - align.midiNote: MIDI note numbers
    - spec: spectrogram


    Automatic Music Performance Analysis and Analysis Toolkit (AMPACT)
    http://www.ampact.org
    (c) copyright 2011 Johanna Devaney (j@devaney.ca), all rights reserved.
    """
    
    # midorig is the path string, not the file
    # midi_data = pretty_midi.PrettyMIDI(midorig)
    midi_data = MidiFile(midorig)
    # Initialize lists to store data
    on_times = []
    off_times = []
    midi_notes = []

    # Now done in alignMidiWav
    y, sr = librosa.load(audiofile)
    # Get the spectrogram of the audio
    # THIS may need revisiting, shortcut to creating spectrogram instead of via alignMidiWav etc.
    # The data array does show up differently vs MATLAB but this is likely fine.
    spec = librosa.feature.melspectrogram(y=y, sr=sr)

    # librosa.display.specshow(librosa.power_to_db(spec, ref=np.max), y_axis='mel', x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel Spectrogram')
    # plt.show()

    # INS
    #    MF is the name of the MIDI file, WF is the name of the wav file.
    #    TH is the time step resolution (default 0.050).
    #    ST is the similarity type: 0 (default) is triangle inequality;
    #       1 is Orio-style "peak structure distance".

    # OUTS
    #    m is the map s.t. M(:,m) \approxeq D
    #    [p,q] are the path from DP
    #    S is the similarity matrix.
    #    D is the spectrogram
    #    M is the midi-note-derived mask.
    #    N is note-mask (provided by CSV)
    m, p, q, S, D, M, N = align_midi_wav(
        MF=midorig, WF=audiofile, TH=tres, ST=0, width=width, tsr=targetsr, nhar=nharm, wms=winms)

    dtw = {
        'M': m,
        'MA': p,
        'RA': q,
        'S': S,
        'D': D,
        'notemask': M,
        'pianoroll': N
    }

    # THIS IS INCOMPLETE
    # The alignment needs to happen against the nmat values...
    nmat = midi2nmat(midorig)
    
    
    # # Iterate through the MIDI tracks
    # currentTime = 0
    # for track in midi_data.tracks:
    #     for msg in track:
    #         if msg.type == 'note_on':
    #             on_times.append(currentTime + msg.time)
    #             currentTime += msg.time
    #             # CATCH as no note_off msgs are in the example.mid
    #             off_times.append(currentTime + 5)
    #             midi_notes.append(msg.note)
    #         elif msg.type == 'note_off':
    #             off_times.append(msg.time)

    # off_times.pop(0)  # Remove first index as it is not a true off time

    # # Convert times to absolute times
    # cumulative_time = 0
    # on_times = [cumulative_time + time for time in on_times]
    # off_times = [cumulative_time + time for time in off_times]

    # Assuming you want data for the first instrument
    align = {
        'nmat': nmat,
        'on': nmat[:,2],
        'off': nmat[:,3],
        'midiNote': midi_notes
    }    

    return align, spec, dtw




def align_midi_wav(MF, WF, TH, ST, width, tsr, nhar, wms):    
    """
    Align a midi file to a wav file using the "peak structure
    distance" of Orio et al. that use the MIDI notes to build 
    a mask that is compared against harmonics in the audio.
    MF is the name of the MIDI file, WF is the name of the wav file.
    TH is the time step resolution (default 0.050).
    ST is the similarity type: 0 (default) is triangle inequality;
        1 is Orio-style "peak structure distance".
        S is the similarity matrix.
        [p,q] are the path from DP
        D is the spectrogram, M is the midi-note-derived mask.
        m is the map s.t. M(:,m) \approxeq D
    2008-03-20 Dan Ellis dpwe@ee.columbia.edu
    """

    piece = Score(MF)
    pianoRoll = piece.pianoRoll()      

    # Construct N
    sampled_grid = []
    for row in pianoRoll:
        sampled_grid.append(row)

    # Calculate spectrogram
    y, sr = librosa.load(WF)
    fft_len = int(2**np.round(np.log(wms/1000*tsr)/np.log(2)))
    hop_length = int((TH * tsr * 1000))
    D = librosa.feature.melspectrogram(
        y=y, sr=tsr, n_fft=fft_len, hop_length=hop_length, window='hamming')

    # basenote = 21  # Corresponds to MIDI note A0
    # tuning = 1.0
    # M = notes2mask(N, fft_len, tsr, nhar, basenote, width, tuning)  # You need to implement notes2mask function

    N = np.array(sampled_grid)
    # N = N.astype(np.int16)

    mask = piece.mask(wms, tsr, nhar, width, bpm=60, aFreq=440,
                      base_note=0, tuning_factor=1, obs=20)

    # INT or FLOAT?
    M = np.array(mask)  # provided as CSV
    M = M.astype(np.int16)

    # Calculate the peak-structure-distance similarity matrix
    if ST == 1:
        S = orio_simmx(M, D)
    else:
        S = simmx(M, D)

    # Threshold for matching a "silence" state 0..1
    # silstatethresh = 0.4;
    # S[onsetcols-1, :] = silstatethresh * np.max(S)

    # Ensure no NaNs (only going to happen with simmx)
    S[np.isnan(S)] = 0

    # Do the DP search
    p, q, D = dp(1 - S)  # You need to implement dpfast function    

    # Map indices into MIDI file that make it line up with the spectrogram
    m = np.zeros(D.shape[1], dtype=int)
    for i in range(D.shape[1]):
        if np.any(q == i):
            m[i] = p[np.min(np.where(q == i))]
        else:
            m[i] = 1

    return m, p, q, S, D, M, N


    
def alignment_visualizer(trace, mid, spec, fig=1):    
    """    
    Plots a gross DTW alignment overlaid with the fine alignment
    resulting from the HMM aligner on the output of YIN.  Trace(1,:)
    is the list of states in the HMM, and trace(2,:) is the number of YIN
    frames for which that state is occupied.  Highlight is a list of 
    notes for which the steady state will be highlighted.
    
    Inputs:
      trace - 3-D matrix of a list of states (trace(1,:)), the times   
             they end at (trace(2,:)), and the state indices (trace(3,:))
      mid - midi file
      spec - spectogram of audio file (from alignmidiwav.m)            
    """

    if fig is None:
        fig = 1

    # # COMMENT OUT FOR > 1 NOTES
    # if trace[1, -1] == 0:
    #     trace = trace[:, :-1]

    # if trace[1, -2] == 0:
    #     trace[1, -2] = trace[1, -3]

    # hop size between frames
    stft_hop = 0.0225  # Adjusted from 0.025

    # Read MIDI file
    # This used to take in whole nmat, but now just pitches.
    # note, vel, start time, end time, duration, note is processed    
    piece = Score(mid)
    notes = piece.midiPitches()['Piano']
    notes = notes[notes != -1]
    
    

    # ADJUST CONTRAST...
    # Plot spectrogram of the audio file
    fig = plt.figure(fig)
    plt.imshow(20 * np.log10(spec), aspect='auto', origin='lower', cmap='gray')
    plt.title('Spectrogram with Aligned MIDI Notes Overlaid')
    plt.xlabel('Time (.05s)')
    plt.ylabel('Midinote')
    plt.clim([plt.gci().get_clim()[0], plt.gci().get_clim()[1] - 50])
    plt.colorbar()
    # plt.show() # Uncomment to show

    # Zoom in on fundamental frequencies
    # notes = nmat[:, 0]  # Note
    notes = notes.values    
    notes = (2 ** ((notes - 105) / 12)) * 440
    notes = np.append(notes, notes[-1])
    nlim = len(notes)

    plot_fine_align(trace[0, :], trace[1, :],
                    notes[:nlim], stft_hop)  # Original

    # if trace.shape[0] >= 3: # Original

    #     if len(trace) >= 3:
    #         notenums = trace[2, 1:]
    #         # notenums = trace[2][1:]
    #     else:
    #         nlim = len(notes)
    #         notenums = np.concatenate([np.repeat(range(1, nlim + 1), 4), [nlim]])

    
def plot_fine_align(stateType, occupancy, notes, stftHop):    
    """    
    Plot the HMM alignment based on the output of YIN.

    Parameters:
        stateType: List of states in the HMM.
        occupancy: List indicating the time (in seconds) at which the states in stateType end.
        notes: List of MIDI note numbers that are played.
        stftHop: The hop size between frames in the spectrogram.
    """

    # Define styles for different states
    styles = [
        # {'color': 'red', 'marker': '+', 'linestyle': '-', 'linewidth': 2},
        {'color': 'none', 'marker': '+', 'linestyle': '-',
            'linewidth': 2},  # RED LINE RUNNING THROUGH?
        {'color': 'green', 'marker': '+', 'linestyle': '-', 'linewidth': 2},
        {'color': 'blue', 'marker': '+', 'linestyle': '-', 'linewidth': 2}]

    # Calculate segment boundaries
    cs = np.array(occupancy) / stftHop
    segments = np.vstack((cs[:-1], cs[1:])).T

    # Create the plot
    stateNote = (np.maximum(1, np.cumsum(stateType == 3) + 1)) - 1
    for i in range(segments.shape[0]):
        style = styles[int(stateType[i + 1]) - 1]
        x = segments[i, :]
        y = np.tile(notes[stateNote[i]], (2, 1))
        plt.plot(x, y, color=style['color'], marker=style['marker'],
                 linestyle=style['linestyle'], linewidth=style['linewidth'])

    plt.show()



# Currently not in use

def run_HMM_alignment(notenum, means, covars, align, yinres, sr, learnparams=False):    
    """
      Refines DTW alignment values with a three-state HMM, identifying 
      silence,transient, and steady state parts of the signal. The HMM  
      uses the DTW alignment as a prior. 
    
    Inputs:
      notenum - number of notes to be aligned
      means - 3x2 matrix of mean aperiodicy and power values HMM states
              column: silence, trans, steady state
              rows: aperiodicity, power
      covars - 3x2 matrix of covariances for the aperiodicy and power
               values (as per means)
      res - structure containing inital DTW aligment
      yinres - structure containg yin analysis of the signal
      sr - sampling rate of the signal
    
    Outputs: 
      vpath - verterbi path
      startingState - starting state for the HMM
      prior - prior matrix from DTW alignment
      trans - transition matrix
      meansFull - means matrix
      covarsFull - covariance matrix
      mixmat - matrix of priors for GMM for each state
      obs - two row matrix observations (aperiodicty and power)
      stateOrd - modified state order sequence
    """

    if not learnparams:
        shift = 0

    # Create vectors of onsets and offsets times from DTW alignment
    align['on'] = np.array(align['on'])
    align['off'] = np.array(align['off'])
    ons = np.floor(align['on'] * sr / 32).astype(int)
    offs = np.floor(align['off'] * sr / 32).astype(int)

    # Create observation matrix
    # obs = np.zeros((3, offs[notenum] + 50))
    obs = np.zeros((3, yinres['ap'].size))

    # - 1 to account for 0 index of Python
    obs[0, :] = np.sqrt(yinres['ap'][:offs[notenum - 1] + 50])
    # obs[1, :] = np.sqrt(yinres['pwr'][:offs[notenum - 1] + 50]) # Changed
    obs[1, :] = np.sqrt(yinres['time'][:offs[notenum - 1] + 50])
    # obs[2, :] = 69 + 12 * yinres['f0'][:offs[notenum - 1] + 50]  # Convert octave to MIDI note

    yinres['f0'] = np.ceil(yinres['f0'])
    midiPitches = librosa.hz_to_midi(yinres['f0'])
    # Convert octave to MIDI note
    obs[2, :] = midiPitches[:offs[notenum - 1] + 50]

    # Replace any NaNs in the observation matrix with zeros
    obs[np.isnan(obs)] = 0

    # Refine the list of onsets and offsets according to the number of notes
    prior_ons = ons[:notenum]  # Ignore added 0 placeholder
    prior_offs = offs[:notenum]
    notes = len(prior_ons)  # Normalize

    # Define states: silence, trans, steady state
    # Rows: aperiodicity, power
    state_ord_seed = [1, 2, 3, 2, 1]
    # state_ord = np.tile(state_ord_seed[:-1], notes) + [state_ord_seed[-1]] # This is 21 size
    state_ord = np.concatenate([np.tile(
        state_ord_seed[:-1], notes), [state_ord_seed[-1]]])  # This gives both 20 size

    # Use state_ord to expand means and covars for each appearance
    midi_notes = np.tile(align['midiNote'][:notenum], len(state_ord_seed) - 1)
    midi_notes = np.append(midi_notes, align['midiNote'][notenum - 1])
    # Changed state_ord - 1
    means_full = np.vstack((means[:, state_ord - 1], midi_notes))
    covars = covars.reshape(3, 2, 2)
    covars[0, 1, 0] = 100
    covars[1, 1, 0] = 5
    covars[2, 1, 0] = 1
    covars_full = covars[state_ord - 1, :, :]  # deleted one :, to make 2-D

    mixmat = np.ones(len(state_ord))

    # Transition matrix seed
    # {steady state, transient, silence, transient, steady state}
    # Original, commented out 4th index to see results...
    trans_seed = np.zeros((5, 5))
    # trans_seed = np.zeros((4, 4))
    trans_seed[0, 0] = 0.99
    trans_seed[1, 1] = 0.98
    trans_seed[2, 2] = 0.98
    trans_seed[3, 3] = 0.98
    trans_seed[4, 4] = 0.99
    trans_seed[0, 1] = 0.0018
    trans_seed[0, 2] = 0.0007
    trans_seed[0, 3] = 0.0042
    trans_seed[0, 4] = 0.0033
    trans_seed[1, 2] = 0.0018
    trans_seed[1, 3] = 0.0102
    trans_seed[1, 4] = 0.0080
    trans_seed[2, 3] = 0.0112
    trans_seed[2, 4] = 0.0088
    trans_seed[3, 4] = 0.02

    # Call filltransmat to expand the transition matrix to the appropriate size
    trans = fill_trans_mat(trans_seed, notes)

    # Create starting state space matrix
    starting_state = np.zeros(4 * notes + 1)
    starting_state[0] = 1

    prior = fill_priormat_gauss(obs.shape[0], prior_ons, prior_offs, 5)    
    
    if learnparams:
        # Use the fit function from the hmmlearn library to learn the HMM parameters
        model = hmm.GMMHMM(n_components=5, n_mix=1,
                           covariance_type='diag', n_iter=1)
        model.startprob_ = starting_state
        model.transmat_ = trans
        model.means_ = means_full.T
        model.covars_ = covars_full.T
        model.fit(obs.T)

    # like = mixgauss_prob(obs, means_full, covars_full, mixmat)

    # Use the Viterbi algorithm to find the most likely path
    # pr_like = prior * like
    # vpath = hmm.ViterbiHMM(starting_state, trans, pr_like)

    # Define the filename
    # pLikeData = "./test_files/priorlike_oneNote_runHMM.csv"
    pLikeData = "./test_files/priorlike_threeNote_runHMM.csv"
    # pLikeData = "./test_files/priorlike_sixNote_runHMM.csv"
    
    # Read the data from the file
    dtype = {'index': str, 'value': float}
    pr_like = pd.read_csv(pLikeData, dtype=dtype,
                          sep='\s+', names=['index', 'value'])

    # Initialize an empty dictionary to store the data
    data_dict = {}

    # Open the text file for reading
    with open(pLikeData, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into two parts based on whitespace
            parts = line.split()
            # Extract the index from the first part and convert it to a tuple
            index = tuple(map(int, parts[0].strip('()').split(',')))
            # Parse the value from the second part
            value = float(parts[1])
            # Store the data in the dictionary with the index as the key
            data_dict[index] = value

    # Determine the shape of the numpy array based on the maximum index values
    num_rows = max(index[0] for index in data_dict.keys())
    num_cols = max(index[1] for index in data_dict.keys())

    # Initialize a numpy array with zeros
    pr_like = np.zeros((num_rows, num_cols))

    # Fill the numpy array with the values from the dictionary
    for index, value in data_dict.items():
        pr_like[index[0] - 1, index[1] - 1] = value

    vpath = viterbi_path(starting_state, trans, pr_like)
    # vpath = librosa.sequence.viterbi(prob=starting_state, transition=trans, pr_like)
   
    return vpath, starting_state, prior, trans, means_full, covars_full, mixmat, obs, state_ord
