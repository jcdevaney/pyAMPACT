import os
curr_dir = os.getcwd()
from symbolic import Score

import numpy as np
import mido

from sklearn.covariance import ShrunkCovariance
from hmmlearn import hmm



def get_ons_offs(onsoffs):
    """
    Extracts a list of onset and offset from an inputted 
             3*N matrix of states and corresponding ending times 
             from AMPACT's HMM-based alignment algorithm
    Inputs:
        onsoffs - a 3*N alignment matrix, the first row is a list of N states
           the second row is the time which the state ends, and the
           third row is the state index
    Outputs:
        res.ons - list of onset times
        res.offs - list of offset times
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



def midi2nmat(filename):
    """
    Read midi file FILENAME into Matlab variable NMAT (Beta)
    Based on Ken Schutte's m-files (readmidi, midiInfo, getTempoChanges)
    This beta might replace the mex-files used in the previous version of the toolbox as 
    newer versions of Matlab (7.4+) and various OS's need new compilations 
    of the mex files. Using the C sources and the compiled mex files provides
    faster reading of midi files but because the compatibility is limited, this 
    simple workaround is offered. This beta version is very primitive,
    though. - Tuomas Eerola

    KNOWN PROBLEMS: - Tempo changes are handled in a simple way
                    - Extra messages are not retained  
                    - Channels may not be handled correctly    

    For more information on Ken Schutte's functions, see 
    http://www.kenschutte.com/software

    CREATED ON 31.12.2007 BY TE (MATLAB 7.4 MacOSX 10.4)
    """
    mid = mido.MidiFile(filename)
    nmat = []

    # Convert ticks per beat to seconds per tick
    ticks_per_beat = mid.ticks_per_beat    
    seconds_per_tick = 60 / (5000000 / ticks_per_beat)

    current_tempo = 500000  # Default tempo

    for track in mid.tracks:
        cum_time = 0
        starttime = 0

        for msg in track:
            cum_time += msg.time

            if msg.type == 'set_tempo':
                tempo = msg.tempo
                current_tempo = tempo

            if msg.type == 'note_on' and msg.velocity > 0:
                note = msg.note
                velocity = msg.velocity
                start_time = cum_time * seconds_per_tick
                nmat.append([note, velocity, start_time, 0, 0, 0])

            if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                for event in reversed(nmat):
                    if event[2] is not None and event[2] <= cum_time * seconds_per_tick:
                        end_time = cum_time * seconds_per_tick
                        duration = end_time - event[2]
                        event[3] = end_time
                        event[4] = duration
                        event[5] = 1  # Mark the note as processed
                        break

            if msg.type == 'end_of_track':
                if len(nmat) > 0:
                    last_event = nmat[-1]
                    last_event[5] = 1  # Mark the note as processed

    # Filter out unprocessed notes
    nmat = [event for event in nmat if event[5] == 1]

    return np.array(nmat)

    
def get_timing_data(midifile, times):
    """
    Create a note matrix with performance timings

    Inputs:
        midifile - name of midifile
        times - note onset and offset times

    Outputs:
        nmatNew - MIDI toolbox note matrix with performance timings
                   (mim@mr-pc.org), all rights reserved
    """    
    # Read quantized MIDI file
    midi_data = Score(midifile)            
    nmat_from_script = midi_data.nmats() 

    nmat_vals = nmat_from_script['Piano'].values
    new_nmat_from_script = nmat_vals[nmat_vals[:, 4] != -1]
    print(new_nmat_from_script)


    nmat_from_func = midi2nmat(midifile) # This instead?
    print(nmat_from_func[1::2])
    

    # START HERE!!!
    # Then build the proper nmat as from MATLAB.  DOES THIS NEED TO BE BUILT???
    nmat_old = np.empty((0,7))    
    
    nmat_old[:, :2] /= 2 # Problem???    
    
    # Replace timing information in MIDI file with performance timings
    nmat_new = nmat_old.copy()
    
    nmat_new[:, 5:7] = [times['ons'].values, (times['offs'] - times['ons']).values]
    offset = nmat_new[0, 5] - nmat_old[0, 0]
    nmat_new[:, 5] -= offset
    nmat_new[:, :2] = nmat_new[:, 5:7]

    return nmat_new


# Work in Progress...

# def nearestPD(A):
#     B = (A + A.T) / 2
#     _, s, V = np.linalg.svd(B)
#     H = np.dot(V.T, np.dot(np.diag(s), V))
#     A2 = (B + H) / 2
#     A3 = (A2 + A2.T) / 2
#     if isPD(A3):
#         return A3
#     spacing = np.spacing(np.linalg.norm(A))
#     I = np.eye(A.shape[0])
#     k = 1
#     while not isPD(A3):
#         mineig = np.min(np.real(np.linalg.eigvals(A3)))
#         A3 += I * (-mineig * k**2 + spacing)
#         k += 1
#     return A3


# def isPD(B):
#     try:
#         _ = np.linalg.cholesky(B)
#         return True
#     except np.linalg.LinAlgError:
#         return False

# def select_states(starting_state, prior, trans, means_full, covars_full, mixmat, obs, state_o, note_num, sr):
#     # Provided variables
#     starting_state = np.array([1., 0., 0., 0., 0.])
#     prior = np.random.dirichlet(alpha=np.ones(3), size=5)
#     trans = np.array([[9.90e-01, 9.90e-01, 9.90e-01, 9.90e-01, 0.00e+00],
#                       [9.90e-01, 9.90e-01, 1.80e-03, 7.00e-04, 4.20e-03],
#                       [9.90e-01, 0.00e+00, 9.80e-01, 1.80e-03, 1.02e-02],
#                       [9.90e-01, 0.00e+00, 0.00e+00, 9.80e-01, 1.12e-02],
#                       [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 9.80e-01]])
#     means_full = np.array([[5.4908e-01, 2.9714e-01, 2.9415e-02, 2.9714e-01, 5.4908e-01],
#                            [1.1829e-02, 1.7721e-01, 9.3534e-01,
#                                1.7721e-01, 1.1829e-02],
#                            [6.9000e+01, 6.9000e+01, 6.9000e+01, 6.9000e+01, 6.9000e+01]])
#     covars_full = np.array([[[2.870e-02, 0.000e+00],
#                              [1.000e+02, 3.500e-03]],
#                             [[6.000e-02, 0.000e+00],
#                              [5.000e+00, 3.420e-02]],
#                             [[4.900e-03, 0.000e+00],
#                              [1.000e+00, 4.474e-01]],
#                             [[6.000e-02, 0.000e+00],
#                              [5.000e+00, 3.420e-02]],
#                             [[2.870e-02, 0.000e+00],
#                              [1.000e+02, 3.500e-03]]])
#     mixmat = np.array([1., 1., 1., 1., 1.])
#     obs = np.array([[0., 0., 1., 1., 1., 1.],
#                     [0.1, 0.1, 1., 1., 1., 1.],
#                     [0., 0., 71.31665852, 70.10501279, 70.10501279, 70.10501279]])
#     state_ord2 = np.array([1, 3, 1])
#     note_num = np.array([1, 1, 1])
#     sr = 4000
#     # Create new versions of the inputted variables based on the state sequence stateO
#     vec = (state_ord2 + (note_num - 1) * 4)
#     # starting_state2 = starting_state[state_ord2]
#     prior2 = prior[vec, :]
#     trans2 = trans[vec, :][:, vec]
#     trans2 = np.diag(1. / np.sum(np.atleast_2d(trans2), axis=1)
#                      ) @ np.atleast_2d(trans2)
#     means_full2 = means_full[:, vec]
#     covars_full2 = covars_full[vec, :, :]
#     # mixmat2 = mixmat[vec, :]

#     # Ensure covariance matrices are symmetric and positive-definite
#     covars_full2_fixed = np.zeros_like(covars_full2)

#     for i in range(len(vec)):
#         cov_matrix = covars_full2[i]
#         shrunk_cov = ShrunkCovariance().fit(cov_matrix).covariance_
#         covars_full2_fixed[i] = 0.5 * (shrunk_cov + shrunk_cov.T)

#     # Create an HMM model and set the parameters
#     model = hmm.GaussianHMM(n_components=len(vec), covariance_type="full")
#     model.startprob_ = prior2
#     model.transmat_ = trans2
#     model.means_ = means_full2.T
#     model.covars_ = covars_full2_fixed
#     # model.weights_ = mixmat2

#     # Calculate the Viterbi path
#     vpath2 = model.predict(obs.T)

#     # Create a vector of the modified alignment times
#     histvals2, _ = np.histogram(vpath2, bins=np.arange(1, max(vpath2) + 2))
#     cumsumvals2 = np.cumsum(histvals2 * 32 / sr)

#     return vpath2, histvals2, cumsumvals2