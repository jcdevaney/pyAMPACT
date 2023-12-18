import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

import sys

from mido import MidiFile

from alignMidiWav import align_midi_wav
from midi2nmat import midi2nmat


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

    Dependencies:
    - Ellis, D. P. W. 2003. Dynamic Time Warp (DTW) in Matlab.
      Available from: http://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/
    - Ellis, D. P. W. 2008. Aligning MIDI scores to music audio.
      Available from: http://www.ee.columbia.edu/~dpwe/resources/matlab/alignmidiwav/
    - Toiviainen, P. and T. Eerola. 2006. MIDI Toolbox.
      Available from: https://www.jyu.fi/hum/laitokset/musiikki/en/research/coe/materials/miditoolbox/

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
