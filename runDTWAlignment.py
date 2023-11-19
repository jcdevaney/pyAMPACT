import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

import sys

from mido import MidiFile

from alignMidiWav import align_midi_wav
from midi2nmat import midi2nmat


def runDTWAlignment(audiofile, midorig, tres, width, targetsr, nharm, winms):
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

    # Iterate through the MIDI tracks
    currentTime = 0
    for track in midi_data.tracks:
        for msg in track:
            if msg.type == 'note_on':
                on_times.append(currentTime + msg.time)
                currentTime += msg.time
                # CATCH as no note_off msgs are in the example.mid
                off_times.append(currentTime + 5)
                midi_notes.append(msg.note)
            elif msg.type == 'note_off':
                off_times.append(msg.time)

    off_times.pop(0)  # Remove first index as it is not a true off time

    # Convert times to absolute times
    cumulative_time = 0
    on_times = [cumulative_time + time for time in on_times]
    off_times = [cumulative_time + time for time in off_times]

    # Assuming you want data for the first instrument
    align = {
        'nmat': nmat,
        'on': on_times,
        'off': off_times,
        'midiNote': midi_notes
    }

    return align, spec, dtw
