import mido
import numpy as np

def midi2nmat(filename):
    mid = mido.MidiFile(filename)
    nmat = []

    # Convert ticks per beat to seconds per tick
    ticks_per_beat = mid.ticks_per_beat
    seconds_per_tick = 60 / (500000 / ticks_per_beat)

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



# SECOND TO LAST
# import mido
# from mido import MidiFile

# def midi2nmat(filename):
#     mid = MidiFile(filename)
#     nmat = []

#     for i, track in enumerate(mid.tracks):
#         for msg in track:
#             if msg.type == 'note_on':
#                 nmat.append([i, msg.channel, msg.note, msg.velocity, msg.time, 0, 0])
#             elif msg.type == 'note_off':
#                 for note in nmat[::-1]:
#                     if note[0] == i and note[1] == msg.channel and note[2] == msg.note and note[7] == -1:
#                         note[5] = msg.time
#                         note[7] = msg.time

#     tempos = []
#     tempos_time = []

#     for msg in mid.tracks[0]:
#         if msg.type == 'set_tempo':
#             tempos_time.append(msg.time)
#             tempo = mido.tempo2bpm(msg.tempo)
#             tempos.append(tempo)

#     nmat2 = nmat.copy()
#     for i, tc_time in enumerate(tempos_time):
#         if i == len(tempos_time) - 1:
#             time_index = [note[4] >= tc_time for note in nmat2]
#         else:
#             time_index = [note[4] >= tc_time for note in nmat2]  # & nmat2[:,4] <= tempos_time[i+1]

#         timeratio = tempos[i] / tempos[0]
#         tmp1 = [note.copy() for note in nmat2 if note[4] >= tc_time]
#         tmp2 = tmp1[0][5] * timeratio
#         realign = tmp1[0][4] - tmp2

#         for note in nmat2:
#             if note[4] >= tc_time:
#                 note[5] = note[5] * timeratio + realign
#                 note[6] = note[6] * timeratio

#     for note in nmat:
#         note[0] = note[4] / (tempos[0] / 1000000)
#         note[1] = (note[5] - note[4]) / (tempos[0] / 1000000)

#     # for note in nmat:
#     #     if note[1] == 0:
#     #         note[1] = 1

#     return nmat

# # FORMER
# import numpy as np
# from mido import MidiFile

# def midi2nmat(filename):
#     # Read MIDI file
#     midi = MidiFile(filename)
    
#     nmat = []
#     n = []
#     tempos = []
#     tempos_time = []
    
#     # Iterate through MIDI tracks
#     for i, track in enumerate(midi.tracks):
#         track_number = i + 1
        
#         for msg in track:
#             if msg.type == 'note_on':
#                 note_number = msg.note
#                 velocity = msg.velocity
#                 time = msg.time / midi.ticks_per_beat  # Convert ticks to beats
#                 duration = msg.time / midi.ticks_per_beat  # Convert ticks to beats

#                 nmat.append([track_number, note_number, velocity, time, duration])
                
#             elif msg.type == 'set_tempo':
#                 microsecond_per_quarter_note = msg.tempo
#                 time = sum(tempos_time) / midi.ticks_per_beat  # Convert ticks to beats
#                 tempos_time.append(time)
#                 tempos.append(microsecond_per_quarter_note)
    
#     # Convert tempos to beats
#     tempos_in_beats = [60.0 / (tempo / 1000000.0) for tempo in tempos]
    
#     # Update timing information in nmat
#     for event in nmat:
#         event[3] = event[3] / midi.ticks_per_beat  # Convert ticks to beats
#         event[4] = event[4] / midi.ticks_per_beat  # Convert ticks to beats
    
#     return np.array(nmat), tempos_in_beats

