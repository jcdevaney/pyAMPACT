from mido import MidiFile, MidiTrack, Message


def nmat2midi(nmat, filename, ticks=120):
    # Create a new MIDI file
    mid = MidiFile()

    # Create a single track in the MIDI file
    track = MidiTrack()
    mid.tracks.append(track)

    # Set ticks per quarter note
    mid.ticks_per_beat = ticks

    # Iterate through the notes in the nmat and add them to the track
    for row in nmat:
        note_on = Message('note_on', note=int(
            row[2]), velocity=int(row[3]), time=int(row[4] * ticks))
        note_off = Message('note_off', note=int(
            row[2]), velocity=0, time=int(row[5] * ticks))
        track.append(note_on)
        track.append(note_off)

    # Save the MIDI file
    mid.save(filename)
