import pretty_midi

def get_timing_data(midifile, times):
    # Read quantized MIDI file
    midi_data = pretty_midi.PrettyMIDI(midifile)
    
    # Convert note onset and offset times to seconds
    # onset_seconds = [midi_data.time_to_tick(time) / midi_data.tick_to_time(1) for time in times['ons']] # Original
    onset_seconds = [midi_data.time_to_tick(time) / midi_data.tick_to_time(1) for time in times]
    
    return onset_seconds
   
