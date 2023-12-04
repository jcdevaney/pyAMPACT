import numpy as np

def get_cent_vals(times, yinres, sr):
    # CENTS IN BETWEEN SEQUENTIAL NOTES
    # Check the reference of yin in MATLAB vs yin in librosa
    """
    Get cent values (in relation to A, 440 Hz) for each note.
    

    Parameters:
    - times: Dictionary-like object with 'ons' and 'offs' representing onset and offset times.
    - yinres: Dictionary-like object with 'f0' and 'sr'.

    Returns:
    - cents: List of NumPy arrays containing cent values for each note.
    """

    cents = []

    # Index into f0 estimates in YIN structure with onset and offset times        
    oct = np.log2(yinres['f0'] / 440)
    
    # midi_notes = 12 * np.log2(yinres['f0'] / 440) + 69    
    # oct = (midi_notes - 67) / 12    
    for i in range(len(times['ons'])):
        onset_index = round(times['ons'][i] / 32 * sr)
        offset_index = round(times['offs'][i] / 32 * sr)                        
        pitch_segment = oct[onset_index:offset_index] * 1200
        cents.append(pitch_segment)
    return cents


# # COMPLETE NEEDS TESTING

# def get_cent_vals(times, yinres):
#     cents = []

#     # Index into f0 estimates in YIN structure with onset and offset times
#     for i in range(len(times['ons'])):
#         onset_sample = int(times['ons'][i] / 32 * yinres['sr'])
#         offset_sample = int(times['offs'][i] / 32 * yinres['sr'])

#         # Extract the corresponding portion of f0 and convert to cents
#         f0_segment = yinres['f0'][onset_sample:offset_sample] * 1200
#         cents.append(f0_segment)

#     return cents
