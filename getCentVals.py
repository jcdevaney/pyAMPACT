import numpy as np
import math
import sys
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
    for i in range(len(times['ons'])):
        # Define the reference pitch (A4 = 440 Hz)
        ref_pitch = 440.0

        onset_index = round(times['ons'][i] / 32 * sr)
        offset_index = round(times['offs'][i] / 32 * sr)                        

        # Example array of frequencies
        frequencies = yinres['f0'][onset_index:offset_index]

        # Calculate the cent values for each frequency
        cent = 1200 * np.log2(frequencies / ref_pitch)
        

        # average_cent = np.mean(cent)        
        # WITHOUT AVERAGING        
        # cents.append(cent)

        # Check if the array is not empty and contains finite values
        if np.any(np.isfinite(cent)):
        # Calculate the mean only if the array is not empty and has finite values
            # average_cent = np.nanmean(cent)
            # cents.append(average_cent)
            cents.append(cent)            
        else:
            break
        
    
   

    return cents

    
