import numpy as np
import pandas as pd
import os
import json
import sys


from .performance import estimate_perceptual_parameters, get_cent_vals

__all__ = [
    "data_compilation"
]

def data_compilation(f0_values, sig_pwr, freq_mat, mag_mat, nmat, target_sr, hop_length, audio_file):
    # total_rows = sum(len(df) for df in nmat.values())
    # Iterate over the indices of XML_IDs
    for key, df in nmat.items():
        total_duration = df['OFFSET_SEC'].iloc[-1]  # Assuming your DataFrame is named df        
        for i, row in df.iterrows():                                     
            start_time = row['ONSET_SEC']
            end_time = row['OFFSET_SEC']            
            start_idx = int(start_time * len(f0_values) / total_duration)
            end_idx = int(end_time * len(f0_values) / total_duration)     

            # Extract values for the current time interval
            f0_chunk = f0_values[start_idx:end_idx]
            pwr_chunk = sig_pwr[start_idx:end_idx]                            
            mag_mat_chunk = mag_mat[start_idx:end_idx]        
            perceptual_params = estimate_perceptual_parameters(f0_vals=f0_chunk, pwr_vals=pwr_chunk, M=mag_mat_chunk, SR=target_sr, hop=hop_length, gt_flag=True, X=audio_file)        

            pwr_chunk = perceptual_params['pwr_vals'][start_idx:end_idx]
            slope_chunk = perceptual_params['spec_slope'][start_idx:end_idx]
            flux_chunk = perceptual_params['spec_flux'][start_idx:end_idx]
            flat_chunk = perceptual_params['spec_flat'][start_idx:end_idx]    


            # Create a dictionary for the current time interval - added np.mean                
            # df.loc[i,'f0Vals'] = str(f0_chunk)
            df.loc[i,'ppitch1'] = perceptual_params['ppitch'][0]
            df.loc[i,'ppitch2'] = perceptual_params['ppitch'][1]
            df.loc[i,'jitter'] = perceptual_params['jitter']
            # df.loc[i,'vibratoDepth'] = perceptual_params['vibrato_depth']
            # df.loc[i,'vibratoRate'] = perceptual_params['vibrato_rate']
            # df.loc[i,'pwrVals'] = str(pwr_chunk)        
            df.loc[i,'avgPwr'] = np.mean(perceptual_params['pwr_vals'])
            df.loc[i,'shimmer'] = perceptual_params['shimmer']
            # df.loc[i,'specCent'] = perceptual_params['spec_centroid'] # can get this via librosa
            # df.loc[i,'specCentMean'] = 1370.1594532691213
            # df.loc[i,'specSlope'] = str(slope_chunk)
            df.loc[i,'meanSpecSlope'] = perceptual_params['mean_spec_slope']
            # df.loc[i,'spec_flux'] = str(flux_chunk)
            df.loc[i,'mean_spec_flux'] = perceptual_params['mean_spec_flux']
            # df.loc[i,'spec_flat'] = str(flat_chunk)
            df.loc[i,'mean_spec_flat'] = perceptual_params['mean_spec_flat']        
            # Add other parameters and their corresponding chunks here


    # Convert each DataFrame to a dictionary
    # Create an empty dictionary to hold the structured data
    dfs_dict = {}

    for part, df in nmat.items():
        part_data = {}
        for xml_id, row in df.iterrows():        
            part_data[str(xml_id)] = row.to_dict()    
        dfs_dict[part] = part_data
    audio_file_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = f"./output_files/alignment_cdata_{audio_file_name}.json"
    with open(output_file, "w") as f:
        json.dump(dfs_dict, f, indent=4)



    # Everything beyond here is not yet live...
    sys.exit()


    # # Get cent values for each note
    # cents = get_cent_vals(times, yin_res, target_sr)  # Original

    # # Calculate intervals size, perceived pitch, vibrato rate, and vibrato depth
    # # vibrato_depth, vibrato_rate, interval_size, perceived_pitch = get_pitch_vibrato_data(cents, target_sr)

    # # Get DCT values for each note
    # mins = []
    # maxes = []
    # x_mids = []
    # y_mids = []
    # smoothed_f0s = []
    # steady = []
    # dct_vals = []
    # approx = []


    # # CHECK FOR COMPARISON TO MATLAB.  SPECIFICALLY, FOR EXAMPLE:
    # # Line 147 is actually: min_idx[i], max_idx[i] = find_peaks(cents[i], win_ms, int(target_sr/32), 60)
    # for i in range(len(cents)):    
    #     # Find peaks and troughs in the F0 trace for each note    

    #     min_idx, max_idx = find_peaks(cents[i], win_ms, int(target_sr/32), 60)
    #     mins.append(min_idx)
    #     maxes.append(max_idx)


    #     # Find midpoints between mins and maxes
    #     x_mid, y_mid = find_mids(cents[i], mins[i], maxes[i], 100, int(target_sr/32))
    #     x_mids.append(x_mid)
    #     y_mids.append(y_mid)    

    #     """
    #     BUGGED AFTER THIS POINT

    #     # Generate a smoothed trajectory of a note by connecting the midpoints,    
    #     smoothed_f0 = smooth_note(cents[i], x_mid, y_mid)
    #     smoothed_f0s.append(smoothed_f0)        

    #     # Find the steady-state portion of a note
    #     steady = find_steady(cents[i], mins[i], maxes[i], x_mid, y_mid, 1)    
    #     # steady.append([steady_start, steady_end])

    #     # # Compute the DCT of a signal and approximate it with the first 3 coefficients
    #     # dct_val, approx_val = note_dct(smoothed_f0[steady_start:steady_end], 3, int(target_sr/32))
    #     # dct_vals.append(dct_val)
    #     # approx.append(approx_val)

    # # Get loudness values for each note (you will need to implement this function)
    # # loudness_estimates, loudness_structure = getLoudnessEstimates(audio_file, onsets, offsets)
    # """