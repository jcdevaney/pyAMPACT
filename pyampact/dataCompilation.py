import numpy as np


from pyampact.performance import estimate_perceptual_parameters
from pyampact.symbolic import Score

__all__ = [
    "data_compilation"
]

def data_compilation(f0_values, sig_pwr, mag_mat, nmat, target_sr, hop_length, y, audio_file_path=None, export_path=None):    
    # Iterate over the indices of XML_IDs
    for key, df in nmat.items():
        total_duration = df['OFFSET_SEC'].iloc[-1]
        for i, row in df.iterrows():                        
            start_time = row['ONSET_SEC']
            end_time = row['OFFSET_SEC']            
            start_idx = int(start_time * len(f0_values) / total_duration)
            end_idx = int(end_time * len(f0_values) / total_duration)     

            # Extract values for the current time interval
            f0_chunk = f0_values[start_idx:end_idx]
            pwr_chunk = sig_pwr[start_idx:end_idx]                            
            mag_mat_chunk = mag_mat[start_idx:end_idx]        
            perceptual_params = estimate_perceptual_parameters(f0_vals=f0_chunk, pwr_vals=pwr_chunk, M=mag_mat_chunk, SR=target_sr, hop=hop_length, gt_flag=True, y=y)        

            pwr_chunk = perceptual_params['pwr_vals'][start_idx:end_idx]              


            # Create a dictionary for the current time interval - added np.mean                
            # df.loc[i,'f0Vals'] = str(f0_chunk)
            df.loc[i,'ppitch1'] = perceptual_params['ppitch'][0]
            df.loc[i,'ppitch2'] = perceptual_params['ppitch'][1]
            df.loc[i,'jitter'] = perceptual_params['jitter']
            df.loc[i,'vibratoDepth'] = perceptual_params['vibrato_depth']
            df.loc[i,'vibratoRate'] = perceptual_params['vibrato_rate']
            df.loc[i,'pwrVals'] = str(pwr_chunk)        
            df.loc[i,'avgPwr'] = np.mean(perceptual_params['pwr_vals'])
            df.loc[i,'shimmer'] = perceptual_params['shimmer']
            # df.loc[i,'specCent'] = perceptual_params['spec_centroid']
            df.loc[i,'specCentMean'] = perceptual_params['mean_spec_centroid']
            # df.loc[i,'specSlope'] = perceptual_params['spec_slope']
            df.loc[i,'meanSpecSlope'] = perceptual_params['mean_spec_slope']
            # df.loc[i,'specFlux'] = perceptual_params['spec_flux']
            df.loc[i,'meanSpecFlux'] = perceptual_params['mean_spec_flux']
            # df.loc[i,'specFlat'] = perceptual_params['spec_flat']
            df.loc[i,'meanSpecFlat'] = perceptual_params['mean_spec_flat']        
            # Add other parameters and their corresponding chunks here

    nmat, jsonData = Score.toJSON(nmat)
    print(nmat)
    return nmat, jsonData
    