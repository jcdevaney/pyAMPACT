# COMPLETE/CONDITIONAL PASS TEST (see note in test class)
import numpy as np

def perceived_pitch(f0s, sr, gamma=100000):        
    # Remove NaN values from f0s
    f0s = f0s[~np.isnan(f0s)]
    
    # Create an index to remove outliers by using the central 80% of the sorted vector
    ord = np.argsort(f0s)
    ind = ord[int(len(ord)*0.1):int(len(ord)*0.9)]

    # Calculate the rate of change
    deriv = np.append(np.diff(f0s) * sr, -100)        
            
    # Set weights for the quickly changing vs slowly changing portions
    # WEIGHTS ARE 0., incorrect!!
    weights = np.exp(-gamma * np.abs(deriv))

    # But is this?
    # weights = np.exp(-gamma / np.abs(deriv))    
    
    # Calculate two versions of the perceived pitch
    pp1 = np.sum(f0s * weights) / np.sum(weights)
    pp2 = np.sum(f0s[ind] * weights[ind]) / np.sum(weights[ind])
    
    return pp1, pp2    

# Example usage:
# pp1, pp2 = perceived_pitch(f0s=np.array([220.0, 330.0, 440.0, 330.0, 220.0]), sr=(4000/32), gamma=100000)
