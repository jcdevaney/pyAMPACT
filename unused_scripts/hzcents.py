# COMPLETE
import numpy as np


def hzcents(x1, x2):
    
    # Calculates the difference in cents between two frequencies.

    # Parameters:
    # x1 (float): Frequency one in hertz.
    # x2 (float): Frequency two in hertz.

    # Returns:
    # cents (float): Size of the interval in cents between x1 and x2.
    

    if x1 == 0 or x2 == 0:
        cents = 0
    else:
        cents = 1200 * np.log2(x2 / x1)

    return cents