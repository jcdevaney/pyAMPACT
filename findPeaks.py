import numpy as np

import sys
#########################################################################
# [mins maxes] = findPeaks(x, windowLength_ms, sr, minCount)
#
# Description: Find peaks and troughs in a waveform
#              Finds the max and min in a window of a given size and keeps
#              track of how many windows each point is the min or max of.
#              Points that are the min or max of more than minCount windows
#              are returned.
#
# Inputs:
#  x - inputted signal
#  windowLength_ms - window length in ms
#  sr - sampling rate
#  minCount - minimum number of windows that a point needs to be the max
#             of to be considered a minimum or a maximum
#
# Outputs:
#  mins - minimum values in the signal
#  maxes - maximum values in the signal
#
# Automatic Music Performance Analysis and Analysis Toolkit (AMPACT)
# http://www.ampact.org
# (c) copyright 2011 Johanna Devaney (j@devaney.ca) and Michael Mandel
#                    (mim@mr-pc.org), all rights reserved
#########################################################################


def find_peaks(x, window_length_ms, sr, min_count):
    # Create an array    
    x = np.array(x)    
    
    # Create arrays of zeros for mins and maxes
    mins = np.zeros_like(x)
    maxes = np.zeros_like(x)    

    # Calculate window length in samples
    window_length = int(window_length_ms * sr / 1000)    
    # Calculate the minimum and maximum values
    for i in range(len(x) - window_length):
        w = x[i:i+window_length]
        di_min = np.argmin(w)
        di_max = np.argmax(w)
        mins[i + di_min] += 1 # THIS IS LIKELY THE ISSUE, DOING 0 - 1 versus what Python wants (yinres normalizing?)
        maxes[i + di_max] += 1

    # Prune mins and maxes to only those that occur in minCount or more windows
    # START HERE!        
    mins = np.where(mins >= min_count)[0]
    maxes = np.where(maxes >= min_count)[0]

    return mins, maxes
