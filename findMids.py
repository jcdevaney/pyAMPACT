# COMPLETE/TESTED
import numpy as np

#########################################################################
# mids = findMids(x, mins, maxes, windowLength_ms, sr)
#
# Description: Find the midpoints between mins and maxes in a signal x.
#              mins and maxes could come from findPeaks.  Finds the y 
#              values of peaks and then finds the x values of the signal 
#              that are closest to the average between the min and max 
#              peak.
#
# Inputs:
#  x - inputted signal in cents
#  mins - indices of minima of x
#  maxes - indices of maxima of x
#  windowLength_ms - window length in miliseconds
#  sr - sampling rate of x (frame rate of frequency analysis)
#
# Outputs:
#  x_mids - midpoint locations in x axis between peaks and troughs  
#  y_mids - midpoint locations in y axis between peaks and troughs  
#
# Automatic Music Performance Analysis and Analysis Toolkit (AMPACT) 
# http://www.ampact.org
# (c) copyright 2011 Johanna Devaney (j@devaney.ca) and Michael Mandel
#                    (mim@mr-pc.org), all rights reserved
#########################################################################

def find_mids(x, mins, maxes, windowLength_ms, sr):
    # Convert window length from milliseconds to frames
    windowLength = int(round(windowLength_ms * sr / 1000.0) * 2)

    # Combine minima and maxima indices and sort them
    pks = np.sort(np.concatenate((maxes, mins)))

    # Initialize an array to store neighborhood means
    neighborhoods = np.zeros(len(pks))

    # Compute the average frequency estimate around each peak
    for i in range(len(pks)):
        start_idx = max(pks[i] - windowLength // 2, 0)
        end_idx = min(pks[i] + windowLength // 2, len(x))
        idx = range(start_idx, end_idx)
        neighborhoods[i] = np.mean(x[idx])

    # Find the mid-points in frequency between peaks
    y_mids = (neighborhoods[:-1] + neighborhoods[1:]) / 2

    # Find the index of the point in the signal between each peak
    # with its value closest to the mid-point in frequency
    x_mids = np.zeros(len(y_mids), dtype=int)
    for i in range(len(y_mids)):
        idx = range(pks[i], pks[i + 1] + 1)
        offset = np.argmin(np.abs(y_mids[i] - x[idx]))
        x_mids[i] = pks[i] + offset

    return x_mids, y_mids


# OTHER VERSION
# import numpy as np

# def find_mids(x, mins, maxes, windowLength_ms, sr):

#     #########################################################################
#     # mids = findMids(x, mins, maxes, windowLength_ms, sr)
#     #
#     # Description: Find the midpoints between mins and maxes in a signal x.
#     #              mins and maxes could come from findPeaks.  Finds the y 
#     #              values of peaks and then finds the x values of the signal 
#     #              that are closest to the average between the min and max 
#     #              peak.
#     #
#     # Inputs:
#     #  x - inputted signal in cents
#     #  mins - indices of minima of x
#     #  maxes - indices of maxima of x
#     #  windowLength_ms - window length in miliseconds
#     #  sr - sampling rate of x (frame rate of frequency analysis)
#     #
#     # Outputs:
#     #  x_mids - midpoint locations in x axis between peaks and troughs  
#     #  y_mids - midpoint locations in y axis between peaks and troughs  
#     #
#     # Automatic Music Performance Analysis and Analysis Toolkit (AMPACT) 
#     # http://www.ampact.org
#     # (c) copyright 2011 Johanna Devaney (j@devaney.ca) and Michael Mandel
#     #                    (mim@mr-pc.org), all rights reserved
#     #########################################################################
    
#     # window length in frames
#     windowLength = round(windowLength_ms * sr / 2000) * 2;
    
#     # sort the peaks    
#     pks = np.sort(np.concatenate((maxes, mins)))
    
#     # average the frequency estimate of the points around each peak
#     for i in range(len(pks)):
#         idx = slice(max(pks[i]-windowLength/2, 1), min(pks[i]+windowLength/2, len(x)))
#         neighborhoods[i] = sum(x[idx]) / len(idx)
    
#     # find the mid-points in frequency between peaks
#     x_mids = []
#     y_mids = [(neighborhoods[i] + neighborhoods[i+1]) / 2 for i in range(len(neighborhoods)-1)]
    
#     # find the index of the point in the signal between each peak with its
#     # value closest to the mid-point in frequency
#     for i in range(len(y_mids)):
#         idx = pks[i:pks[i+1]]
#         d, offset = min(abs(y_mids[i] - x[idx]))
#         x_mids[i] = pks[i] + offset - 1
    
#     return x_mids, y_mids