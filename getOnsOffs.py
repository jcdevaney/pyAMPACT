# PASS, same in MATLAB
import numpy as np

#########################################################################
# res=getOnsOffs(onsoffs)
#
# Description: Extracts a list of onset and offset from an inputted 
#              3*N matrix of states and corresponding ending times 
#              from AMPACT's HMM-based alignment algorithm
#
# Inputs:
#  onsoffs - a 3*N alignment matrix, the first row is a list of N states
#            the second row is the time which the state ends, and the
#            third row is the state index
#
# Outputs:
#  res.ons - list of onset times
#  res.offs - list of offset times
#
# Automatic Music Performance Analysis and Analysis Toolkit (AMPACT) 
# http://www.ampact.org
# (c) copyright 2011 Johanna Devaney (j@devaney.ca) 
#########################################################################

def get_ons_offs(onsoffs):
    # Find indices where the first row is equal to 3
    stopping = np.where(onsoffs[0] == 3)[0]
    # Calculate starting indices by subtracting 1 from stopping indices
    starting = stopping - 1

    res = {'ons': [], 'offs': []}

    for i in range(len(starting)):
        res['ons'].append(onsoffs[1, starting[i]])
        res['offs'].append(onsoffs[1, stopping[i]])
    
    return res
