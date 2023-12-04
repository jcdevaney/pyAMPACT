# COMPLETE NEEDS TESTING
import itertools
import numpy as np

def genMeansCovars(notes, vals, voiceType):
    # Initialize variables
    numVoice = len(voiceType)
    noteMean1 = np.zeros((numVoice, 3))
    noteCovar1 = np.zeros((numVoice, 3))
    noteMean2 = np.zeros((numVoice, 3))
    noteCovar2 = np.zeros((numVoice, 3))
    versions = {}
    
    # Populate noteMean1, noteCovar1, noteMean2, noteCovar2
    for i in range(numVoice):
        noteMean1[i, 0] = vals[0][voiceType[i]][0]
        noteMean1[i, 1] = vals[1][voiceType[i]][0]
        noteMean1[i, 2] = vals[1][voiceType[i]][0]
        noteCovar1[i, 0] = vals[0][voiceType[i]][1]
        noteCovar1[i, 1] = vals[1][voiceType[i]][1]
        noteCovar1[i, 2] = vals[1][voiceType[i]][1]
        noteMean2[i, 0] = vals[1][voiceType[i]][0]
        noteMean2[i, 1] = vals[1][voiceType[i]][0]
        noteMean2[i, 2] = vals[0][voiceType[i]][0]
        noteCovar2[i, 0] = vals[1][voiceType[i]][1]
        noteCovar2[i, 1] = vals[1][voiceType[i]][1]
        noteCovar2[i, 2] = vals[0][voiceType[i]][1]
    
    # Generate versions using itertools combinations
    for nVoice in range(1, numVoice + 1):
        versions[nVoice] = list(itertools.combinations(range(numVoice), nVoice))
    
    meansSeed = {}
    covarsSeed = {}
    
    for nVoice, version_combinations in versions.items():
        meansSeed[nVoice] = {}
        covarsSeed[nVoice] = {}
        
        for iVer, version in enumerate(version_combinations):
            nMean1 = noteMean1[list(version), :]
            nMean2 = noteMean2[list(version), :]
            nVar1 = noteCovar1[list(version), :]
            nVar2 = noteCovar2[list(version), :]
            
            notes2 = np.vstack([np.array(notes[nVoice][v]) for v in version])
            
            meansSeed[nVoice][iVer] = np.zeros((2 * nVoice, len(notes2)))
            covarsSeed[nVoice][iVer] = np.zeros((2 * nVoice, 2 * nVoice, len(notes2)))
            
            for v in range(nVoice):
                meansSeed[nVoice][iVer][2 * v, :] = nMean1[v, notes2[v, :]]
                meansSeed[nVoice][iVer][2 * v + 1, :] = nMean2[v, notes2[v, :]]
                covarsSeed[nVoice][iVer][2 * v, 2 * v, :] = nVar1[v, notes2[v, :]]
                covarsSeed[nVoice][iVer][2 * v + 1, 2 * v + 1, :] = nVar2[v, notes2[v, :]]
    
    return meansSeed, covarsSeed, versions
