import itertools
import numpy as np

def genMeansCovars(notes, vals, voiceType):
    numVoice = len(voiceType)

    noteMean1 = np.zeros((numVoice, 3))
    noteCovar1 = np.zeros((numVoice, 3))
    noteMean2 = np.zeros((numVoice, 3))
    noteCovar2 = np.zeros((numVoice, 3))

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
        noteCovar2[i, 0] = vals[2][voiceType[i]][1]
        noteCovar2[i, 1] = vals[2][voiceType[i]][1]
        noteCovar2[i, 2] = vals[1][voiceType[i]][1]

    versions = [list(comb) for i in range(1, numVoice + 1) for comb in itertools.combinations(range(numVoice), i)]

    meansSeed = {}
    covarsSeed = {}

    for nVoice in range(1, numVoice + 1):
        meansSeed[nVoice] = {}
        covarsSeed[nVoice] = {}
        for iVer in range(len(versions)):
            nMean1 = noteMean1[versions[iVer], :]
            nMean2 = noteMean2[versions[iVer], :]
            nVar1 = noteCovar1[versions[iVer], :]
            nVar2 = noteCovar2[versions[iVer], :]
            notes2 = np.vstack(notes[nVoice - 1]).T
            meansSeed[nVoice][iVer + 1] = np.zeros((2 * nVoice, len(notes2[0])))
            covarsSeed[nVoice][iVer + 1] = np.zeros((2 * nVoice, 2 * nVoice, len(notes2[0])))
            for v in range(nVoice):
                meansSeed[nVoice][iVer + 1][2 * v, :] = nMean1[v, notes2[v, :]]
                meansSeed[nVoice][iVer + 1][2 * v + 1, :] = nMean2[v, notes2[v, :]]
                covarsSeed[nVoice][iVer + 1][2 * v, 2 * v, :] = nVar1[v, notes2[v, :]]
                covarsSeed[nVoice][iVer + 1][2 * v + 1, 2 * v + 1, :] = nVar2[v, notes2[v, :]]

    return meansSeed, covarsSeed, versions
