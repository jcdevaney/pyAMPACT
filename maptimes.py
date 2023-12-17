import numpy as np

def maptimes(t, intime, outtime):
    # map the times in t according to the mapping that each point 
    # in intime corresponds to that value in outtime
    # 2008-03-20 Dan Ellis dpwe@ee.columbia.edu

    tr, tc = t.shape
    t = t.flatten()  # make into a row
    nt = len(t)
    nr = len(intime)

    # Decidedly faster than outer-product-array way
    u = t.copy()
    for i in range(nt):
        u[i] = outtime[min([np.argmax(intime > t[i]), len(outtime)-1])]

    u = u.reshape(tr, tc)
    return u
