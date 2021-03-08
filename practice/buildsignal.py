import numpy as _N
import numpy.polynomial.polynomial as _Npp
from LOST.kflib import createDataAR

def build_signal(N, comps, wgts, innovation_var=0.01, normalize=True):
    dt = 0.001
    nComps = len(comps)
    obsvd = _N.zeros((N, nComps+1))
    ic = -1

    for comp in comps:
        ic += 1

        print(comp)
        r_comp = comp[1]
        osc_comp = comp[0]

        l_alfa = []
        for thr in osc_comp:
            r = thr[1]
            hz= thr[0]
            th = 2*hz*dt

            l_alfa.append(r*(_N.cos(_N.pi*th) + 1j*_N.sin(_N.pi*th)))
            l_alfa.append(r*(_N.cos(_N.pi*th) - 1j*_N.sin(_N.pi*th)))
        for r in r_comp:
            l_alfa.append(r)

        alfa  = _N.array(l_alfa)
        ARcoeff          = (-1*_Npp.polyfromroots(alfa)[::-1][1:]).real
        sgnlC, y = createDataAR(N, ARcoeff, innovation_var, 0)
        obsvd[:, ic] = wgts[ic]*sgnlC

        obsvd[:, nComps] += obsvd[:, ic]

    if normalize:
        for ic in range(nComps):
            obsvd[:, ic] /= _N.std(obsvd[:, nComps])
        obsvd[:, nComps] /= _N.std(obsvd[:, nComps])

    if len(comps) == 1:
        return ARcoeff, obsvd
    else:
        return obsvd
