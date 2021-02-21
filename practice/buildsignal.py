import numpy as _N
import numpy.polynomial.polynomial as _Npp
from LOST.kflib import createDataAR

def build_signal(N, comps, wgts):
    dt = 0.001
    obsvd = _N.zeros(N)
    ic = -1

    for comp in comps:
        ic += 1
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
            print(r)
            l_alfa.append(r)

        alfa  = _N.array(l_alfa)
        ARcoeff          = (-1*_Npp.polyfromroots(alfa)[::-1][1:]).real
        sgnlC, y = createDataAR(N, ARcoeff, 0.000001, 0.00001)
        obsvd += wgts[ic] * sgnlC

    return obsvd / _N.std(obsvd)
