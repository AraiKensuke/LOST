import numpy.polynomial.polynomial as _Npp
from mcmcARpPlot import plotWFandSpks
from shutil import copyfile
from utildirs import setFN
from kassdirs import resFN
import utilities as _U
from kflib import createDataAR
import numpy as _N
import matplotlib.pyplot as _plt

setname="LIF-"
#  dV = -V/(RC) + I/C
#  dV = -V/tau + I/C  = -V/tau + eps I

TR  = 60;   N   = 1000;   dt  = 0.001     #  1ms
trim=0;

tau = .2      #  ms.  time constant = 1/RC
bcksig =210   #  background firing 
rst   = 0
thr   = 1

lowQpc     = 0;        lowQs    = None
isis       = None;     rpsth    = None
us         = None

rs         = None;     ths      = None;      alfa     = None;

errH  = 0.004
errL  = 0.00001
obsnz= 1e-1

def create(setname):
    global lowQs, lowQpc
    copyfile("%s.py" % setname, "%(s)s/%(s)s.py" % {"s" : setname, "to" : setFN("%s.py" % setname, dir=setname, create=True)})

    ARcoeff = _N.empty((nRhythms, 2))
    for n in xrange(nRhythms):
        ARcoeff[n]          = (-1*_Npp.polyfromroots(alfa[n])[::-1][1:]).real

    V     = _N.empty(N)

    dV    = _N.empty(N)
    dN    = _N.zeros(N)

    xprbsdN= _N.empty((N, 3*TR))
    isis  = []
    lowQs = []

    spksPT= _N.empty(TR)
    for tr in xrange(TR):
        V[0]  = 0.2*_N.random.randn()
        eps = bcksig*_N.random.randn(N)   # time series
        err = nzs[0, 1]

        if _N.random.rand() < lowQpc:
            err = nzs[0, 0]
            lowQs.append(tr)
        sTs   = []

        x, y= createDataAR(N, ARcoeff[0], err, obsnz, trim=trim)
        dN[:] = 0
        for n in xrange(N-1):
            dV[n] = -V[n] / tau + eps[n] + Is[tr] + x[n] + psth[n]
            V[n+1] = V[n] + dV[n]*dt

            if V[n+1] > thr:
                V[n + 1] = rst
                dN[n] = 1
                sTs.append(n)
        spksPT[tr] = len(sTs)

        xprbsdN[:, tr*3]     = x
        xprbsdN[:, tr*3 + 1] = V
        xprbsdN[:, tr*3 + 2] = dN
        isis.extend(_U.toISI([sTs])[0])

    fmt = "% .2e %.3f %d " * TR
    _N.savetxt(resFN("xprbsdN.dat", dir=setname, create=True), xprbsdN, fmt=fmt)

    plotWFandSpks(N-1, dN, [x], sTitle="AR2 freq %(f).1fHz  num spks %(d).0f   spk Hz %(spkf).1fHz" % {"f" : (500*ths[0]/_N.pi), "d" : _N.sum(dN), "spkf" : (_N.sum(dN) / (N*0.001))}, sFilename=resFN("generative", dir=setname))
    #plotWFandSpks(N-1, dN, [x], sTitle="AR2 freq %(f).1fHz  num spks %(d).0f   spk Hz %(spkf).1fHz" % {"f" : (500*ths[0]), "d" : _N.sum(dN), "spkf" : (_N.sum(dN) / (N*0.001))}, sFilename=resFN("generative", dir=setname))



    fig = _plt.figure(figsize=(8, 4))
    _plt.hist(isis, bins=range(100), color="black")
    _plt.grid()
    _plt.savefig(resFN("ISIhist", dir=setname))
    _plt.close()

    cv = _N.std(spksPT)**2/_N.mean(spksPT)
    fig = _plt.figure(figsize=(13, 4))
    _plt.plot(spksPT, marker=".", color="black", ms=8)
    _plt.ylim(0, max(spksPT)*1.1)
    _plt.grid()
    _plt.suptitle("avg. Hz %(hz).1f   cv=%(cv).2f" % {"hz" : (_N.mean(spksPT) / (N*0.001)), "cv" : cv})
    _plt.savefig(resFN("spksPT", dir=setname))
    _plt.close()

    """
    cv   = _N.std(isis) / _N.mean(isis)
    fig  = _plt.figure(figsize=(7, 3.5))
    _plt.hist(isis, bins=range(0, 50), color="black")
    _plt.grid()
    _plt.suptitle("ISI cv %.2f" % cv)
    _plt.savefig(resFN("ISIhist", dir=setname))
    _plt.close()
    """
