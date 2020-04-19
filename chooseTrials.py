from kassdirs import resFN, datFN
import numpy as _N
import matplotlib.pyplot as _plt


def chooseTrials(setname, chosen, scopyNum, N):
    dat = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))

    cdat = _N.empty((N, 4*len(chosen)))

    phs  = []
    for ct in xrange(len(chosen)):
        cdat[:, 4*ct]   = dat[:, 4*chosen[ct]]
        cdat[:, 4*ct+1] = dat[:, 4*chosen[ct]+1]
        cdat[:, 4*ct+2] = dat[:, 4*chosen[ct]+2]
        cdat[:, 4*ct+3] = dat[:, 4*chosen[ct]+3]
        ts = _N.where(cdat[:, 2+4*ct] == 1)[0]
        phs.extend(cdat[ts, 3+4*ct])

    csetname = "%(sn)s-%(cn)s" % {"sn" : setname, "cn" : scopyNum}
    _N.savetxt(resFN("xprbsdN.dat", dir=csetname, create=True), cdat, fmt=("% .1f % .1f %d %.3f  " * len(chosen)))

    _N.savetxt(resFN("chosen.dat", dir=csetname, create=True), _N.array(chosen), fmt="%d")

    fig = _plt.figure(figsize=(13, 4))
    _plt.plot(_N.sum(cdat[:, 2::4], axis=0), marker=".", ms=10)
    _plt.xticks()
    _plt.grid()
    _plt.savefig(resFN("spksPtrl", dir=csetname, create=True))
    _plt.close()

    cols, TR4 = cdat.shape
    TR        = TR4 / 4

    isis   = []
    for tr in xrange(TR):
        sts = _N.where(dat[:, 2+tr*4] == 1)
        isis.extend(_N.diff(sts[0]).tolist())

    fig = _plt.figure(figsize=(4, 4))
    _plt.hist(isis, bins=_N.linspace(0, 600, 301))
    _plt.grid()
    _plt.yscale("log")
    _plt.savefig(resFN("ISI-log-scale", dir=csetname))
    _plt.close()

