import statsmodels.api as _sm
import numpy as _N
import matplotlib.pyplot as _plt
from kassdirs import resFN, datFN

class spikeHist:
    LHbin       = 10  # bin sizes for long history
    nLHBins     = 16  #  (nLHBins+1) x oo.LHbin  is total history
    startTR     = 0
    endTR       = 0
    t0          = 0
    t1          = 0
    COLS        = 3
    setname     = None
    dat         = None

    def __init__(self, setname, COLS=3):
        self.setname = setname
        self.dat = _N.loadtxt(resFN("xprbsdN.dat", dir=self.setname))
        self.COLS = COLS

    def fitGLM(self):
        oo = self

        N, TR = oo.dat.shape
        if oo.t1 - oo.t0 > N:
            print "ERROR  t1-t0 > N"
            return
        if oo.endTR - oo.startTR > TR:
            print "ERROR  endTR-startTR > TR"
            return

        st = oo.dat[oo.t0:oo.t1, oo.COLS*oo.startTR+2:oo.COLS*oo.endTR+2:oo.COLS]

        N  = oo.t1-oo.t0
        TR = oo.endTR-oo.startTR

        #  The design matrix
        #  # of params LHBin + nLHBins + 1

        Ldf = N - oo.LHbin*(oo.nLHBins+1)
        X  = _N.empty((TR, Ldf, oo.LHbin + oo.nLHBins + 1))
        X[:, :, 0] = 1  #  offset
        y  = _N.empty((TR, Ldf))

        for tr in xrange(TR):
            for t in xrange(oo.LHbin*(oo.nLHBins+1), N):
                #  0:9
                hist = st[t-oo.LHbin*(oo.nLHBins+1):t, tr][::-1]

                sthcts  = hist[0:oo.LHbin]   #  
                lthcts  = _N.sum(hist[oo.LHbin:oo.LHbin*(oo.nLHBins+1)].reshape(oo.nLHBins, oo.LHbin), axis=1)
                X[tr, t-oo.LHbin*(oo.nLHBins+1), 1:oo.LHbin+1] = sthcts
                X[tr, t-oo.LHbin*(oo.nLHBins+1), oo.LHbin+1:]  = lthcts
                y[tr, t-oo.LHbin*(oo.nLHBins+1)]            = st[t, tr]


        yr  = y.reshape(TR*Ldf)
        Xr  = X.reshape(TR*Ldf, oo.LHbin + oo.nLHBins + 1)
        est = _sm.GLM(yr, Xr, family=_sm.families.Poisson()).fit()
        oo.offs  = est.params[0]
        oo.shrtH = est.params[1:oo.LHbin+1]
        oo.oscH  = est.params[oo.LHbin+1:]

        cfi = est.conf_int()
        oscCI = cfi[oo.LHbin+1:]

        fig = _plt.figure(figsize=(12, 6))
        xlab = _N.arange(oo.LHbin, (oo.nLHBins+1)*oo.LHbin, oo.LHbin)
        _plt.fill_between(xlab, _N.exp(oscCI[:, 0]), _N.exp(oscCI[:, 1]), color="blue", alpha=0.2)

        _plt.plot(xlab, _N.exp(oo.oscH), lw=2, color="black")
        _plt.xticks(xlab, fontsize=20)
        _plt.yticks(fontsize=20)
        _plt.xlabel("lags (ms)", fontsize=22)
        _plt.xlim(xlab[0], xlab[-1])
        _plt.axhline(y=1, ls="--", color="grey")
        fig.subplots_adjust(left=0.1, bottom=0.15, top=0.94)
        _plt.savefig(resFN("glmfit_LHBins=%(LHBins)d_%(binsz)d_strt=%(trS)d_%(trE)d_t0=%(t0)d_t1=%(t1)d" % {"trS" : oo.startTR, "trE" : oo.endTR, "t0" : oo.t0, "t1" : oo.t1, "LHBins" : oo.nLHBins, "binsz" : oo.LHbin}, dir=oo.setname))
        _plt.close()

        return est, X, y

    def fitGLMwTrlOffset(self):
        oo = self

        N, TR = oo.dat.shape
        if oo.t1 - oo.t0 > N:
            print "ERROR  t1-t0 > N"
            return
        if oo.endTR - oo.startTR > TR:
            print "ERROR  endTR-startTR > TR"
            return

        #  spikes
        st = oo.dat[oo.t0:oo.t1, oo.COLS*oo.startTR+2:oo.COLS*oo.endTR+2:oo.COLS]

        N  = oo.t1-oo.t0
        TR = oo.endTR-oo.startTR

        #  The design matrix
        #  # of params LHBin + nLHBins + 1

        Ldf = N - oo.LHbin*(oo.nLHBins+1)
        X  = _N.zeros((TR, Ldf, oo.LHbin + oo.nLHBins + TR))
        y  = _N.empty((TR, Ldf))

        for tr in xrange(TR):
            X[tr, :, tr] = 1  #  offset
            for t in xrange(oo.LHbin*(oo.nLHBins+1), N):
                #  0:9
                hist = st[t-oo.LHbin*(oo.nLHBins+1):t, tr][::-1]

                #  short Term Hist  (one bin)
                sthcts  = hist[0:oo.LHbin]   
                #  long Term Hist counts (multiple bins)
                lthcts  = _N.sum(hist[oo.LHbin:oo.LHbin*(oo.nLHBins+1)].reshape(oo.nLHBins, oo.LHbin), axis=1)
                X[tr, t-oo.LHbin*(oo.nLHBins+1), TR:oo.LHbin+TR] = sthcts
                X[tr, t-oo.LHbin*(oo.nLHBins+1), oo.LHbin+TR:]  = lthcts
                y[tr, t-oo.LHbin*(oo.nLHBins+1)]            = st[t, tr]

        yr  = y.reshape(TR*Ldf)
        Xr  = X.reshape(TR*Ldf, oo.LHbin + oo.nLHBins + TR)
        est = _sm.GLM(yr, Xr, family=_sm.families.Poisson()).fit()
        oo.offs  = est.params[0:TR]
        oo.shrtH = est.params[TR:oo.LHbin+TR]
        oo.oscH  = est.params[oo.LHbin+TR:]

        cfi = est.conf_int()
        oscCI = cfi[oo.LHbin+TR:]

        """
        fig = _plt.figure(figsize=(12, 6))
        xlab = _N.arange(oo.LHbin, (oo.nLHBins+1)*oo.LHbin, oo.LHbin)
        _plt.fill_between(xlab, _N.exp(oscCI[:, 0]), _N.exp(oscCI[:, 1]), color="blue", alpha=0.2)

        _plt.plot(xlab, _N.exp(oo.oscH), lw=2, color="black")
        _plt.xticks(xlab, fontsize=20)
        _plt.yticks(fontsize=20)
        _plt.xlabel("lags (ms)", fontsize=22)
        _plt.xlim(xlab[0], xlab[-1])
        _plt.axhline(y=1, ls="--", color="grey")
        fig.subplots_adjust(left=0.1, bottom=0.15, top=0.94)
        _plt.savefig(resFN("glmfit_LHBins=%(LHBins)d_%(binsz)d_strt=%(trS)d_%(trE)d_t0=%(t0)d_t1=%(t1)d" % {"trS" : oo.startTR, "trE" : oo.endTR, "t0" : oo.t0, "t1" : oo.t1, "LHBins" : oo.nLHBins, "binsz" : oo.LHbin}, dir=oo.setname))
        _plt.close()
        """
        return est, X, y
