#  compare glm CIF vs. AR latent state.  Does the CIF represent the oscillation
#  well?

#  save oscMn from smplLatent
#  save est.params, X, from glm, LHbin, 
import filter as _flt
import numpy as _N
import scipy.stats as _ss
import matplotlib.pyplot as _plt
import mcmcFigs as mF
import myColors as myC

def compare(mARp, est, X, spkHist, oscMn, dat, gkW=20, useRefr=True):
    dt     = 0.001

    gk = _flt.gauKer(gkW)
    gk /= _N.sum(gk)

    TR  = oscMn.shape[0]

    #  params, stT
    params = _N.array(est.params)

    stT = spkHist.LHbin * (spkHist.nLHBins + 1)    #  first stT spikes used for initial history
    ocifs  = _N.empty((spkHist.endTR - spkHist.startTR, spkHist.t1-spkHist.t0 - stT))
    ##  

    sur = "refr"
    if not useRefr:
        params[spkHist.endTR:spkHist.endTR+spkHist.LHbin] = params[spkHist.endTR+spkHist.LHbin]
        sur = "NOrefr"

    for tr in xrange(spkHist.endTR - spkHist.startTR):
        ocifs[tr] = _N.exp(_N.dot(X[tr], params)) / dt

    cglmAll = _N.zeros((TR, mARp.N+1))
    infrdAll = _N.zeros((TR, mARp.N+1))
    xt   = _N.arange(stT, mARp.N+1)

    for tr in xrange(spkHist.startTR, TR):
        _gt = dat[stT:, tr*3]
        gt = _N.convolve(_gt, gk, mode="same")
        gt /= _N.std(gt)

        glm = (ocifs[tr] - _N.mean(ocifs[tr])) / _N.std(ocifs[tr])
        cglm = _N.convolve(glm, gk, mode="same")
        cglm /= _N.std(cglm)

        infrd = oscMn[tr, stT:] / _N.std(oscMn[tr, stT:])
        infrd /= _N.std(infrd)

        pc1, pv1 = _ss.pearsonr(glm, gt)
        pc1c, pv1c = _ss.pearsonr(cglm, gt)
        pc2, pv2 = _ss.pearsonr(infrd, gt)

        cglmAll[tr, stT:] = cglm
        infrdAll[tr, stT:] = infrd
        
        fig = _plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 1, 1)
        _plt.plot(xt, infrd, color=myC.infrdM, lw=2)
        _plt.plot(xt, cglm, color=myC.infrdM, lw=2., ls="--")
        #_plt.plot(xt, glm, color=myC.infrdM, lw=2., ls="-.")
        _plt.plot(xt, gt, color=myC.grndTruth, lw=4)

        MINx = _N.min(infrd)
        MAXx = _N.max(infrd)

        AMP  = MAXx - MINx
        ht   = 0.08*AMP
        ys1  = MINx - 0.5*ht
        ys2  = MINx - 3*ht

        for n in xrange(stT, mARp.N+1):
            if mARp.y[tr, n] == 1:
                _plt.plot([n, n], [ys1, ys2], lw=2.5, color="black")
        _plt.ylim(ys2 - 0.05*AMP, MAXx + 0.05*AMP)

        _plt.xlim(stT, mARp.N+1)
        mF.arbitraryAxes(ax, axesVis=[False, False, False, False], xtpos="bottom", ytpos="none")
        mF.setLabelTicks(_plt, yticks=[], yticksDsp=None, xlabel="time (ms)", ylabel=None, xtickFntSz=24, xlabFntSz=26)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.85)
        _plt.savefig("cmpGLMAR_%(ur)s_%(tr)d.eps" % {"tr" : tr, "ur" : sur})
        _plt.close()


        corrs[tr] = pc1, pc1c, pc2
    mF.histPhase0_phaseInfrd(mARp, cglmAll, t0=stT, t1=(mARp.N+1), bRealDat=False, normed=True, maxY=1.8, fn="smthdGLMPhaseGLM%s" % sur)
    mF.histPhase0_phaseInfrd(mARp, infrdAll, t0=stT, t1=(mARp.N+1), bRealDat=False, normed=True, maxY=1.8, fn="smthdGLMPhaseInfrd")

    print _N.mean(corrs[:, 0])
    print _N.mean(corrs[:, 1])
    print _N.mean(corrs[:, 2])

    fig = _plt.figure(figsize=(8, 3.5))
    ax  = fig.add_subplot(1, 2, 1)
    _plt.hist(corrs[:, 1], bins=_N.linspace(-0.5, max(corrs[:, 2])*1.05, 30), color=myC.hist1)
    mF.bottomLeftAxes(ax)
    ax  = fig.add_subplot(1, 2, 2)
    _plt.hist(corrs[:, 2], bins=_N.linspace(-0.5, max(corrs[:, 2])*1.05, 30), color=myC.hist1)
    mF.bottomLeftAxes(ax)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.88, wspace=0.2, hspace=0.2)

    _plt.savefig("cmpGLMAR_hist")
    _plt.close()


def getGLMphases(TR, t0, t1, est, X, spkHist, dat, gkW=20, useRefr=True):
    params = _N.array(est.params)

    stT = spkHist.LHbin * (spkHist.nLHBins + 1)    #  first stT spikes used for initial history
    ocifs  = _N.empty((spkHist.endTR - spkHist.startTR, spkHist.t1-spkHist.t0 - stT))
    dt     = 0.001

    ##  

    sur = "refr"
    if not useRefr:
        params[spkHist.endTR:spkHist.endTR+spkHist.LHbin] = params[spkHist.endTR+spkHist.LHbin]
        sur = "NOrefr"

    for tr in xrange(spkHist.endTR - spkHist.startTR):
        ocifs[tr] = _N.exp(_N.dot(X[tr], params)) / dt

    gk = _flt.gauKer(gkW)
    gk /= _N.sum(gk)

    cglmAll = _N.zeros((TR, t1-t0))

    for tr in xrange(spkHist.startTR, TR):  #  spkHist.statTR usually 0
        _gt = dat[stT:, tr*3]
        gt = _N.convolve(_gt, gk, mode="same")
        gt /= _N.std(gt)

        glm = (ocifs[tr] - _N.mean(ocifs[tr])) / _N.std(ocifs[tr])
        cglm = _N.convolve(glm, gk, mode="same")
        cglm /= _N.std(cglm)

        cglmAll[tr, stT:] = cglm

    return stT, cglmAll

def compareWF(mARp, ests, Xs, spkHists, oscMn, dat, gkW=20, useRefr=True, dspW=None):
    """
    instead of subplots, plot 3 different things with 3 largely separated y-values
    """
    
    
    glmsets = len(ests)     #  horrible hack
    TR  = oscMn.shape[0]
    infrdAll = _N.zeros((TR, mARp.N+1))
    dt     = 0.001
    gk = _flt.gauKer(gkW)
    gk /= _N.sum(gk)

    paramss = [];    ocifss = [];    stTs   = []

    for gs in xrange(glmsets):  #  for the 2 glm conditions
        params = _N.array(ests[gs].params)
        X      = Xs[gs]
        spkHist = spkHists[gs]

        stTs.append(spkHist.LHbin * (spkHist.nLHBins + 1))    #  first stT spikes used for initial history
        ocifss.append(_N.empty((spkHist.endTR - spkHist.startTR, spkHist.t1-spkHist.t0 - stTs[gs])))

        for tr in xrange(spkHist.endTR - spkHist.startTR):
            ocifss[gs][tr] = _N.exp(_N.dot(X[tr], params)) / dt

    stT = min(stTs)

    #cglmAll = _N.zeros((TR, mARp.N+1))

    xt = _N.arange(stT, mARp.N+1)
    xts = [_N.arange(stTs[0], mARp.N+1), _N.arange(stTs[1], mARp.N+1)]
    lss  = [":", "-"]
    lws  = [3.8, 2]
    cls  = [myC.infrdM]
    for tr in xrange(spkHist.startTR, TR):
        _gt = dat[stT:, tr*3]
        gt = _N.convolve(_gt, gk, mode="same")
        gt /= _N.std(gt)

        infrd = oscMn[tr, stT:] / _N.std(oscMn[tr, stT:])
        infrd /= _N.std(infrd)
        infrdAll[tr, stT:] = infrd

        fig = _plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        _plt.plot(xt, gt, color=myC.grndTruth, lw=4)
        #_plt.plot(xt, infrd, color="brown", lw=4)

        up1 = _N.max(gt) - _N.min(gt)
        ##  mirror
        _plt.plot(xt, gt + up1*1.25, color=myC.grndTruth, lw=4)
        _plt.plot(xt, infrd+up1*1.25, color=myC.infrdM, lw=2)

        for gs in xrange(glmsets):
            ocifs = ocifss[gs]
            glm = (ocifs[tr] - _N.mean(ocifs[tr])) / _N.std(ocifs[tr])
            cglm = _N.convolve(glm, gk, mode="same")
            cglm /= _N.std(cglm)

            _plt.plot(xts[gs], cglm, color=myC.infrdM, lw=lws[gs], ls=lss[gs])

        MINx = _N.min(infrd)
        #MAXx = _N.max(infrd)
        MAXx = _N.max(gt)+up1*1.35

        AMP  = MAXx - MINx
        ht   = 0.08*AMP
        ys1  = MINx - 0.5*ht
        ys2  = MINx - 3*ht

        for n in xrange(stT, mARp.N+1):
            if mARp.y[tr, n] == 1:
                _plt.plot([n, n], [ys1, ys2], lw=2.5, color="black")
        _plt.ylim(ys2 - 0.05*AMP, MAXx + 0.05*AMP)

        if dspW is None:
            _plt.xlim(stT, mARp.N+1)
        else:
            _plt.xlim(dspW[0], dspW[1])
        mF.arbitraryAxes(ax, axesVis=[False, False, False, False], xtpos="bottom", ytpos="none")
        mF.setLabelTicks(_plt, yticks=[], yticksDsp=None, xlabel="time (ms)", ylabel=None, xtickFntSz=24, xlabFntSz=26)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.85)
        _plt.savefig("cmpGLMAR_%(tr)d.eps" % {"tr" : tr}, transparent=True)
        _plt.close()


