from kassdirs import resFN
import scipy.signal as _ssig
import re as _re
from filter import lpFilt, bpFilt, base_q4atan, gauKer
import numpy as _N
import matplotlib.pyplot as _plt
import random as _ran
import myColors as mC
import itertools as itls

#  [3.3, 11, 1, 15]
def modhistAll(setname, shftPhase=0, fltPrms=[3.3, 11, 1, 15], t0=None, t1=None, tr0=0, tr1=None, trials=None, fn=None, maxY=None, yticks=None, normed=False, surrogates=1, color=None, nofig=False, flatten=False, filtCol=0, xlabel=None, smFnt=20, bgFnt=22):
    """
    shftPhase from 0 to 1.  
    yticks should look like [[0.5, 1, 1.5], ["0.5", "1", "1.5"]]
    """

    _dat     = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
    #  modulation histogram.  phase @ spike
    Na, cols = _dat.shape

    t0 = 0 if (t0 is None) else t0
    t1 = Na if (t1 is None) else t1

    dat = _dat[t0:t1, :]
    N   = t1-t0

    p = _re.compile("^\d{6}")   # starts like "exptDate-....."
    m = p.match(setname)

    bRealDat, COLS, sub, phC = True, 4, 2, 3
    
    if m == None:
        bRealDat, COLS, sub = False, 3, 1

    print "realDat is %s" % str(bRealDat)

    TR   = cols / COLS
    tr1 = TR if (tr1 is None) else tr1

    trials = _N.arange(tr0, tr1) if (trials is None) else trials
    if type(trials) == list:
        trials = _N.array(trials)

    phs  = []
    cSpkPhs= []
    sSpkPhs= []

    for tr in trials:
        if fltPrms is not None:
            x   = dat[:, tr*COLS]
            if len(fltPrms) == 2:
                fx = lpFilt(fltPrms[0], fltPrms[1], 500, x)
            elif len(fltPrms) == 4: 
                # 20, 40, 10, 55 #(fpL, fpH, fsL, fsH, nyqf, y):
                fx = bpFilt(fltPrms[0], fltPrms[1], fltPrms[2], fltPrms[3], 500, x)
        else:
            fx   = dat[:, tr*COLS+filtCol]

        ht_x  = _ssig.hilbert(fx)
        ph_x  = (_N.arctan2(ht_x.imag, ht_x.real) + _N.pi) / (2*_N.pi)
        ph_x  = _N.mod(ph_x + shftPhase, 1)

        ispks  = _N.where(dat[:, tr*COLS+(COLS-sub)] == 1)[0]
        cSpkPhs.append(_N.cos(2*_N.pi*ph_x[ispks]))
        sSpkPhs.append(_N.sin(2*_N.pi*ph_x[ispks]))
        phs.append(ph_x[ispks])

    if nofig:
        if not flatten:
            return phs
        else:
            fl = []
            for i in xrange(len(phs)):
                fl.extend(phs[i])
            return fl
    return figCircularDistribution(phs, cSpkPhs, sSpkPhs, trials, surrogates=surrogates, normed=normed, fn=fn, maxY=maxY, yticks=yticks, setname=setname, color=color, xlabel=xlabel, smFnt=smFnt, bgFnt=bgFnt)

def histPhase0_phaseInfrdAll(mARp, _mdn, t0=None, t1=None, bRealDat=False, trials=None, fltPrms=None, maxY=None, yticks=None, fn=None, normed=False, surrogates=1, shftPhase=0, color=None):
    if not bRealDat:
        return _histPhase0_phaseInfrdAll(mARp.TR, mARp.N+1, mARp.x, _mdn, t0=None, t1=None, bRealDat=False, trials=None, fltPrms=None, maxY=None, yticks=None, fn=None, normed=False, surrogates=1, shftPhase=0, color=color)
    else:   #  REAL DAT
        return _histPhase0_phaseInfrdAll(mARp.TR, mARp.N+1, mARp.fx, _mdn, t0=None, t1=None, bRealDat=False, trials=None, fltPrms=None, maxY=None, yticks=None, fn=None, normed=False, surrogates=1, shftPhase=0, color=color)

def _histPhase0_phaseInfrdAll(TR, N, x, _mdn, t0=None, t1=None, bRealDat=False, trials=None, fltPrms=None, maxY=None, yticks=None, fn=None, normed=False, surrogates=1, shftPhase=0, color=None):
    """
    what is the inferred phase when ground truth phase is 0
    """
    pInfrdAt0 = []

    if (fltPrms is not None) and (not bRealDat):
        _fx = _N.empty((TR, N))
        for tr in xrange(TR):
            if len(fltPrms) == 2:
                _fx[tr] = lpFilt(fltPrms[0], fltPrms[1], 500, x[tr])
            elif len(fltPrms) == 4: 
                # 20, 40, 10, 55 #(fpL, fpH, fsL, fsH, nyqf, y):
                _fx[tr] = bpFilt(fltPrms[0], fltPrms[1], fltPrms[2], fltPrms[3], 500, x[tr])
    else:
        _fx  = x

    gk  = gauKer(1) 
    gk /= _N.sum(gk)

    if trials is None:
        trials = _N.arange(TR)
        TR     = TR
    else:
        TR     = len(trials)
        trials = _N.array(trials)

    #trials, TR = range(mARp.TR), mARp.TR if (trials is None) else trials, len(trials)

    nPh0s = _N.zeros(TR)
    t1    = t1-t0   #  mdn already size t1-t0
    t0    = 0

    mdn = _mdn
    fx  = _fx
    if _mdn.shape[0] != t1 - t0:
        mdn = _mdn[:, t0:t1]
    if _fx.shape[0] != t1 - t0:
        fx = _fx[:, t0:t1]

    itr   = 0

    phs  = []   #  phase 0 of inferred is at what phase of GT or LFP?
    cSpkPhs= []
    sSpkPhs= []

    for tr in trials:
        itr += 1
        cv = _N.convolve(mdn[tr, t0:t1] - _N.mean(mdn[tr, t0:t1]), gk, mode="same")

        ht_mdn  = _ssig.hilbert(cv)
        ht_fx   = _ssig.hilbert(fx[tr, t0:t1] - _N.mean(fx[tr, t0:t1]))
        ph_mdn  = (_N.arctan2(ht_mdn.imag, ht_mdn.real) + _N.pi) / (2*_N.pi)
        ph_mdn  = _N.mod(ph_mdn + shftPhase, 1)
        ph_fx  = (_N.arctan2(ht_fx.imag, ht_fx.real) + _N.pi) / (2*_N.pi)
        ph_fx  = _N.mod(ph_fx + shftPhase, 1)
        #  phase = 0 is somewhere in middle

        inds = _N.where((ph_mdn[0:t1-t0-1] < 1) & (ph_mdn[0:t1-t0-1] > 0.5) & (ph_mdn[1:t1-t0] < 0.25))[0]
        cSpkPhs.append(_N.cos(2*_N.pi*ph_fx[inds+t0]))
        sSpkPhs.append(_N.sin(2*_N.pi*ph_fx[inds+t0]))
        phs.append(ph_fx[inds+t0])
        
        #for i in xrange(t0-t0, t1-t0-1):
        #    if (ph_mdn[i] < 1) and (ph_mdn[i] > 0.5) and (ph_mdn[i+1] < -0.5):
        #        pInfrdAt0.append(ph_fx[i]/2.)

    return figCircularDistribution(phs, cSpkPhs, sSpkPhs, trials, surrogates=surrogates, normed=normed, fn=fn, maxY=maxY, yticks=yticks, color=color, xlabel=xlabel)

def getPhases(_mdn, offset=0):
    """
    what is the inferred phase when ground truth phase is 0
    """
    TR = _mdn.shape[0]
    ph = _N.array(_mdn)

    gk  = gauKer(2) 
    gk /= _N.sum(gk)


    mdn = _mdn
    itr   = 0

    for tr in xrange(TR):
        itr += 1
        cv = _N.convolve(mdn[tr] - _N.mean(mdn[tr]), gk, mode="same")

        ht_mdn  = _ssig.hilbert(cv)
        ph[tr]  = (_N.arctan2(ht_mdn.imag, ht_mdn.real) + _N.pi) / (2*_N.pi)
        if offset != 0:
            ph[tr] += offset
            inds = _N.where(ph[tr] >= 1)[0]
            ph[tr, inds] -= 1

    return ph

def figCircularDistribution(phs, cSpkPhs, sSpkPhs, trials, setname=None, surrogates=1, normed=False, fn=None, maxY=None, yticks=None, color=None, smFnt=20, bgFnt=22, xlabel="phase"):  #  phase histogram
    ltr = len(trials)
    inorderTrials = _N.arange(ltr)   #  original trial IDs no lnger necessary
    R2s = _N.empty(surrogates)
    for srgt in xrange(surrogates):
        if srgt == 0:
            trls = inorderTrials
        else:
            trls = inorderTrials[_N.sort(_N.asarray(_N.random.rand(ltr)*ltr, _N.int))]

        cS = []
        sS = []
        for tr in trls:
            cS.extend(cSpkPhs[tr])
            sS.extend(sSpkPhs[tr])

        Nspks = len(cS)
        vcSpkPhs = _N.array(cS)
        vsSpkPhs = _N.array(sS)

        R2s[srgt]  = _N.sqrt((1./(Nspks*Nspks)) * (_N.sum(vcSpkPhs)**2 + _N.sum(vsSpkPhs)**2))


    vPhs  = _N.fromiter(itls.chain.from_iterable(phs), _N.float)
    fig, ax = _plt.subplots(figsize=(6, 4.2))
    if color is None:
        ec = mC.hist1
    else:
        ec = color
    _plt.hist(vPhs.tolist() + (vPhs + 1).tolist(), bins=_N.linspace(0, 2, 51), color=ec, edgecolor=ec, normed=normed)

    if maxY is not None:
        _plt.ylim(0, maxY)
    elif normed:
        _plt.ylim(0, 1)
    #_plt.title("R = %.3f" % _N.sqrt(R2), fontsize=smFnt)
    _plt.xlabel(xlabel, fontsize=bgFnt)
    _plt.ylabel("prob. density", fontsize=bgFnt)
    _plt.xticks(fontsize=smFnt)
    _plt.yticks(fontsize=smFnt)
    if yticks is not None:
        #  plotting 2 periods, probability is halved, so boost it by 2
        _plt.yticks(yticks[0], yticks[1])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].axis.axes.tick_params(direction="outward", width=2)
    ax.spines["bottom"].axis.axes.tick_params(direction="outward", width=2)

    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    #    for tic in ax.xaxis.get_major_ticks():
    #   tic.tick1On = tic.tick2On = False

    fig.subplots_adjust(left=0.17, bottom=0.19, right=0.95, top=0.92)

    if fn is None:
        fn = "modulationHistogram,R=%.3f.eps" % R2s[0]
    else:
        fn = "%(1)s,R=%(2).3f.eps" % {"1" : fn, "2" : R2s[0]}
        
    if setname is not None:
        _plt.savefig(resFN(fn, dir=setname), transparent=True)
    else:
        _plt.savefig(fn, transparent=True)
    _plt.close()
    return R2s

def oscPer(setname, fltPrms=[5, 13, 1, 20], t0=None, t1=None, tr0=0, tr1=None, trials=None, fn=None, showHist=True, osc=None):
    """
    find period of oscillation
    """

    if t0 is None:
        t0 = 0
    if t1 is None:
        t1 = Na

    if setname is not None:
        _dat     = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))

        #  modulation histogram.  phase @ spike
        Na, cols = _dat.shape

        p = _re.compile("^\d{6}")   # starts like "exptDate-....."
        m = p.match(setname)

        bRealDat = True
        COLS = 4
        sub  = 1

        if m == None:
            bRealDat = False
            COLS = 3
            sub  = 0

        TR   = cols / COLS
        if trials is None:
            if tr1 is None:
                tr1 = TR
            trials = _N.arange(tr0, tr1)

        N   = t1-t0
        dat = _dat[t0:t1, :]
    elif  osc is not None:
        TR, N  = osc.shape
        trials = _N.arange(TR)

        bRealDat = False
        COLS     = 3
        sub      = 0
        dat      = _N.empty((N, COLS*TR))
        dat[:, sub::COLS]      = osc.T

    Ts = []
    for tr in trials:
        x   = dat[:, tr*COLS+sub]

        if fltPrms is None:
            fx = x
        elif len(fltPrms) == 2:
            fx = lpFilt(fltPrms[0], fltPrms[1], 500, x)
        else: # 20, 40, 10, 55    #(fpL, fpH, fsL, fsH, nyqf, y):
            fx = bpFilt(fltPrms[0], fltPrms[1], fltPrms[2], fltPrms[3], 500, x)

        intvs = _N.where((fx[1:] < 0) & (fx[0:-1] >= 0))

        Ts.extend(_N.diff(intvs[0]))

    if showHist:
        fig = _plt.figure()
        _plt.hist(Ts, bins=range(min(Ts) - 1, max(Ts)+1))
    mn = _N.mean(Ts)
    std= _N.std(Ts)
    print "mean Hz %(f).3f    cv: %(cv).3f" % {"cv" : (std/mn), "f" : (1000/mn)}
