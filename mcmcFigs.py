import numpy as _N
import scipy.signal as _ssig
import scipy.stats as _ss
from ARcfSmplFuncs import dcmpcff
import matplotlib.pyplot as _plt
from filter import gauKer, lpFilt
import scipy.signal as _ssig
import myColors as mC
import modhist2 as mh2
from tmrsclTest import timeRescaleTest, zoom

def histPhase0_phaseInfrd(mARp, _mdn, t0=None, t1=None, bRealDat=False, trials=None, filtParams=None, maxY=None, yticks=None, fn=None, normed=False):
    #  what is the inferred phase when ground truth phase is 0
    pInfrdAt0 = []

    #if (filtParams is not None) and (not bRealDat):
    if not bRealDat:
        _fx = _N.empty((mARp.TR, mARp.N+1))
        for tr in xrange(mARp.TR):
            _fx[tr] = lpFilt(30, 40, 1000, mARp.x[tr])
    else:
        _fx  = mARp.x

    if bRealDat:
        _fx = mARp.fx
    gk  = gauKer(1) 
    gk /= _N.sum(gk)

    if trials is None:
        trials = range(mARp.TR)
        TR     = mARp.TR
    else:
        TR     = len(trials)
    nPh0s = _N.zeros(TR)
    #t1    = t1-t0   #  mdn already size t1-t0
    #t0    = 0

    mdn = _mdn
    fx  = _fx
    #if _mdn.shape[0] != t1 - t0:
    #    mdn = _mdn[:, t0:t1]
    #if _fx.shape[0] != t1 - t0:
    #    fx = _fx[:, t0:t1]

    itr   = 0

    for tr in trials:
        itr += 1
        cv = _N.convolve(mdn[tr, t0:t1] - _N.mean(mdn[tr, t0:t1]), gk, mode="same")
        #cv = mdn[tr, t0:t1] - _N.mean(mdn[tr, t0:t1])

        ht_mdn  = _ssig.hilbert(cv)
        #ht_fx   = _ssig.hilbert(fx[tr, t0:t1] - _N.mean(fx[tr, t0:t1]))
        ht_fx   = _ssig.hilbert(fx[tr, t0:t1] - _N.mean(fx[tr, t0:t1]))
        ph_mdn  = _N.arctan2(ht_mdn.imag, ht_mdn.real) / _N.pi
        ph_fx   = ((_N.arctan2(ht_fx.imag, ht_fx.real)   / _N.pi) + 1)

        #  phase = 0 is somewhere in middle

        for i in xrange(t0-t0, t1-t0-1):
            if (ph_mdn[i] < 1) and (ph_mdn[i] > 0.5) and (ph_mdn[i+1] < -0.5):
                nPh0s[itr-1] += 1
                pInfrdAt0.append(ph_fx[i]/2.)
                pInfrdAt0.append((ph_fx[i]+2)/2.)

    bgFnt = 22
    smFnt = 20

    fig, ax = _plt.subplots(figsize=(6, 4.2))
    pInfrdAt0A = _N.array(pInfrdAt0[::2])#  0 to 2
    Npts = len(pInfrdAt0A)
    R2   = (1./(Npts*Npts)) * (_N.sum(_N.cos(2*_N.pi*pInfrdAt0A))**2 + _N.sum(_N.sin(2*_N.pi*pInfrdAt0A))**2)
    _plt.hist(pInfrdAt0, bins=_N.linspace(0, 2, 41), color=mC.hist1, edgecolor=mC.hist1, normed=normed)
    print "maxY!!!  %f" % maxY
    if (maxY is not None):
        _plt.ylim(0, maxY)
        
    _plt.xlabel("phase", fontsize=bgFnt)
    _plt.ylabel("frequency", fontsize=bgFnt)
    _plt.xticks(fontsize=smFnt)
    if yticks is not None:
        _plt.yticks(yticks)
    _plt.yticks(fontsize=smFnt)
    if yticks is not None:
        _plt.yticks(yticks)
    if normed:
        _plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

    fig.subplots_adjust(left=0.17, bottom=0.17, right=0.95, top=0.9)
    if fn is None:
        fn = "fxPhase1_phaseInfrd,R=%.3f.eps" % _N.sqrt(R2)
    else:
        fn = "%(1)s,R=%(2).3f.eps" % {"1" : fn, "2" : _N.sqrt(R2)}

    _plt.savefig(fn, transparent=True)
    _plt.close()
    return pInfrdAt0, _N.sqrt(R2), nPh0s

def last_fsamps(mARp, tr0, tr1):
    amps    = mARp.amps[tr0:tr1, 0]
    fs      = mARp.fs[tr0:tr1, 0]
    minamps = min(amps)
    maxamps = max(amps)
    minfs   = min(fs)
    maxfs   = max(fs)
    nbins = int((tr1 - tr0)/10.)
    if nbins > 80:
        nbins = 80
    fig = _plt.figure(figsize=(11, 4))
    _plt.subplot(1, 2, 1)
    _plt.hist(fs, bins=_N.linspace(minfs, maxfs, nbins))
    _plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    _plt.subplot(1, 2, 2)
    _plt.hist(amps, bins=_N.linspace(minamps, maxamps, nbins))
    _plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    _plt.title("tr0=%(1)d  tr1=%(2)d" % {"1" : tr0, "2" : tr1})
    _plt.savefig("last_fsamps.eps")
    _plt.close()

def showFsHist(mARp, low, hi, nbins=100):
    t0 = mARp.burn
    t1 = mARp.burn + mARp.NMC
    _plt.hist(mARp.fs[t0:t1, 0], bins=_N.linspace(low, hi, nbins))
    _plt.grid()

def filterByFsAmps(mARp, flow, fhi, alow=0.9, ahigh=1, t0=None, t1=None):
    if t0 is None:
        t0 = mARp.burn
    if t1 is None:
        t1 = mARp.burn + mARp.NMC
    inds = _N.where((mARp.fs[t0:t1, 0] > flow) & (mARp.fs[t0:t1, 0] < fhi) &
                    (mARp.amps[t0:t1, 0] > alow) & (mARp.amps[t0:t1, 0] < ahigh))
    return inds[0] + t0


def getComponents(mARp):
    TR    = mARp.TR
    NMC   = mARp.NMC
    burn  = mARp.burn
    R     = mARp.R
    C     = mARp.C
    ddN   = mARp._d.N

    rt = _N.empty((TR, NMC, ddN+2, R))    #  real components   N = ddN
    zt = _N.empty((TR, NMC, ddN+2, C))    #  imag components 

    for tr in xrange(TR):
        for it in xrange(burn, burn + NMC):
            i    = it - burn
            b, c = dcmpcff(alfa=mARp.allalfas[it])

            for r in xrange(R):
                rt[tr, i, :, r] = b[r] * mARp.uts[tr, it, r, :]

            for z in xrange(C):
                #print "z   %d" % z
                cf1 = 2*c[2*z].real
                gam = mARp.allalfas[it, R+2*z]
                #cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
                cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
                #print "%(1).3f    %(2).3f" % {"1": cf1, "2" : cf2}
                zt[tr, i, 0:ddN+2, z] = cf1*mARp.wts[tr, it, z, 1:ddN+3] - cf2*mARp.wts[tr, it, z, 0:ddN+2]
                #for n in xrange(1, ddN+3):
                #    zt[tr, i, n-1, z] = cf1*mARp.wts[tr, it, z, n] - cf2*mARp.wts[tr, it, z, n-1]
    return rt, zt


###############################################
def plotWFandSpks(mARp, zts0, sFilename="latnt,GrdTr,Spks", tMult=1, intv=None, sTitle=None, tr=None, cls2EvryO=None, bRealDat=False, norm=False):
    """
    tMult    multiply by time 
    """

    mdn = _N.empty((mARp.TR, mARp.N + 1))

    it0 = mARp.burn
    it1 = mARp.burn+mARp.NMC
    cmp = 0   #  AR component to show

    ITER  = it1-it0
    avgDs  = _N.empty(ITER)
    avgD = _N.empty(ITER)
    df = _N.empty((ITER, mARp.N+1))

    xLFPGT =mARp.x
    if bRealDat:
        xLFPGT =mARp.fx

    for tr in xrange(mARp.TR):
        mdn[tr] = _N.mean(zts0[tr, cls2EvryO], axis=0)
        if not norm:
            MINx = _N.min(zts0[tr])
            MAXx = _N.max(zts0[tr])
        else:
            MINx = -1.
            MAXx =  1.

        fig, ax = _plt.subplots(figsize=(12, 4))

        AMP  = MAXx - MINx
        ht   = 0.08*AMP
        ys1  = MINx - 0.5*ht
        ys2  = MINx - 3*ht

        #for it in xrange(1, len(cls2EvryO), 5):
        trls = range(min(cls2EvryO), max(cls2EvryO), 10)
        for it in trls:
            _plt.plot(zts0[tr, it], lw=1.5, color=mC.infrd95)  
        _plt.plot(mdn[tr], lw=3.5, color=mC.infrdM)
        AMP = 1
        if norm:
            AMP = _N.std(mdn[tr]) / _N.std(xLFPGT[tr])
        AMP = _N.std(mdn[tr]) / _N.std(xLFPGT[tr])
        _plt.plot(xLFPGT[tr]*AMP, color=mC.grndTruth, lw=2)

        for n in xrange(mARp.N+1):
            if mARp.y[tr, n] == 1:
                _plt.plot([n, n], [ys1, ys2], lw=2.5, color="black")
        _plt.ylim(ys2 - 0.05*AMP, MAXx + 0.05*AMP)
        #_plt.yticks(fontsize=20)
        _plt.yticks([])

        tcks = _plt.xticks()
        ot   = _N.array(tcks[0], dtype=_N.int)
        mt   = _N.array(tcks[0] * tMult, dtype=_N.int)
        _plt.xticks(ot, mt, fontsize=24)

        #_plt.xticks(_N.linspace(0, mARp.N+1, 7, dtype=_N.int), _N.linspace(0, mARp.N+1, 7, dtype=_N.int)*tMult, fontsize=20)
        _plt.xlim(0, (mARp.N+1)+1)
        _plt.xlabel("millisecond", fontsize=26)
        #_plt.axhline(y=0, color="grey", lw=2, ls="--")

        #fig.patch.set_visible(False)
        #ax.axis("off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.set_ticks_position("none")
        ax.xaxis.set_ticks_position("bottom")
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False

        if intv is not None:
            _plt.xlim(intv[0], intv[1])
        if sTitle is not None:
            _plt.title(sTitle)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.85)
        if sFilename is not None:
            _plt.savefig("%(fn)s,tr=%(tr)d.eps" % {"fn" : sFilename, "tr" : tr}, transparent=True)
            _plt.close()

    return mdn

def plotSingleCIFwSpikes(y, cifs, trl, filename="singleTrialCIF", ylim=None, xlim=None, yticks=None, xticks=None):
    fig = _plt.figure(figsize=(13, 4))
    ax  = fig.add_subplot(1, 1, 1)
    _plt.plot(cifs[trl]*1000, color=mC.smpld, lw=2)

    spkts = _N.where(y[trl] == 1)[0]
    sky   = _N.mean(cifs[trl])*1000 * 0.2
    for ts in spkts:
        _plt.plot([ts, ts], [0, sky], color="black", lw=2)

    bottomLeftAxes(ax)
    setTicksAndLims(xlabel="time (ms)", ylabel="Hz", xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim, tickFS=26, labelFS=28)

    fig.subplots_adjust(left=0.08, bottom=0.25, top=0.95, right=0.95)
    if filename != None:
        _plt.savefig("%(fn)s,tr=%(tr)d.eps" % {"fn" : filename, "tr" : trl}, transparent=True)
    _plt.close()

def plotPSTH(mARp):
    fig = _plt.figure(figsize=(6, 4))
    meanPSTH = _N.mean(_N.dot(mARp.B.T, mARp.smp_aS[mARp.burn:mARp.burn+mARp.NMC].T), axis=1)
    _plt.plot(meanPSTH, lw=2)
    _plt.savefig("psth-splines")
    _plt.close()
    return meanPSTH

def plotFsAmp(mARp, tr0=None, tr1=None, xticks=None, yticksFrq=None, yticksMod=None, yticksAmp=None, fMult=1, dir=None, fn="fs_amps", ylimAmp=None):
    __plotFsAmp(mARp.burn, mARp.NMC, mARp.fs, mARp.amps, mARp.mnStds, tr0=tr0, tr1=tr1, xticks=xticks, yticksFrq=yticksFrq, yticksMod=yticksMod, yticksAmp=yticksAmp, fMult=fMult, dir=dir, fn=fn, ylimAmp=ylimAmp)

def plotFsAmpDUMP(lm, burn, NMC, tr0=None, tr1=None, xticks=None, yticksFrq=None, yticksMod=None, yticksAmp=None, fMult=1, dir=None, fn="fs_amps", ylimAmp=None):
    fs     = lm["fs"]
    amps   = lm["amps"]
    mnStds = lm["mnStds"]

    if xticks is None:
        xticks = range(0, fs.shape[0]+1, fs.shape[0]/2)
    __plotFsAmp(burn, NMC, fs, amps, mnStds, tr0=tr0, tr1=tr1, xticks=xticks, yticksFrq=yticksFrq, yticksMod=yticksMod, yticksAmp=yticksAmp, fMult=fMult, dir=dir, fn=fn, ylimAmp=ylimAmp)

    # for it in xrange(burn + NMC):
    #     prt, rank, f, amp = ampAngRep(afs[it], f_order=True)
    # __plotFsAmp(burn, NMC, lm["fs"], tr0=tr0, tr1=tr1, xticks=xticks, yticksFrq=yticksFrq, yticksMod=yticksMod, yticksAmp=yticksAmp, fMult=fMulti, dir=dir, fn=fn)

def __plotFsAmp(burn, NMC, fs, amps, mnStds, mARp=None, dump=None, tr0=None, tr1=None, xticks=None, yticksFrq=None, yticksMod=None, yticksAmp=None, fMult=1, dir=None, fn="fs_amps", ylimAmp=None):
    if tr0 is None:
        tr0 = 1
    if tr1 is None:
        tr1 = burn + NMC
    fig = _plt.figure(figsize=(12, 3))
    ax  = fig.add_subplot(1, 3, 1)
    fs_vals = (fs[tr0:tr1:5, 0]/fMult)*1000
    max_fs_vals = _N.max(fs_vals)
    _plt.plot(range(tr0, tr1, 5), fs_vals, color=mC.smpld)
    setTicksAndLims(xlabel="iterations", ylabel="Hz", xticks=xticks, yticks=yticksFrq, xlim=tr1, tickFS=20, labelFS=20, ylim=[0, max_fs_vals*1.1])
    bottomLeftAxes(ax)
    ax.locator_params(axis="y", nbins=3)
    ######
    ax = fig.add_subplot(1, 3, 2)
    _plt.plot(range(tr0, tr1, 5), amps[tr0:tr1:5, 0], color=mC.smpld)
    _plt.ylim(0.97, 1)
    _plt.yticks([0.97, 0.98, 0.99, 1])
    setTicksAndLims(xlabel="iterations", ylabel="modulus", xticks=xticks, yticks=yticksMod, xlim=tr1, tickFS=20, labelFS=20)
    bottomLeftAxes(ax)
    ######
    ax = fig.add_subplot(1, 3, 3)
    amplim = ylimAmp if ylimAmp is not None else [0, _N.max(mnStds[tr0:tr1:5])*1.1]
    _plt.plot(range(tr0, tr1, 5), mnStds[tr0:tr1:5], color=mC.smpld)
    setTicksAndLims(xlabel="iterations", ylabel="amplitude", xticks=xticks, yticks=yticksAmp, xlim=tr1, ylim=amplim, tickFS=20, labelFS=20)
    bottomLeftAxes(ax)
    ax.locator_params(axis="y", nbins=4)
    fig.subplots_adjust(left=0.1, bottom=0.25, top=0.95, right=0.95, wspace=0.4, hspace=0.4)
    if dir is None:
        _plt.savefig("%s.eps" % fn, transparent=True)
    else:
        _plt.savefig("%(d)s/%(f)s.eps" % {"d" : dir, "f" :fn}, transparent=True)
    _plt.close()


def corrcoeffs(mARp, mdn, bRealDat=False):
    xLFPGT =mARp.x
    if bRealDat:
        xLFPGT =mARp.fx
    fig = _plt.figure()
    pcs = _N.empty(mARp.TR)
    for tr in xrange(mARp.TR):
        pc, pv = _ss.pearsonr(mdn[tr], xLFPGT[tr])
        pcs[tr] = pc
    _plt.hist(pcs, bins=_N.linspace(-0.5, 0.5, 41), color="black")
    _plt.axvline(x=_N.mean(pcs), ls="--", color="red", lw=1.5)
    _plt.xlim(-0.6, 0.6)
    _plt.suptitle("pc mean %.2f" % _N.mean(pcs))
    _plt.grid()
    _plt.savefig("pcs")
    _plt.close()

def acorrs(mARp, mdn, realDat=False):
    fx = mARp.fx
    if not realDat:
        fx = mARp.x
    _plt.acorr(fx.flatten(), usevlines=False, maxlags=300, color="red", linestyle="-", marker=".", ms=0, lw=2)
    _plt.acorr(mdn.flatten(), usevlines=False, maxlags=300, color="black", linestyle="-", marker=".", ms=0, lw=2)

    _plt.savefig("acorrs")
    _plt.close()

def findCentral(aorf, firstTR, lastTR, nbins=50, ofMax=0.8, fn=None):
    """
    nbins    bin size for drawing histogram of aorf
    ofMax    the relative height of left right lims
    """
    #############################################
    fsLO = _N.min(aorf[firstTR:lastTR, 0])
    fsHI = _N.max(aorf[firstTR:lastTR, 0])
    df   = fsHI - fsLO

    cts, bins, lines = _plt.hist(aorf[firstTR:lastTR, 0], _N.linspace(fsLO - df*0.05, fsHI + df*0.05, nbins))

    x = (bins[0:-1] + bins[1:])*0.5
    xm = _N.mean(x)
    xxm = x - xm

    a, b, c, d, e, f, g, h = _N.polyfit(x - xm, cts, 7)

    yF = a*xxm**7 + b*xxm**6 + c*xxm**5 + d*xxm**4 + e*xxm**3 + f*xxm**2 + g*xxm + h

    _plt.plot(x, yF)

    maxF   = yF.max()
    iStart = yF.tolist().index(maxF)

    iL     = -1
    iR     = -1
    i      = iStart - 1
    while i > 0:
        if yF[i] < ofMax*maxF:
            iL = i
            break 
        i -= 1

    i      = iStart + 1

    while i < nbins:
        if yF[i] < ofMax*maxF:
            iR = i
            break
        i += 1

    fMin = x[iL]      #  min f (or amp)
    fMax = x[iR]      #  max f (or amp)
    _plt.axvline(x=x[iL])
    _plt.axvline(x=x[iR])
    
    if fn is not None:
        _plt.savefig(fn)
    _plt.close()

    return fMin, fMax


def findStationaryMCMCIters(mARp, win=30):
    """
    win      win size use to estimate spread during small win in iterations
    """
    bNMC = mARp.burn + mARp.NMC
    win  = 30                  #  take 30 iters each

    minStdF =   _N.std(mARp.fs[bNMC-win:bNMC, 0])
    minStdA = _N.std(mARp.amps[bNMC-win:bNMC, 0])

    stdFs     = []
    stdAs     = []
    its       = []
    for it in xrange(bNMC - 2*win, 2, -1*win):
        it0 = it
        it1 = it + win

        stdF =   _N.std(mARp.fs[it0:it1, 0])
        stdA = _N.std(mARp.amps[it0:it1, 0])
        stdFs.append(stdF)
        stdAs.append(stdA)
        its.append(it)

        if stdF < minStdF:
            minStdF = stdF
        if stdA < minStdA:
            minStdA = stdA

    stdAs.reverse()
    stdFs.reverse()
    its.reverse()

    #  Find the ones the intervals with small fluctuations
    keep = []
    for it in xrange(len(its)):
        if (stdAs[it] < 3*minStdA) and (stdFs[it] < 3*minStdF):
            keep.append(it)


    cons = _N.diff(keep)
    bigSkip = _N.where(cons > 1)

    if len(bigSkip[0]) > 0.2*len(cons):
        print "Not fairly smooth"
    else:
        print "looks like it hit stationary state"
        if its[keep[-1]] == bNMC - 2*win:
            lastTR = bNMC
        else:
            lastTR = its[keep[-1]] + win
        firstTR = its[keep[0]]
        print "use from trial=%(1)d  to %(2)d" % {"1" : firstTR, "2" : lastTR}

    return firstTR, lastTR


def smplLatentFitTest(mARp, trSt=50, trEn=200, trSkp=5, fontsize=22, uo=None):
    us = _N.array(mARp.us)
    if uo is not None:
        us += uo
    osc = mARp.Bsmpx[:, ((trEn+trSt)/2), 2:]
    us = us.reshape(mARp.TR, 1)

    cif = mARp.CIF(us, mARp.aS, osc)
    fr       = cif/mARp.dt
    m        = 10
    dtm      = mARp.dt / m
    x, zss, bs, bsp, bsn = timeRescaleTest(fr, mARp.y, mARp.dt, mARp.TR, m, nohist=False)

    fig, ax = _plt.subplots(figsize=(5, 5))
    _plt.fill_between(x, bsn, bsp, color="darkgrey", alpha=0.8)
    ax.set_aspect("equal", "datalim")
    _plt.xlim(-0.02, 1.02)
    _plt.ylim(-0.02, 1.02)

    iii=0

    pvDs   = _N.empty(((trEn-trSt)/trSkp, 2))
    for ii in xrange(trSt, trEn, trSkp):
        us = _N.array(mARp.us)
        if uo is not None:
            us += uo
        osc = mARp.Bsmpx[:, ii, 2:]
        us = us.reshape(mARp.TR, 1)

        cif = mARp.CIF(us, mARp.aS, osc)

        fr       = cif/mARp.dt

        m        = 10
        dtm      = mARp.dt / m
        x, zss, bs, bsp, bsn = timeRescaleTest(fr, mARp.y, mARp.dt, mARp.TR, m, nohist=False)
        #allzsss[iii] = zss
        _plt.plot(x, zss, color=mC.smpld, lw=0.8)

        KS_D, pv = _ss.kstest(zss, "uniform")
        pvDs[iii, :] = KS_D, pv

        iii+=1

    _plt.plot(x, bs, ls="--", color="black", lw=2)
    _plt.xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"], fontsize=fontsize)
    _plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"], fontsize=fontsize)
    fig.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.95)
    
    if uo is None:
        _plt.savefig("AR-KS.eps", transparent=True)
    else:
        _plt.savefig("AR-KS%.3f.eps" % uo, transparent=True)
    
    return pvDs

def cmpAR2GLM(pvDs, bestD, bestpv, xticksD=None, xtickspv=None, fontsize=20, uo=None):
    fig = _plt.figure(figsize=(10, 3))

    DMax  = max(_N.max(pvDs[:, 0]), bestD)
    DMin  = min(_N.min(pvDs[:, 0]), bestD)

    lpvMax = max(_N.max(_N.log10(pvDs[:, 1])), _N.log10(bestpv))
    lpvMin = min(_N.min(_N.log10(pvDs[:, 1])), _N.log10(bestpv))
    #
    sbp = fig.add_subplot(1, 2, 1)
    ax     = sbp.get_axes()
    _plt.hist(pvDs[:, 0], bins=_N.linspace(0.9*DMin, 1.1*DMax, 20), color=mC.hist1)
    _plt.axvline(x=bestD, color="red", lw=2)
    if xticksD is None:
        _plt.xticks(fontsize=fontsize);  _plt.yticks(fontsize=fontsize)
    else:
        _plt.xticks(xticksD, fontsize=fontsize);  _plt.yticks(fontsize=fontsize)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    sbp    = fig.add_subplot(1, 2, 2)
    ax     = sbp.get_axes()
    _plt.hist(_N.log10(pvDs[:, 1]), bins=_N.linspace(lpvMin-0.5, lpvMax+0.5, 20), color=mC.hist1)
    _plt.axvline(x=_N.log10(bestpv), color="red", lw=2)
    if xtickspv is None:
        _plt.xticks(fontsize=fontsize);  _plt.yticks(fontsize=fontsize)
    else:
        _plt.xticks(xtickspv, fontsize=fontsize);  _plt.yticks(fontsize=fontsize)
    fig.subplots_adjust(left=0.13, bottom=0.13, top=0.95, right=0.95)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    if uo is None:
        _plt.savefig("histD-pv.eps", transparent=True)
    else:
        _plt.savefig("histD-pv%.3f.eps" % uo, transparent=True)
    _plt.close()

def setTicksAndLims(xlabel=None, ylabel=None, xticks=None, yticks=None, xticksD=None, yticksD=None, xlim=None, ylim=None, tickFS=26, labelFS=28):
    if xticks is not None:
        if xticksD is None:
            _plt.xticks(xticks, fontsize=tickFS)
        else:
            _plt.xticks(xticks, xticksD, fontsize=tickFS)
    else:
        _plt.xticks(fontsize=tickFS)
    if yticks is not None:
        if yticksD is None:
            _plt.yticks(yticks, fontsize=tickFS)
        else:
            _plt.yticks(yticks, yticksD, fontsize=tickFS)
    else:
        _plt.yticks(fontsize=tickFS)

    if xlim is not None:
        if type(xlim) == list:
            _plt.xlim(xlim[0], xlim[1])
        else:
            _plt.xlim(0, xlim)
    if ylim is not None:
        if type(ylim) == list:
            _plt.ylim(ylim[0], ylim[1])
        else:
            _plt.ylim(0, ylim)

    if xlabel is not None:
        _plt.xlabel(xlabel, fontsize=labelFS)
    if ylabel is not None:
        _plt.ylabel(ylabel, fontsize=labelFS)

def bottomLeftAxes(ax, bottomVis=True, leftVis=True):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(bottomVis)
    ax.spines["left"].set_visible(leftVis)

    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["right"].axis.axes.tick_params(direction="outward", width=2)
    ax.spines["top"].axis.axes.tick_params(direction="outward", width=2)

def vstackedPlots(ax, bottom=False):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if not bottom:
        ax.spines["bottom"].set_visible(False)
        _plt.xticks([])
    else:
        ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.spines["right"].axis.axes.tick_params(direction="outward", width=2)
    if bottom:
        ax.spines["top"].axis.axes.tick_params(direction="outward", width=2)

def arbitraryAxes(ax, axesVis=[True, True, True, True], xtpos="bottom", ytpos="left"):
    """
    Left, Bottom, Right, Top        Axes line itself
    xtpos               "bottom", "top", "both", "none"
    "left", "right", "both", "none"
    """
    ax.spines["left"].set_visible(axesVis[0])
    ax.spines["bottom"].set_visible(axesVis[1])
    ax.spines["right"].set_visible(axesVis[2])
    ax.spines["top"].set_visible(axesVis[3])

    ax.xaxis.set_ticks_position(xtpos)
    ax.yaxis.set_ticks_position(ytpos)
    ax.xaxis.set_label_position(xtpos)
    ax.yaxis.set_label_position(ytpos)

    ax.spines["left"].axis.axes.tick_params(direction="inward", width=2)
    ax.spines["bottom"].axis.axes.tick_params(direction="outward", width=2)
    ax.spines["right"].axis.axes.tick_params(direction="outward", width=2)
    ax.spines["top"].axis.axes.tick_params(direction="outward", width=2)

