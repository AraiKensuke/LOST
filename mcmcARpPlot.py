import matplotlib.pyplot as _plt
import numpy as _N
import scipy.stats as _ss
from kassdirs import resFN

def plotFigs(setname, N, k, burn, NMC, x, y, Bsmpx, smp_u, smp_q2, _t0, _t1, Cs, Cn, C, baseFN, TR, tr, bRealDat=False, ID_q2=False):
    if not bRealDat:
        pcs = _N.zeros(burn+NMC)
        offs = 0
        for i in range(1, burn+NMC):
            pc, pv = _ss.pearsonr(Bsmpx[tr, i, 2+offs:], x[tr, offs:])
            pcs[i] = pc

    fig = _plt.figure(figsize=(8.5, 3*4.2))
    fig.add_subplot(3, 1, 1)
    _plt.plot(smp_q2[tr], lw=1.5, color="black")
    _plt.ylabel("q2")
    fig.add_subplot(3, 1, 2)
    _plt.plot(smp_u[tr], lw=1.5, color="black")
    _plt.ylabel("u")
    if not bRealDat:
        fig.add_subplot(3, 1, 3)
        _plt.plot(range(1, burn+NMC), pcs[1:], color="black", lw=1.5)
        _plt.axhline(y=0, lw=1, ls="--", color="blue")
        _plt.ylabel("CC")
        _plt.xlabel("ITER")
    fig.subplots_adjust(left=0.15)
    if (ID_q2 == True) and (TR > 1):
        _plt.savefig(resFN("%(bf)s_tr=%(tr)d_smps.png" % {"bf" : baseFN, "tr" : tr}, dir=setname, create=True))
    else:
        _plt.savefig(resFN("%s_smps.png" % baseFN, dir=setname, create=True))
    _plt.close()

    msmpx = _N.mean(Bsmpx[tr, burn:, 2:], axis=0)
    MINx = min(x[tr, 50:])
    MAXx = max(x[tr, 50:])

    AMP  = MAXx - MINx
    ht   = 0.08*AMP
    ys1  = MINx - 0.5*ht
    ys2  = MINx - 3*ht

    fig = _plt.figure(figsize=(14.5, 3.3))
    if not bRealDat:
        pc2, pv2 = _ss.pearsonr(msmpx[offs:], x[tr, offs:])
    else:
        pc2 = 0
    _plt.plot(x[tr], color="black", lw=2)
    _plt.plot(msmpx, color="red", lw=1.5)
    for n in range(N+1):
        if y[tr, n] == 1:
            _plt.plot([n, n], [ys1, ys2], lw=1.5, color="blue")
    _plt.ylim(ys2 - 0.05*AMP, MAXx + 0.05*AMP)
    _plt.xticks(fontsize=20)
    _plt.yticks(fontsize=20)
    _plt.grid()
    _plt.title("avg. smpx AR%(k)d   %(pc).2f   # spks %(n)d" % {"k" : k, "pc" : pc2, "n" : _N.sum(y[tr])})
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.85)
    if TR > 1:
        _plt.savefig(resFN("%(s)s_tr=%(tr)d_infer.png" % {"s" : baseFN, "tr" : tr}, dir=setname, create=True))
    else:
        _plt.savefig(resFN("%s_infer.png" % baseFN, dir=setname, create=True))
    _plt.close()

def plotARcomps(setname, N, k, burn, NMC, fs, amps, _t0, _t1, Cs, Cn, C, baseFN, TR, tr, bRealDat=False):
    for i in range(2):
        c0 = 0
        c1 = Cs
        sSN = "SG"
        if i == 1:
            c0 = Cs
            c1 = C
            sSN = "NZ"

        rows = 2
        fig = _plt.figure(figsize=(9, rows*3.5))
        ax1 = fig.add_subplot(2, 1, 1)
        if i == 0:
            _plt.ylim(0, 0.2)
            #_plt.ylim(0, 1.)
        if i == 1:
            _plt.ylim(0, 1.)
        _plt.ylabel("freqs")
        _plt.grid()
        ax2 = fig.add_subplot(2, 1, 2)
        _plt.ylim(0, 1)
        _plt.ylabel("amps")
        _plt.xlabel("MCMC iter")
        clrs=["black", "blue", "green", "red", "orange", "gray", "brown", "purple", "pink", "cyan"]
        for cmp in range(c0, c1):
            lw=1.5
            if cmp == 0:
                lw = 3.5
            ax1.plot(fs[:, cmp], lw=lw, color=clrs[cmp%10])
            ax2.plot(amps[:, cmp], lw=lw, color=clrs[cmp%10])
        _plt.grid()
        fig.subplots_adjust(top=0.97, bottom=0.08, left=0.11, right=0.96, hspace=0.18, wspace=0.18)
        _plt.savefig(resFN("%(bF)s_compsTS_%(sSN)s.png" % {"bF" : baseFN, "sSN" : sSN}, dir=setname, create=True))
        _plt.close()

        rows = C
        fig = _plt.figure(figsize=(9, rows*3.5))
        for cmp in range(C):
            ax1 = _plt.subplot2grid((rows, 2), (cmp, 0), colspan=1)
            _plt.hist(fs[burn:, cmp], bins=40, color=clrs[cmp%10])
            _plt.title("comp %d" % cmp)
            if cmp == C - 1:
                _plt.xlabel("freqs")
            ax2 = _plt.subplot2grid((rows, 2), (cmp, 1), colspan=1)
            _plt.hist(amps[burn:, cmp], bins=40, color=clrs[cmp%10])
            if cmp == C - 1:
                _plt.xlabel("amps")
        fig.subplots_adjust(top=0.97, bottom=0.05, left=0.06, right=0.96, hspace=0.21, wspace=0.21)
        _plt.savefig(resFN("%(bF)s_comps_%(sSN)s.png" % {"bF" : baseFN, "sSN" : sSN}, dir=setname, create=True))
        _plt.close()

def plotWFandSpks(N, spks, wfs, tMult=None, sTitle=None, sFilename=None, intv=None):
    """
    tMult    multiply by time 
    """
    nSigs = len(wfs)
    MINx = min(wfs[0])
    MAXx = max(wfs[0])

    AMP  = MAXx - MINx
    ht   = 0.08*AMP
    ys1  = MINx - 0.5*ht
    ys2  = MINx - 3*ht

    fig = _plt.figure(figsize=(14.5, 3.3), frameon=False)

    clrs = ["black", "red", "blue", "green"]
    lws  = [3, 2.5, 2.5, 2.5]
    for ns in range(nSigs):
        _plt.plot(wfs[ns], color=clrs[ns % 4], lw=lws[ns])

    for n in range(N+1):
        if spks[n] == 1:
            _plt.plot([n, n], [ys1, ys2], lw=2.5, color="grey")
    _plt.ylim(ys2 - 0.05*AMP, MAXx + 0.05*AMP)
    _plt.xticks(fontsize=20)
    if tMult != None:
        tcks = _plt.xticks()
        ot   = _N.array(tcks[0], dtype=_N.int)
        mt   = _N.array(tcks[0] * tMult, dtype=_N.int)
        _plt.xticks(ot, mt)
    _plt.yticks([], fontsize=20)
    _plt.axhline(y=0, color="grey", lw=2, ls="--")

    if intv != None:
        _plt.xlim(intv[0], intv[1])
    if sTitle != None:
        _plt.title(sTitle)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.85)
    if sFilename != None:
        _plt.savefig(sFilename)
        _plt.close()


def plotQ2(setdir, baseFN, burn, NMC, TR0, TR1, smp_q2, hilite=None):
    fig = _plt.figure(figsize=(8.5, 4.2))
    if hilite != None:   #  these are the ones that should be weakly modulated
        for tr in range(TR0, TR1):
            try:
                ind = hilite.index(tr - TR0)
                _plt.plot(_N.sqrt(smp_q2[tr-TR0]), lw=2.5, color="black")
            except ValueError:
                _plt.plot(_N.sqrt(smp_q2[tr-TR0]), lw=1.5, color="grey")
        _plt.savefig(resFN("%s_q2_smps.png" % baseFN, dir=setdir))
        _plt.close()
    else:
        for tr in range(TR0, TR1):
            _plt.plot(smp_q2[tr-TR0], lw=1.5, color="black")
            _plt.savefig(resFN("%s_q2_smps.png" % baseFN, dir=setdir))
            _plt.close()
    fig = _plt.figure(figsize=(8.5, 4.2))
    _plt.bar(range(TR0, TR1), _N.sqrt(_N.mean(smp_q2[:, burn+NMC-100:], axis=1)), align="center")
    _plt.xlim(TR0-0.5, TR1-0.5)
    _plt.savefig(resFN("%s_sqr_q2_smps.png" % baseFN, dir=setdir))
    _plt.close()


