import numpy as _N
import commdefs as _cd
#from ARcfSmpl import ARcfSmpl

from filter import gauKer, lpFilt, bpFilt
from LOSTdirs import resFN
import re as _re
import os


def loadDat(setname, model, t0=0, t1=None, filtered=False, phase=False):  ##################################
    if model== "bernoulli":
        x_st_cnts = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
        y_ch      = 2   #  spike channel
        p = _re.compile("^\d{6}")   # starts like "exptDate-....."
        m = p.match(setname)
        
        bRealDat = True
        dch = 4    #  # of data columns per trial
        
        if m == None:
            bRealDat = False
            dch = 3
        else:
            flt_ch      = 1    # Filtered LFP
            ph_ch       = 3    # Hilbert Trans
    else:
        x_st_cnts = _N.loadtxt(resFN("cnt_data.dat", dir=setname))
        y_ch        = 1    # spks

        dch  = 2
    TR = x_st_cnts.shape[1] / dch

    #  READ parameters of generative model from file
    #  contains N, k, singleFreqAR, u, beta, dt, stNz
    n0  = t0
    N   = x_st_cnts.shape[0] - 1
    if t1 == None:
        t1 = N + 1
    #  meaning of N changes here
    N   = t1 - 1 - t0

    if TR == 1:
        x   = x_st_cnts[t0:t1, 0]
        y   = x_st_cnts[t0:t1, y_ch]
        fx  = x_st_cnts[t0:t1, 0]
        px  = x_st_cnts[t0:t1, y_ch]
        x   = x.reshape(1, t1 - t0)
        y   = y.reshape(1, t1 - t0)
        fx  = x.reshape(1, t1 - t0)
        px  = y.reshape(1, t1 - t0)
    else:
        x   = x_st_cnts[t0:t1, ::dch].T
        y   = x_st_cnts[t0:t1, y_ch::dch].T
        if filtered:
            fx  = x_st_cnts[t0:t1, flt_ch::dch].T
        if phase:
            px  = x_st_cnts[t0:t1, ph_ch::dch].T

    if model== "bernoulli":
        global prbs
        if TR == 1:
            prbs     = x_st_cnts[t0:t1, 1]
        else:
            prbs     = x_st_cnts[t0:t1, 1::dch].T

    #  FIXED params
    r     = 200
    n     = 10

    #  INITIAL samples
    if TR > 1:
        mnCt= _N.mean(y, axis=1)
    else:
        mnCt= _N.array([_N.mean(y)])
        
    #  remove trials where data has no information
    rmTrl = []
    print(TR)
    kpTrl = range(TR)
    if model == "binomial":
        kp  = y - n*0.5
        rn  = n
        p0   = mnCt / float(rn)       #  matches 1 - p of genearted
        u  = _N.log(p0 / (1 - p0))    #  -1*u generated
    elif model == "negative binomial":
        kp  = (y - r) *0.5
        rn  = r
        p0   = mnCt / (mnCt + rn)       #  matches 1 - p of genearted
        u  = _N.log(p0 / (1 - p0))    #  -1*u generated
    else:
        kp  = y - 0.5
        rn  = 1
        dt  = 0.001
        logdt = _N.log(dt)
        if TR > 1:
            ysm = _N.sum(y, axis=1)
            for tr in xrange(TR-1, -1, -1):
                if ysm[tr] == 0:
                    print("!!!!!!!!!!!!!!     trial w/ 0 spks!!!!!!!")
                    rmTrl.append(tr)
                    kpTrl.pop(tr)
                    ysm[tr] = 10  #  this trial will be removed anyway
            u   = _N.log(ysm / ((N+1 - ysm)*dt)) + logdt
        else:
            u   = _N.array([_N.log(_N.sum(y) / ((N+1 - _N.sum(y))*dt)) + logdt])
    TR    = TR - len(rmTrl)
    if filtered and phase:
        return TR, rn, x, y, fx, px, N, kp, u, rmTrl, kpTrl        
    elif filtered and (not phase):
        return TR, rn, x, y, fx, N, kp, u, rmTrl, kpTrl        
    elif (not filtered) and phase:
        return TR, rn, x, y, px, N, kp, u, rmTrl, kpTrl        
    return TR, rn, x, y, N, kp, u, rmTrl, kpTrl

def initBernoulli(model, k, F0, TR, N, y, fSigMax, smpx, Bsmpx):  ############
    if model == "bernoulli":
        w  =  5
        wf =  gauKer(w)
        gk = _N.empty((TR, N+1))
        fgk= _N.empty((TR, N+1))
        for m in xrange(TR):
            gk[m] =  _N.convolve(y[m], wf, mode="same")
            gk[m] =  gk[m] - _N.mean(gk[m])
            gk[m] /= 5*_N.std(gk[m])
            fgk[m] = bpFilt(15, 100, 1, 135, 500, gk[m])   #  we want
            fgk[m, :] /= 2*_N.std(fgk[m, :])

            smpx[m, 2:, 0] = fgk[m, :]
            for n in xrange(2+k-1, N+1+2):  # CREATE square smpx
                smpx[m, n, 1:] = smpx[m, n-k+1:n, 0][::-1]
            for n in xrange(2+k-2, -1, -1):  # CREATE square smpx
                smpx[m, n, 0:k-1] = smpx[m, n+1, 1:k]
                smpx[m, n, k-1] = _N.dot(F0, smpx[m, n:n+k, k-2]) # no noise

            Bsmpx[m, 0, :] = smpx[m, :, 0]

def plot_cmptSpksAndX(N, z, x, y):  #  a component
    x   /= _N.std(x)
    z   /= _N.std(z)
    MINz = min(z)
    MAXz = max(z)
    AMP  = MAXz - MINz

    ht   = 0.08*AMP
    ys1  = MINz - 0.5*ht
    ys2  = MINz - 3*ht

    fig = _plt.figure(figsize=(14.5, 3.3))
    pc2, pv2 = _ss.pearsonr(z, x)
    _plt.plot(x, color="black", lw=2)
    _plt.plot(z, color="red", lw=1.5)
    for n in xrange(N+1):
        if y[n] == 1:
            _plt.plot([n, n], [ys1, ys2], lw=1.5, color="blue")
    _plt.ylim(ys2 - 0.05*AMP, MAXz + 0.05*AMP)
    _plt.xticks(fontsize=20)
    _plt.yticks(fontsize=20)
    _plt.grid()

def loadL2(runDir, fn=None):
    if (fn is not None) and (fn != ""):
        fn = "%(rd)s/%(fn)s" % {"rd" : runDir, "fn" : fn}
        print(fn)
        if os.access(fn, os.F_OK):
            print("***  loaded spike history file \"%s\" ***" % fn)
            spkhist = _N.loadtxt(fn)
            print(spkhist[0:50])
            loghist = _N.log(spkhist)
            print(loghist[0:50])
            print("-----------------")
            tooneg = _N.where(loghist < -6)[0]
            loghist[tooneg] = -6
            return loghist
        print("!!!  NO history file loaded !!!")
        if fn is not None:
            print("!!!  Couldn't find short-term history \"%s\" !!!" % fn)
        
    return None

def runNotes(setname, ID_q2, TR0, TR1):
    fp = open(resFN("notes.txt", dir=setname), "w")
    fp.write("ID_q2=%s\n" % str(ID_q2))
    fp.write("TR0=%d\n" % TR0)
    fp.write("TR1=%d\n" % TR1)
    fp.close()

    #  ID_q2
    #  Trials using

def loadKnown(runDir, trials=None, fn="known.dat"):
    if (fn is not None) and (fn != ""):
        fn = "%(rd)s/%(fn)s" % {"rd" : runDir, "fn" : fn}
        if os.access(fn, os.F_OK):
            print("***  loaded known signal \"%s\" ***" % fn)
            a = _N.loadtxt(fn)
            if trials is None:
                return a.T
            else:
                return a.T[trials]
        print("!!!  NO known signal loaded !!!")
        if fn is not None:
            print("!!!  Couldn't find \"%s\" !!!" % fn)

    return None
