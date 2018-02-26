#
from mcmcARpFuncs import loadL2, runNotes, loadKnown
from filter import bpFilt, lpFilt, gauKer
import mcmcAR as mAR
import ARlib as _arl
import pyPG as lw
import kfardat as _kfardat
import logerfc as _lfc
import commdefs as _cd
import os
import numpy as _N
from kassdirs import resFN, datFN
import re as _re
from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import numpy.polynomial.polynomial as _Npp
from kflib import createDataAR
#
import splineknots as _spknts
import patsy
import pickle
import matplotlib.pyplot as _plt


class mcmcARspk(mAR.mcmcAR):
    ##  
    psthBurns     = 5
    Cn            = None;    Cs            = None;    C             = None
    kntsPSTH      = None;    dfPSTH        = None
    ID_q2         = True
    use_prior     = _cd.__COMP_REF__
    AR2lims       = None
    F_alfa_rep    = None
    pkldalfas     = None

    noAR          = False    #  no oscillation
    #  Sampled 
    smp_u         = None;    smp_aS        = None
    allalfas      = None
    uts           = None;    wts           = None
    rts           = None;    zts           = None
    zts0          = None     #  the lowest component only
    ranks         = None
    pgs           = None
    fs            = None
    amps          = None
    mnStds        = None

    #  Existing data, ground truth
    fx            = None   #  filtered latent state
    px            = None   #  phase of latent state

    histknots     = 10     #  histknots == 9 if p(isi) == max at isi = 1
    #histknots     = 11
    #  LFC
    lfc           = None

    ####  TEMPORARY
    Bi            = None

    #  input data
    histFN        = None
    loghist       = None
    l2            = None
    lrn           = None
    s_lrn           = None   #  saturated lrn
    sprb           = None   #  spiking prob
    lrn_scr2           = None   #  scratch space
    lrn_scr1           = None   #  scratch space
    lrn_iscr1           = None   #  scratch space
    lrn_scr3           = None   #  scratch space
    lrn_scld           = None   #  scratch space
    mean_isi_1st2spks  = None   #  mean isis for all trials of 1st 2 spikes
    maxISI             = None

    #  Gibbs
    ARord         = _cd.__NF__
    
    #  Current values of params and state
    bpsth         = False
    B             = None;    aS            = None; 

    #  coefficient sampling
    fSigMax       = 500.    #  fixed parameters
    freq_lims     = [[1 / .85, fSigMax]]
    sig_ph0L      = -1
    sig_ph0H      = 0

    # psth spline coefficient priors
    u_a          = None;             s2_a         = 2.

    #  knownSig
    knownSigFN      = None
    knownSig        = None
    xknownSig       = 1   #  multiply knownSig by...

    h0_1      = None        # silence following spike
    h0_2      = None        # inhibitory rebound peak
    h0_3      = None        # decayed away peaky part 
    h0_4      = None        # far
    h0_5      = None        # farther

    hS        = None

    dohist    = True
    outSmplFN = "smpls.dump"
    doBsmpx   = False
    BsmpxSkp  = 1

    t0_is_t_since_1st_spk = None    #  o set up spike history, generate a virtual spike before 1st spike in each trial

    def __init__(self):
        if (self.noAR is not None) or (self.noAR == False):
            self.lfc         = _lfc.logerfc()

    def loadDat(self, trials, h0_1=None, h0_2=None, h0_3=None, h0_4=None, h0_5=None): #################  loadDat
        oo = self
        bGetFP = False

        x_st_cnts = _N.loadtxt(resFN("xprbsdN.dat", dir=oo.setname))
        y_ch      = 2   #  spike channel
        p = _re.compile("^\d{6}")   # starts like "exptDate-....."
        m = p.match(oo.setname)

        bRealDat = True
        dch = 4    #  # of data columns per trial

        if m == None:   #  not real data
            bRealDat, dch = False, 3
        else:
            flt_ch, ph_ch, bGetFP = 1, 3, True  # Filtered LFP, Hilb Trans
        TR = x_st_cnts.shape[1] / dch    #  number of trials will get filtered

        #  If I only want to use a small portion of the data
        oo.N   = x_st_cnts.shape[0] - 1
        if oo.t1 == None:
            oo.t1 = oo.N + 1
        #  meaning of N changes here
        N   = oo.t1 - 1 - oo.t0

        x   = x_st_cnts[oo.t0:oo.t1, ::dch].T
        y   = x_st_cnts[oo.t0:oo.t1, y_ch::dch].T
        if bRealDat:
            fx  = x_st_cnts[oo.t0:oo.t1, flt_ch::dch].T
            px  = x_st_cnts[oo.t0:oo.t1, ph_ch::dch].T

        ####  Now keep only trials that have spikes
        kpTrl = range(TR)
        if trials is None:
            trials = range(oo.TR)
        oo.useTrials = []
        for utrl in trials:
            try:
                ki = kpTrl.index(utrl)
                if _N.sum(y[utrl, :]) > 0:
                    oo.useTrials.append(ki)
            except ValueError:
                print "a trial requested to use will be removed %d" % utrl

        ######  oo.y are for trials that have at least 1 spike
        oo.y     = _N.array(y[oo.useTrials], dtype=_N.int)
        oo.x     = _N.array(x[oo.useTrials])
        if bRealDat:
            oo.fx    = _N.array(fx[oo.useTrials])
            oo.px    = _N.array(px[oo.useTrials])

        #  INITIAL samples
        if TR > 1:
            mnCt= _N.mean(oo.y, axis=1)
        else:
            mnCt= _N.array([_N.mean(oo.y)])

        #  remove trials where data has no information
        rmTrl = []

        oo.kp  = oo.y - 0.5
        oo.rn  = 1

        oo.TR    = len(oo.useTrials)
        oo.N     = N

        oo.smpx        = _N.zeros((oo.TR, (oo.N + 1) + 2, oo.k))   #  start at 0 + u
        oo.ws          = _N.empty((oo.TR, oo.N+1), dtype=_N.float)
        oo.lrn   = _N.empty((oo.TR, oo.N+1))

        if oo.us is None:
            oo.us    = _N.zeros(oo.TR)

        tot_isi = 0
        nisi    = 0
        isis = []
        for tr in xrange(oo.TR):
            spkts = _N.where(oo.y[tr] == 1)[0]
            isis.extend(_N.diff(spkts))
        
        #  cnts will always be 0 in frist bin
        sisis = _N.sort(isis)
        Lisi  = len(sisis)
        
        ###  look at the isi distribution

        #  cnts will always be 0 in frist bin
        maxisi = max(isis)
        minisi = min(isis)    #  >= 1

        cnts, bins = _N.histogram(isis, bins=_N.linspace(1, maxisi, maxisi))
        p9 = sisis[int(len(sisis)*0.9)]

        x = bins[minisi:p9]
        y = cnts[minisi-1:p9-1]
        z = _N.polyfit(x, y, 5)
        ply = _N.poly1d(z)
        plyx= ply(x)
        imax = _N.where(plyx == _N.max(plyx))[0][0] + minisi

        if (imax == 1):   #  no rebound excitation or pause
            oo.h0_1 = 1
            oo.h0_2 = int(sisis[int(Lisi*0.5)])#oo.h0_2*3
            oo.h0_3 = int(sisis[int(Lisi*0.65)])#oo.h0_2*3
            oo.h0_4 = int(sisis[int(Lisi*0.8)])#oo.h0_2*3
            print "-----  %(1)d  %(2)d  %(3)d  %(4)d" % {"1" : oo.h0_1, "2" : oo.h0_2, "3" : oo.h0_3, "4" : oo.h0_4}
            oo.hist_max_at_0 = True
            oo.histknots = 9
        else:      #  a pause
            ii = 1
            while cnts[ii] == 0:
                ii += 1
            oo.h0_1 = ii  #  firing prob is 0, oo.h0_1 ms postspike

            imnisi = int(_N.mean(isis))
            #imnisi = int(_N.mean(isis)*0.9)
            pts = _N.array([imax, imnisi, int(0.4*(sisis[int(Lisi*0.97)] - imnisi) + imnisi), int(sisis[int(Lisi*0.97)])])
            #pts = _N.array([int(imax*0.9), imnisi, int(0.3*(sisis[int(Lisi*0.9)] - imnisi) + imnisi), int(sisis[int(Lisi*0.9)])])
            #pts = _N.array([imax, imnisi, int(0.25*(sisis[int(Lisi*0.85)] - imnisi) + imnisi), int(sisis[int(Lisi*0.85)])])
            #pts = _N.array([imax, imnisi, int(0.5*(sisis[int(Lisi*0.995)] - imnisi) + imnisi), int(sisis[int(Lisi*0.995)])])
            #pts = _N.array([imax, imnisi, int(0.4*(sisis[int(Lisi*0.995)] - imnisi) + imnisi), int(sisis[int(Lisi*0.995)])])

            #pts = _N.array([19, 21, 23, 33])  #  quick hack for f64-1-Xaa/wp_0-60_5_1a
            spts = _N.sort(pts)
            for i in xrange(3):
                if spts[i] == spts[i+1]:
                    spts[i+1] += 1
            oo.h0_2 = spts[0]
            oo.h0_3 = spts[1]
            oo.h0_4 = spts[2]
            oo.h0_5 = spts[3]
            # #  max of ISI dist, mean of ISI dist   2, 3, 4, 5 (80th)
            # oo.h0_2 = imax
            # #oo.h0_2 = imax if (imax < 8) else 8
            # #oo.h0_3 = int(sisis[int(Lisi*0.35)])
            # #oo.h0_3 = 15 if (oo.h0_3 > 15) else oo.h0_3
            # #oo.h0_4 = int(sisis[int(Lisi*0.55)])
            # #oo.h0_4 = 22 if (oo.h0_4 > 22) else oo.h0_4
            # oo.h0_5 = int(sisis[int(Lisi*0.85)])
            # oo.h0_3 = int((oo.h0_5 - oo.h0_2)*0.33 + oo.h0_2)
            # oo.h0_4 = int((oo.h0_5 - oo.h0_2)*0.66 + oo.h0_2)

            #oo.h0_5 = 35 if (oo.h0_5 > 35) else oo.h0_5

            print "-----  %(1)d  %(2)d  %(3)d  %(4)d  %(5)d" % {"1" : oo.h0_1, "2" : oo.h0_2, "3" : oo.h0_3, "4" : oo.h0_4, "5" : oo.h0_5}

            oo.h0_1 = oo.h0_1 if h0_1 is None else h0_1
            oo.h0_2 = oo.h0_2 if h0_2 is None else h0_2
            oo.h0_3 = oo.h0_3 if h0_3 is None else h0_3
            oo.h0_4 = oo.h0_4 if h0_4 is None else h0_4
            oo.h0_5 = oo.h0_5 if h0_5 is None else h0_5

            print "-----  %(1)d  %(2)d  %(3)d  %(4)d  %(5)d   (overridden?)" % {"1" : oo.h0_1, "2" : oo.h0_2, "3" : oo.h0_3, "4" : oo.h0_4, "5" : oo.h0_5}

            oo.hist_max_at_0 = False
            oo.histknots = 10
        oo.maxISI  = int(sisis[int(Lisi*0.99)])


        crats = _N.zeros(maxisi-1)
        for n in xrange(0, maxisi-2):
            crats[n+1] = crats[n] + cnts[n]
        crats /= crats[-1]

        ####  generate spike before time=0.  PSTH estimation
        if oo.t0_is_t_since_1st_spk is None:
            oo.t0_is_t_since_1st_spk = _N.empty(oo.TR, dtype=_N.int)
            rands = _N.random.rand(oo.TR)
            for tr in xrange(oo.TR):
                spkts = _N.where(oo.y[tr] == 1)[0]

                if len(spkts) > 0:
                    t0 = spkts[0]
                    t0 = t0 if t0 < len(crats) else len(crats) - 1
                    r0 = crats[t0]   # say 0.3   
                    adjRnd = (1 - r0) * rands[tr]
                    isi = _N.where(crats >= adjRnd)[0][0]  # isi in units of bin sz

                    oo.t0_is_t_since_1st_spk[tr] = isi
        else:
            print "using saved t0_is_t_since_1st_spk"

        oo.loghist = loadL2(oo.setname, fn=oo.histFN)
        oo.dohist = True if oo.loghist is None else False

        oo.knownSig = loadKnown(oo.setname, trials=oo.useTrials, fn=oo.knownSigFN) 
        if oo.knownSig is None:
            oo.knownSig = _N.zeros((oo.TR, oo.N+1))
        else:
            oo.knownSig *= oo.xknownSig

        ###  override knot locations

    def allocateSmp(self, iters, Bsmpx=False):
        oo = self
        print "^^^^^^   allocateSmp  %d" % iters
        ####  initialize
        if Bsmpx:
            oo.Bsmpx        = _N.zeros((oo.TR, iters/oo.BsmpxSkp, (oo.N+1) + 2))
        oo.smp_u        = _N.zeros((oo.TR, iters))
        oo.smp_hS        = _N.zeros((oo.histknots, iters))   # history spline
        oo.smp_hist        = _N.zeros((oo.N+1, iters))   # history spline

        if oo.bpsth:
            oo.smp_aS        = _N.zeros((iters, oo.B.shape[0]))
        oo.smp_q2       = _N.zeros((oo.TR, iters))
        oo.smp_x00      = _N.empty((oo.TR, iters, oo.k))
        #  store samples of

        oo.allalfas     = _N.empty((iters, oo.k), dtype=_N.complex)
        if oo.pkldalfas is not None:
            oo.allalfas[0]  = oo.pkldalfas
            for r in xrange(oo.R):
                oo.F_alfa_rep[r] = oo.pkldalfas[r].real
            for c in xrange(oo.C):
                oo.F_alfa_rep[oo.R+2*c]   = oo.pkldalfas[oo.R+2*c]
                oo.F_alfa_rep[oo.R+2*c+1] = oo.pkldalfas[oo.R+2*c + 1]
            print oo.F_alfa_rep
        #oo.uts          = _N.empty((oo.TR, iters, oo.R, oo.N+2))
        #oo.wts          = _N.empty((oo.TR, iters, oo.C, oo.N+3))
        oo.ranks        = _N.empty((iters, oo.C), dtype=_N.int)
        oo.pgs          = _N.empty((oo.TR, iters, oo.N+1))
        oo.fs           = _N.empty((iters, oo.C))
        oo.amps         = _N.empty((iters, oo.C))

        oo.mnStds       = _N.empty(iters)

    def setParams(self):
        oo = self
        # #generate initial values of parameters
        oo._d = _kfardat.KFARGauObsDat(oo.TR, oo.N, oo.k)
        oo._d.copyData(oo.y)

        #  baseFN_inter   baseFN_comps   baseFN_comps

        radians      = buildLims(oo.Cn, oo.freq_lims, nzLimL=1.)
        oo.AR2lims      = 2*_N.cos(radians)

        oo.smpx        = _N.zeros((oo.TR, (oo.N + 1) + 2, oo.k))   #  start at 0 + u
        oo.ws          = _N.empty((oo.TR, oo._d.N+1), dtype=_N.float)

        if oo.F_alfa_rep is None:
            oo.F_alfa_rep  = initF(oo.R, oo.Cs, oo.Cn, ifs=oo.ifs).tolist()   #  init F_alfa_rep

        print ampAngRep(oo.F_alfa_rep)
        if oo.q20 is None:
            oo.q20         = 0.00077
        oo.q2          = _N.ones(oo.TR)*oo.q20

        oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]
        ########  Limit the amplitude to something reasonable
        xE, nul = createDataAR(oo.N, oo.F0, oo.q20, 0.1)
        mlt  = _N.std(xE) / 0.5    #  we want amplitude around 0.5
        oo.q2          /= mlt*mlt
        xE, nul = createDataAR(oo.N, oo.F0, oo.q2[0], 0.1)

        w  =  5
        wf =  gauKer(w)
        gk = _N.empty((oo.TR, oo.N+1))
        fgk= _N.empty((oo.TR, oo.N+1))
        for m in xrange(oo.TR):
            gk[m] =  _N.convolve(oo.y[m], wf, mode="same")
            gk[m] =  gk[m] - _N.mean(gk[m])
            gk[m] /= 5*_N.std(gk[m])
            fgk[m] = bpFilt(15, 100, 1, 135, 500, gk[m])   #  we want
            fgk[m, :] /= 3*_N.std(fgk[m, :])

            if oo.noAR:
                oo.smpx[m, 2:, 0] = 0
            else:
                oo.smpx[m, 2:, 0] = fgk[m, :]

            for n in xrange(2+oo.k-1, oo.N+1+2):  # CREATE square smpx
                oo.smpx[m, n, 1:] = oo.smpx[m, n-oo.k+1:n, 0][::-1]
            for n in xrange(2+oo.k-2, -1, -1):  # CREATE square smpx
                oo.smpx[m, n, 0:oo.k-1] = oo.smpx[m, n+1, 1:oo.k]
                oo.smpx[m, n, oo.k-1] = _N.dot(oo.F0, oo.smpx[m, n:n+oo.k, oo.k-2]) # no noise

        if oo.bpsth:
            psthKnts, apsth, aWeights = _spknts.suggestPSTHKnots(oo.dt, oo.TR, oo.N+1, oo.y.T, iknts=4)
            _N.savetxt("apsth.txt", apsth, fmt="%.4f")
            _N.savetxt("psthKnts.txt", psthKnts, fmt="%.4f")

            apprx_ps = _N.array(_N.abs(aWeights))
            oo.u_a   = -_N.log(1/apprx_ps - 1)

            #  For oo.u_a, use the values we get from aWeights 

            print psthKnts

            oo.B = patsy.bs(_N.linspace(0, (oo.t1 - oo.t0)*oo.dt, (oo.t1-oo.t0)), knots=(psthKnts*oo.dt), include_intercept=True)    #  spline basis

            oo.B = oo.B.T    #  My convention for beta
            oo.aS    = _N.array(oo.u_a)
            # fig = _plt.figure(figsize=(4, 7))
            # fig.add_subplot(2, 1, 1)
            # _plt.plot(apsth)
            # fig.add_subplot(2, 1, 2)
            # _plt.plot(_N.dot(oo.B.T, aWeights))
        else:
            oo.B = patsy.bs(_N.linspace(0, (oo.t1 - oo.t0)*oo.dt, (oo.t1-oo.t0)), df=4, include_intercept=True)    #  spline basis
 
            oo.B = oo.B.T    #  My convention for beta
            oo.aS = _N.zeros(4)

            #oo.Hbf = patsy.bs(_N.linspace(0, (oo.N+1), oo.N+1, endpoint=False), knots=_N.array([oo.h0_1, oo.h0_2, oo.h0_3, oo.h0_4, oo.h0_5, int(0.7*(oo.N+1))]), include_intercept=True)    #  spline basisp

        #farknot = oo.maxISI*2# < (oo.t1-oo.t0) if oo.maxISI*2  else int((oo.t1-oo.t0) *0.9)
        farknot = oo.maxISI*1.3# < (oo.t1-oo.t0) if oo.maxISI*2  else int((oo.t1-oo.t0) *0.9)
        if oo.hist_max_at_0:
            oo.Hbf = patsy.bs(_N.linspace(0, (oo.N+1), oo.N+1, endpoint=False), knots=_N.array([oo.h0_1, oo.h0_2, oo.h0_3, oo.h0_4, farknot]), include_intercept=True)    #  spline basisp
        else:
            oo.Hbf = patsy.bs(_N.linspace(0, (oo.N+1), oo.N+1, endpoint=False), knots=_N.array([oo.h0_1, oo.h0_2, oo.h0_3, oo.h0_4, oo.h0_5, farknot]), include_intercept=True)    #  spline basisp

    def stitch_Hist(self, ARo, hcrv, stsM):  # history curve
        #  this has no direct bearing on sampling of history knots
        #  however, 
        oo = self
        for m in xrange(oo.TR):
            sts = stsM[m]
            for it in xrange(len(sts)-1):
                t0 = sts[it]
                t1 = sts[it+1]
                ARo[m, t0+1:t1+1] = hcrv[t0-t0:t1-t0]
            T = oo.N+1 - sts[-1]
            ARo[m, t1+1:] = hcrv[0:T-1]
            isiHiddenPrt = oo.t0_is_t_since_1st_spk[m] + 1
            ARo[m, 0:sts[0]+1] = hcrv[isiHiddenPrt:isiHiddenPrt + sts[0]+1]

    def getComponents(self):
        oo    = self
        TR    = oo.TR
        NMC   = oo.NMC
        burn  = oo.burn
        R     = oo.R
        C     = oo.C
        ddN   = oo.N

        oo.rts = _N.empty((TR, burn+NMC, ddN+2, R))    #  real components   N = ddN
        oo.zts = _N.empty((TR, burn+NMC, ddN+2, C))    #  imag components 

        for tr in xrange(TR):
            for it in xrange(1, burn + NMC):
                b, c = dcmpcff(alfa=oo.allalfas[it])
                print b
                print c
                for r in xrange(R):
                    oo.rts[tr, it, :, r] = b[r] * oo.uts[tr, it, r, :]

                for z in xrange(C):
                    #print "z   %d" % z
                    cf1 = 2*c[2*z].real
                    gam = oo.allalfas[it, R+2*z]
                    cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
                    oo.zts[tr, it, 0:ddN+2, z] = cf1*oo.wts[tr, it, z, 1:ddN+3] - cf2*oo.wts[tr, it, z, 0:ddN+2]

        oo.zts0 = _N.array(oo.zts[:, :, 1:, 0], dtype=_N.float16)

    def dump(self):
        oo    = self
        pcklme = [oo]
        oo.Bsmpx = None
        oo.smpx  = None
        oo.wts   = None
        oo.uts   = None
        oo._d    = None
        oo.lfc   = None
        oo.rts   = None
        oo.zts   = None

        dmp = open("mARp.dump", "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()

        # import pickle
        # with open("mARp.dump", "rb") as f:
        # lm = pickle.load(f)


    def readdump(self):
        oo    = self

        with open("mARp.dump", "rb") as f:
            lm = pickle.load(f)
        f.close()
        oo.F_alfa_rep = lm[0].allalfas[-1].tolist()
        oo.q20 = lm[0].q2[0]
        oo.aS  = lm[0].aS
        oo.us  = lm[0].us

    """
    def CIF(TR, N, us, B, alps, osc, it):
        ARo   = _N.empty((TR, N+1))

        BaS = _N.dot(B.T, alps).reshape((1, 1200))
        usr = us.reshape((40, 1))

        Msts = []
        for m in xrange(TR):
            Msts.append(_N.where(y[m] == 1)[0])

        oo.stitch_Hist(ARo, smp_hist[:, it], Msts)

        usr + ARo 
        cif = _N.exp(usr + ARo + osc + BaS) / (1 + _N.exp(usr + ARo + osc + BaS))

        return cif
    """

    def CIF(self, us, B, alps, hS, osc):
        oo = self
        ARo   = _N.empty((oo.TR, oo.N+1))

        BaS = _N.dot(oo.B.T, alps)
        Msts = []
        for m in xrange(oo.TR):
            Msts.append(_N.where(oo.y[m] == 1)[0])

        oo.loghist = _N.zeros(oo.N+1)
        _N.dot(oo.Hbf, hS, out=oo.loghist)

        oo.stitch_Hist(ARo, oo.loghist, Msts)

        cif = _N.exp(us + ARo + osc + BaS + oo.knownSig) / (1 + _N.exp(us + ARo + osc + BaS + oo.knownSig))

        return cif

    def findMode(self, startIt=None, NB=20, NeighB=1, dir=None):
        oo  = self
        startIt = oo.burn if startIt == None else startIt
        aus = _N.mean(oo.smp_u[:, startIt:], axis=1)
        aSs = _N.mean(oo.smp_aS[startIt:], axis=0)
        hSs = _N.mean(oo.smp_hS[:, startIt:], axis=1)

        L   = oo.burn + oo.NMC - startIt

        hist, bins = _N.histogram(oo.fs[startIt:, 0], _N.linspace(_N.min(oo.fs[startIt:, 0]), _N.max(oo.fs[startIt:, 0]), NB))
        indMfs =  _N.where(hist == _N.max(hist))[0][0]
        indMfsL =  max(indMfs - NeighB, 0)
        indMfsH =  min(indMfs + NeighB+1, NB-1)
        loF, hiF = bins[indMfsL], bins[indMfsH]

        hist, bins = _N.histogram(oo.amps[startIt:, 0], _N.linspace(_N.min(oo.amps[startIt:, 0]), _N.max(oo.amps[startIt:, 0]), NB))
        indMamps  =  _N.where(hist == _N.max(hist))[0][0]
        indMampsL =  max(indMamps - NeighB, 0)
        indMampsH =  min(indMamps + NeighB+1, NB)
        loA, hiA = bins[indMampsL], bins[indMampsH]

        fig = _plt.figure(figsize=(8, 8))
        fig.add_subplot(2, 1, 1)
        _plt.hist(oo.fs[startIt:, 0], bins=_N.linspace(_N.min(oo.fs[startIt:, 0]), _N.max(oo.fs[startIt:, 0]), NB), color="black")
        _plt.axvline(x=loF, color="red")
        _plt.axvline(x=hiF, color="red")
        fig.add_subplot(2, 1, 2)
        _plt.hist(oo.amps[startIt:, 0], bins=_N.linspace(_N.min(oo.amps[startIt:, 0]), _N.max(oo.amps[startIt:, 0]), NB), color="black")
        _plt.axvline(x=loA, color="red")
        _plt.axvline(x=hiA, color="red")
        if dir is None:
            _plt.savefig(resFN("chosenFsAmps%d" % startIt, dir=oo.setname))
        else:
            _plt.savefig(resFN("%(sn)s/chosenFsAmps%(it)d" % {"sn" : dir, "it" : startIt}, dir=oo.setname))
        _plt.close()

        indsFs = _N.where((oo.fs[startIt:, 0] >= loF) & (oo.fs[startIt:, 0] <= hiF))
        indsAs = _N.where((oo.amps[startIt:, 0] >= loA) & (oo.amps[startIt:, 0] <= hiA))

        asfsInds = _N.intersect1d(indsAs[0], indsFs[0]) + startIt
        q = _N.mean(oo.smp_q2[0, startIt:])


        #alfas = _N.mean(oo.allalfas[asfsInds], axis=0)
        pcklme = [aus, q, oo.allalfas[asfsInds], aSs, oo.B, hSs, oo.Hbf]
        
        if dir is None:
            dmp = open(resFN("posteriorModes%d.pkl" % startIt, dir=oo.setname), "wb")
        else:
            dmp = open(resFN("%(sn)s/posteriorModes%(it)d.pkl" % {"sn" : dir, "it" : startIt}, dir=oo.setname), "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()

    
    def dump_smps(self, pcklme=None, dir=None, toiter=None):
        oo    = self
        if pcklme is None:
            pcklme = {}

        toiter         = oo.NMC + oo.burn if (toiter is None) else toiter
        if oo.bpsth:
            pcklme["aS"]   = oo.smp_aS[0:toiter]  #  this is last
        pcklme["B"]    = oo.B
        pcklme["q2"]   = oo.smp_q2[:, 0:toiter]
        pcklme["amps"] = oo.amps[0:toiter]
        pcklme["fs"]   = oo.fs[0:toiter]
        pcklme["u"]    = oo.smp_u[:, 0:toiter]
        pcklme["mnStds"]= oo.mnStds[0:toiter]
        pcklme["allalfas"]= oo.allalfas[0:toiter]
        pcklme["smpx"] = oo.smpx
        pcklme["ws"]   = oo.ws
        pcklme["t0_is_t_since_1st_spk"] = oo.t0_is_t_since_1st_spk
        if oo.Hbf is not None:
            pcklme["spkhist"] = oo.smp_hist[:, 0:toiter]
            pcklme["Hbf"]    = oo.Hbf
            pcklme["h_coeffs"]    = oo.smp_hS[:, 0:toiter]
        if oo.doBsmpx:
            pcklme["Bsmpx"]    = oo.Bsmpx
            

        print "saving state in %s" % oo.outSmplFN
        if dir is None:
            dmp = open(oo.outSmplFN, "wb")
        else:
            dmp = open("%(d)s/%(sfn)s" % {"d" : dir, "sfn" : oo.outSmplFN}, "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()

        # import pickle
        # with open("smpls.dump", "rb") as f:
        # lm = pickle.load(f)
