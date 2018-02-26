from mcmcARpFuncs import loadL2, runNotes
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
import patsy

class mcmcARspk1(mAR.mcmcAR):
    ##  
    psthBurns     = 30
    Cn            = None;    Cs            = None;    C             = None
    kntsPSTH      = None;    dfPSTH        = None
    ID_q2         = True
    use_prior     = _cd.__COMP_REF__
    AR2lims       = None
    F_alfa_rep    = None

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

    #  Existing data, ground truth
    fx            = None   #  filtered latent state
    px            = None   #  phase of latent state

    #  LFC
    lfc           = None

    ####  TEMPORARY
    Bi            = None

    #  input data
    histFN        = None
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
    u_a          = None;             s2_a         = 0.5

    def __init__(self):
        if (self.noAR is not None) or (self.noAR == False):
            self.lfc         = _lfc.logerfc()

    def loadDat(self, trials): #################  loadDat
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

        oo.smpx        = _N.zeros((oo.TR, oo.N + 1))   #  start at 0 + u
        oo.ws          = _N.empty((oo.TR, oo.N+1), dtype=_N.float)
        oo.lrn   = _N.empty((oo.TR, oo.N+1))

        oo.us    = _N.zeros(oo.TR)

        tot_isi = 0
        nisi    = 0
        for tr in xrange(oo.TR):
            spkts = _N.where(oo.y[tr] == 1)
            if len(spkts[0]) > 2:
                nisi += 1
                tot_isi += spkts[0][1] - spkts[0][0]
        oo.mean_isi_1st2spks = float(tot_isi) / nisi
        #####  LOAD spike history
        oo.l2 = loadL2(oo.setname, fn=oo.histFN)
        if oo.l2 is None:
            oo.lrn[:] = 1
        else:
            #  assume ISIs near beginning of data are exponentially 
            #  distributed estimate
            for tr in xrange(oo.TR):
                oo.lrn[tr] = oo.build_lrnLambda2(tr)

    def allocateSmp(self, iters):
        oo = self
        print "^^^^^^   allocateSmp  %d" % iters
        ####  initialize
        oo.Bsmpx        = _N.zeros((oo.TR, iters, oo.N+1))
        oo.smp_u        = _N.zeros((oo.TR, iters))
        if oo.bpsth:
            oo.smp_aS        = _N.zeros((iters, oo.dfPSTH))
        oo.smp_q2       = _N.zeros((oo.TR, iters))
        oo.smp_x00      = _N.empty((oo.TR, iters))
        #  store samples of
        oo.pgs          = _N.empty((oo.TR, iters, oo.N+1))

        oo.mnStds       = _N.empty(iters)

    def setParams(self):
        oo = self
        # #generate initial values of parameters
        oo._d = _kfardat.KFARGauObsDat(oo.TR, oo.N, 1)
        oo._d.copyData(oo.y)

        #  baseFN_inter   baseFN_comps   baseFN_comps

        oo.smpx        = _N.zeros((oo.TR, oo.N + 1))   #  start at 0 + u
        oo.ws          = _N.empty((oo.TR, oo._d.N+1), dtype=_N.float)

        if oo.q20 is None:
            oo.q20         = 0.00077
        oo.q2          = _N.ones(oo.TR)*oo.q20

        oo.F0          = _N.array([0.9])
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
            fgk[m, :] /= 2*_N.std(fgk[m, :])

            if oo.noAR:
                oo.smpx[m, 0] = 0
            else:
                oo.smpx[m] = fgk[m]

        oo.s_lrn   = _N.empty((oo.TR, oo.N+1))
        oo.sprb   = _N.empty((oo.TR, oo.N+1))
        oo.lrn_scr1   = _N.empty(oo.N+1)
        oo.lrn_iscr1   = _N.empty(oo.N+1)
        oo.lrn_scr2   = _N.empty(oo.N+1)
        oo.lrn_scr3   = _N.empty(oo.N+1)
        oo.lrn_scld   = _N.empty(oo.N+1)

        if oo.bpsth:
            oo.B = patsy.bs(_N.linspace(0, (oo.t1 - oo.t0)*oo.dt, (oo.t1-oo.t0)), df=oo.dfPSTH, knots=oo.kntsPSTH, include_intercept=True)    #  spline basis

            if oo.dfPSTH is None:
                oo.dfPSTH = oo.B.shape[1] 
            oo.B = oo.B.T    #  My convention for beta

            if oo.aS is None:
                oo.aS = _N.linalg.solve(_N.dot(oo.B, oo.B.T), _N.dot(oo.B, _N.ones(oo.t1 - oo.t0)*0.01))   #  small amplitude psth at first
            oo.u_a            = _N.zeros(oo.dfPSTH)
        else:
            oo.B = patsy.bs(_N.linspace(0, (oo.t1 - oo.t0)*oo.dt, (oo.t1-oo.t0)), df=4, include_intercept=True)    #  spline basis

            oo.B = oo.B.T    #  My convention for beta
            oo.aS = _N.zeros(4)

            #oo.u_a            = _N.ones(oo.dfPSTH)*_N.mean(oo.us)
            oo.u_a            = _N.zeros(oo.dfPSTH)

    def build_lrnLambda2(self, tr):
        oo = self
        #  lmbda2 is short snippet of after-spike depression behavior
        lrn = _N.ones(oo.N + 1)
        lh    = len(oo.l2)

        spkts = _N.where(oo.y[tr] == 1)[0]

        #  P(isi | t - t0 = t').  This prob is zero for isi < t-t0.  What is the shape of the distribution for longer isis?
        for t in spkts:
            maxL = lh if t + lh <= oo.N else oo.N - t
            lrn[t+1:t+1 + maxL] = oo.l2[0:maxL]

        ###  It stands to reason that at t=0, the actual state of the
        #    spiking history lambda is not always == 1, ie a spike
        #    occurred just slightly prior to t=0.  Let's just assume
        #    some virtual observation, and set
        bDone = False

        #print "!!!!!!!!!!!!!!!!!"
        #print oo.mean_isi_1st2spks

        times = -1
        while (not bDone) and (times < 50):
            times += 1
            ivrtISI = int(oo.mean_isi_1st2spks*_N.random.exponential())
            #print "%(1)d    %(2)d" % {"1" : ivrtISI, "2" : spkts[0]}
            if (ivrtISI > 2) and (ivrtISI > spkts[0]):
                bDone = True

        if not bDone:
            ivrtISI = 1  #  spkts[0] is SO large, don't even worry about history
        #  if vrtISI == oo.y[tr, 0] + 2, put virtual 2 bins back in time

        bckwds = ivrtISI - spkts[0]
        if (bckwds >= 0) and (bckwds < lh) :
            lrn[0:lh-bckwds] = oo.l2[bckwds:]

        return lrn

    def build_addHistory(self, ARo, smpx, BaS, us):
        oo = self
        for m in xrange(oo.TR):
            _N.exp(smpx[m] + BaS + us[m], out=oo.lrn_scr1) #ex
            _N.add(1, oo.lrn_scr1, out=oo.lrn_scr2)     # 1 + ex

            _N.divide(oo.lrn_scr1, oo.lrn_scr2, out=oo.lrn_scr3)  #ex / (1+ex)
            _N.multiply(oo.lrn_scr3, oo.lrn[m], out=oo.sprb[m])#(lam ex)/(1+ex)

            _N.exp(-smpx[m] - BaS - us[m], out=oo.lrn_iscr1)  #e{-x}
            _N.add(0.99, 0.99*oo.lrn_iscr1, out=oo.lrn_scld)  # 0.99(1 + e-x)
            sat = _N.where(oo.sprb[m] > 0.99)
            if len(sat[0]) > 0:
                print "bad loc   %(m)d     %(l)d" % {"m" : m, "l" : len(sat[0])}
                #fig = _plt.figure(figsize=(14, 3))
                #                _plt.plot(oo.lrn[m], lw=3, color="blue")
                #_plt.plot(oo.s_lrn[m], lw=2, color="red")

            oo.s_lrn[m, :] = oo.lrn[m]
            oo.s_lrn[m, sat[0]] = oo.lrn_scld[sat[0]]
            # if len(sat[0]) > 0:            
            #     _plt.plot(oo.s_lrn[m], lw=2)

            _N.log(oo.s_lrn[m] / (1 + (1 - oo.s_lrn[m])*oo.lrn_scr1), out=ARo[m])   #  history Offset   ####TRD change

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
        pickle.dump(pcklme, dmp)
        dmp.close()

        # import pickle
        # with open("mARp.dump", "rb") as f:
        # lm = pickle.load(f)


    def CIF(self, us, alps, osc):
        oo = self
        ooTR = oo.TR
        ooN  = oo.N
        ARo   = _N.empty((oo.TR, oo.N+1))

        BaS = _N.dot(oo.B.T, alps)
        oo.build_addHistory(ARo, osc, BaS, us)

        cif = _N.exp(us + ARo + osc + BaS) / (1 + _N.exp(us + ARo + osc + BaS))

        return cif
