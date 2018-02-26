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

# from mcmcARpFuncs import loadL2, runNotes
# from filter import bpFilt, lpFilt, gauKer
# import mcmcAR as mAR
# import ARlib as _arl
# import pyPG as lw
# import kfardat as _kfardat
# import logerfc as _lfc
# import commdefs as _cd
# import os
# import numpy as _N
# from kassdirs import resFN, datFN
# import re as _re
# from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
# import numpy.polynomial.polynomial as _Npp
# from kflib import createDataAR
# import patsy

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

    histknots     = 10

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
    u_a          = None;             s2_a         = 2

    h0_1      = None        # silence following spike
    h0_2      = None        # inhibitory rebound peak
    h0_3      = None        # decayed away peaky part 
    h0_4      = None        # far
    h0_5      = None        # farther

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
        isis = []
        for tr in xrange(oo.TR):
            spkts = _N.where(oo.y[tr] == 1)[0]
            isis.extend(_N.diff(spkts))
        
        #  cnts will always be 0 in frist bin
        sisis = _N.sort(isis)
        Lisi  = len(sisis)
        
        cnts, bins = _N.histogram(isis, bins=_N.linspace(0, oo.N+1, oo.N+2))

        ###  look at the isi distribution
        if (oo.h0_1 is None) or (oo.h0_2 is None):
            ii = 0
            while cnts[ii] == 0:
                ii += 1
            oo.h0_1 = ii  #  firing prob is 0, oo.h0_1 ms postspike
            oo.h0_2 = _N.where(cnts == _N.max(cnts))[0][0]
            oo.h0_2 = oo.h0_1 + 1 if oo.h0_1 >= oo.h0_2 else oo.h0_2
            # while cnts[ii] < 0.5*(cnts[ii+1]+cnts[ii+2]):
            #     ii += 1
            # oo.h0_2 = ii  #  approx peak of post-spike rebound
            
        oo.h0_3= oo.h0_2*3
        oo.h0_4 = int(sisis[int(Lisi*0.7)])
        oo.h0_5 = int(sisis[int(Lisi*0.8)])
        if oo.h0_3 > oo.h0_4:
            oo.h0_4 = oo.h0_2 * 4
        if oo.h0_4 > oo.h0_5:
            oo.h0_5 = oo.h0_4 + 10

        crats = _N.zeros(oo.N+2)
        for n in xrange(0, oo.N+1):
            crats[n+1] = crats[n] + cnts[n]
        crats /= crats[-1]

        ####  generate spike before time=0.  PSTH estimation
        oo.t0_is_t_since_1st_spk = _N.empty(oo.TR, dtype=_N.int)
        rands = _N.random.rand(oo.TR)
        for tr in xrange(oo.TR):
            spkts = _N.where(oo.y[tr] == 1)[0]

            if len(spkts) > 0:
                t0 = spkts[0]
                r0 = crats[t0]   # say 0.3   
                adjRnd = (1 - r0) * rands[tr]
                isi = _N.where(crats >= adjRnd)[0][0]  # isi in units of bin sz

                oo.t0_is_t_since_1st_spk[tr] = isi


    def allocateSmp(self, iters):
        oo = self
        print "^^^^^^   allocateSmp  %d" % iters
        ####  initialize
        oo.Bsmpx        = _N.zeros((oo.TR, iters, oo.N+1))
        oo.smp_u        = _N.zeros((oo.TR, iters))
        oo.smp_hS        = _N.zeros((oo.histknots, iters))   # history spline
        oo.smp_hist        = _N.zeros((oo.N+1, iters))   # history spline

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
            #oo.u_a            = _N.zeros(oo.dfPSTH)

        print "h0_1 %(1)d  h0_2 %(2)d  h0_3 %(3)d  h0_4 %(4)d  h0_5 %(5)d" % {"1" : oo.h0_1, "2" : oo.h0_2, "3" : oo.h0_3, "4" : oo.h0_4, "5" : oo.h0_5}
        oo.Hbf = patsy.bs(_N.linspace(0, (oo.N+1), oo.N+1, endpoint=False), knots=_N.array([oo.h0_1, oo.h0_2, oo.h0_3, oo.h0_4, oo.h0_5, int(0.7*(oo.N+1))]), include_intercept=True)    #  spline basisp

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
