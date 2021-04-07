#
from LOST.mcmcARpFuncs import loadL2, runNotes, loadKnown
from filter import bpFilt, lpFilt, gauKer
import LOST.mcmcAR as mAR
import LOST.ARlib as _arl
import pyPG as lw
#import logerfc as _lfc
import LOST.commdefs as _cd
import os
import numpy as _N
import re as _re
from LOST.ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import numpy.polynomial.polynomial as _Npp
from LOST.kflib import createDataAR, downsamplespkdat, ISIs
#
import LOST.splineknots as _spknts
import patsy
import pickle
import matplotlib.pyplot as _plt


class mcmcARspk(mAR.mcmcAR):
    ##  
    psthBurns     = 5
    k             = None
    Cn            = None;    Cs            = None;    C             = None
    kntsPSTH      = None;    dfPSTH        = None
    ID_q2         = False
    use_prior     = _cd.__COMP_REF__
    AR2lims       = None
    F_alfa_rep    = None
    pkldalfas     = None

    smpls_fn_incl_trls = None

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
    #lfc           = None

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
    psthknts           = 4

    #  Gibbs
    ARord         = _cd.__NF__
    
    #  Current values of params and state
    bpsth         = False
    B             = None;    aS            = None; 

    #  coefficient sampling
    fSigMax       = 500.    #  fixed parameters
    #freq_lims     = [[1 / .85, fSigMax]]
    sig_ph0L      = -1
    sig_ph0H      = 0

    # psth spline coefficient priors
    u_a          = None;             s2_a         = 2.

    #  knownSig
    knownSigFN      = None
    knownSig        = None
    xknownSig       = 1   #  multiply knownSig by...

    hS        = None

    dohist    = True
    outSmplFN = "smpls.dump"
    doBsmpx   = False
    BsmpxSkp  = 1

    t0_is_t_since_1st_spk = None    #  o set up spike history, generate a virtual spike before 1st spike in each trial

    #def __init__(self):
        # if (self.noAR is not None) or (self.noAR == False):
        #     self.lfc         = _lfc.logerfc()

    def loadDat(self, runDir, datfilename, trials, multiply_shape_hyperparam=1, multiply_scale_hyperparam=1, hist_timescale_ms=70, n_interior_knots=8): #################  loadDat
        oo = self
        hist_timescale = hist_timescale_ms*0.001
        bGetFP = False

        x_st_cnts = _N.loadtxt(datfilename)
        #y_ch      = 2   #  spike channel
        y_ch      = 0   #  spike channel
        #p = _re.compile("^\d{6}")   # starts like "exptDate-....."
        #m = p.match(oo.setname)

        #dch = 4    #  # of data columns per trial
        dch = 1

        bRealDat, dch = False, 1

        TR = x_st_cnts.shape[1] // dch    #  number of trials will get filtered

        print("TR    %d" % TR)
        print(trials)
        #  If I only want to use a small portion of the data
        oo.N   = x_st_cnts.shape[0] - 1
        if oo.t1 == None:
            oo.t1 = oo.N + 1
        #  meaning of N changes here
        N   = oo.t1 - 1 - oo.t0

        #x   = x_st_cnts[oo.t0:oo.t1, ::dch].T
        y   = x_st_cnts[oo.t0:oo.t1, y_ch::dch].T
        # if bRealDat:
        #     fx  = x_st_cnts[oo.t0:oo.t1, flt_ch::dch].T
        #     px  = x_st_cnts[oo.t0:oo.t1, ph_ch::dch].T

        ####  Now keep only trials that have spikes
        kpTrl = range(TR)
        if trials is None:
            trials = range(oo.TR)
        oo.useTrials = []
        for utrl in trials:
            try:
                ki = kpTrl.index(utrl)
                if _N.sum(y[utrl, :]) > 1:   #  must see at least 2 spikes
                    oo.useTrials.append(ki)
            except ValueError:
                print("a trial requested to use will be removed %d" % utrl)

        ######  oo.y are for trials that have at least 1 spike
        #y     = _N.array(y[oo.useTrials], dtype=_N.int)
        y     = _N.array(y, dtype=_N.int)

        if oo.downsamp:
            evry, dsdat = downsamplespkdat(y, 0.005, max_evry=3)
        else:
            evry = 1
            dsdat = y
            print("NO downsamp")
        oo.evry     = evry
        oo.dt *= oo.evry
        oo.fSigMax = 0.5/oo.dt
        print("fSigMax   %.3f" % oo.fSigMax)
        oo.freq_lims     = [[0.000001, oo.fSigMax]]*oo.C
        print(oo.freq_lims)

        print("oo.dt   %.3f" % oo.dt)

        print("!!!!!!!!!!!!!!!!!!!!!!   evry    %d" % evry)

        print(oo.useTrials)
        print(dsdat.shape)
        oo.y     = _N.array(dsdat[oo.useTrials], dtype=_N.int)        

        prb_spk_in_bin = _N.sum(oo.y) / (oo.y.shape[0] * oo.y.shape[1])
        oo.u_u   = -_N.log(1/prb_spk_in_bin - 1)
        print(oo.u_u)
        

        num_dat_pts = oo.y.shape[0] * oo.y.shape[1]
        if (oo.a_q2 is None) or (oo.B_q2 is None):
            #  we set a prior here
            #oo.a_q2 = num_dat_pts // 10
            oo.a_q2 = (num_dat_pts // 10) * multiply_shape_hyperparam
            #md = B / (a+1)   B = md
            oo.B_q2 = (1e-4 * (oo.a_q2 + 1) * evry) * multiply_scale_hyperparam
            print("setting prior for innovation %(a)d  %(B).3e" % {"a" : oo.a_q2, "B" : oo.B_q2})

        #oo.x     = _N.array(x[oo.useTrials])
        # if bRealDat:
        #     oo.fx    = _N.array(fx[oo.useTrials])
        #     oo.px    = _N.array(px[oo.useTrials])

        #  remove trials where data has no information
        rmTrl = []

        oo.kp  = oo.y - 0.5
        oo.rn  = 1

        oo.TR    = len(oo.useTrials)
        oo.N     = N
        oo.t1 = oo.t0 + dsdat.shape[1]
        oo.N  = oo.t1 - 1 - oo.t0

        #oo.Bsmpx        = _N.zeros((iters//oo.BsmpxSkp, oo.TR, (oo.N+1) + 2))
        oo.smpx        = _N.zeros((oo.TR, (oo.N + 1) + 2, oo.k))   #  start at 0 + u
        oo.ws          = _N.empty((oo.TR, oo.N+1), dtype=_N.float)
        oo.lrn   = _N.empty((oo.TR, oo.N+1))

        if oo.us is None:
            oo.us    = _N.zeros(oo.TR)

        tot_isi = 0
        nisi    = 0
        
        isis = ISIs(oo.y)
        
        #  cnts will always be 0 in frist bin
        sisis = _N.sort(isis)
        Lisi  = len(sisis)
        
        ###  look at the isi distribution

        #  cnts will always be 0 in frist bin
        maxisi = max(isis)
        minisi = min(isis)    #  >= 1

        print("*****************")
        print(maxisi)
        print(oo.N)
        print("*****************")
        cnts, bins = _N.histogram(isis, bins=_N.linspace(0.5, maxisi+0.5, maxisi+1))    #  cnts[0]   are number of ISIs of size 1


        smallisi   = int(sisis[int(Lisi*0.1)])
        #  hist_timescale in ms
        asymptote  = smallisi + int(hist_timescale / oo.dt)   #  100 ms

        hist_interior_knots  = _N.empty(n_interior_knots)
        lin01 = _N.linspace(0, 1, n_interior_knots, endpoint=True)
        sqr01 = lin01**2
        hist_interior_knots[0:8] = smallisi + sqr01 * (asymptote - smallisi)

        crats = _N.zeros(maxisi-1)
        for n in range(0, maxisi-2):
            crats[n+1] = crats[n] + cnts[n]
        crats /= crats[-1]

        ####  generate spike before time=0.  PSTH estimation
        if oo.t0_is_t_since_1st_spk is None:
            oo.t0_is_t_since_1st_spk = _N.empty(oo.TR, dtype=_N.int)
            rands = _N.random.rand(oo.TR)
            for tr in range(oo.TR):
                spkts = _N.where(oo.y[tr] == 1)[0]

                if len(spkts) > 0:
                    t0 = spkts[0]
                    t0 = t0 if t0 < len(crats) else len(crats) - 1
                    r0 = crats[t0]   # say 0.3   
                    adjRnd = (1 - r0) * rands[tr]
                    isi = _N.where(crats >= adjRnd)[0][0]  # isi in units of bin sz

                    oo.t0_is_t_since_1st_spk[tr] = isi
        else:
            print("using saved t0_is_t_since_1st_spk")

        oo.loghist = loadL2(runDir, fn=oo.histFN)
        oo.dohist = True if oo.loghist is None else False

        oo.knownSig = loadKnown(runDir, trials=oo.useTrials, fn=oo.knownSigFN) 
        if oo.knownSig is None:
            oo.knownSig = _N.zeros((oo.TR, oo.N+1))
        else:
            oo.knownSig *= oo.xknownSig

        ###  override knot locations

        upto = oo.N+1 if int(maxisi * 1.3) > oo.N+1 else int(maxisi * 1.3)
        print("upto   %d" % upto)
        print("oo.N   %d" % oo.N)
        print("maxisi %d" % maxisi)

        oo.Hbf = patsy.bs(_N.linspace(0, upto, upto+1, endpoint=False), knots=hist_interior_knots, include_intercept=True)    #  spline basisp

        max_locs = _N.empty(oo.Hbf.shape[1])
        for i in range(oo.Hbf.shape[1]):
            max_locs[i] = _N.where(oo.Hbf[:, i] == _N.max(oo.Hbf[:, i]))[0]
        print(max_locs)
        #  find the knot that's closest to   hist_interior_knots[4] (90th %tile)
        dist_from_90th = _N.abs(max_locs - asymptote)
        #print(dist_from_90th)
        oo.iHistKnotBeginFixed = _N.where(dist_from_90th == _N.min(dist_from_90th))[0][0]
        oo.histknots = oo.Hbf.shape[1]


    def allocateSmp(self, iters, Bsmpx=False):
        oo = self
        print("^^^^^^   allocateSmp  %d" % iters)
        ####  initialize
        if Bsmpx:
            oo.Bsmpx        = _N.zeros((iters//oo.BsmpxSkp, oo.TR, (oo.N+1) + 2))
            #oo.Bsmpx        = _N.zeros((iters//oo.BsmpxSkp, oo.TR, (oo.N+1) + 2))
        oo.smp_u        = _N.zeros((iters, oo.TR))
        oo.smp_hS        = _N.zeros((iters, oo.histknots))   # history spline
        oo.smp_hist        = _N.zeros((iters, oo.Hbf.shape[0]))   # history spline

        if oo.bpsth:
            oo.smp_aS        = _N.zeros((iters, oo.B.shape[0]))
        oo.smp_q2       = _N.zeros((iters, oo.TR))
        #  store samples of

        oo.allalfas     = _N.zeros((iters, oo.k), dtype=_N.complex)
        if oo.pkldalfas is not None:
            oo.allalfas[0]  = oo.pkldalfas
            for r in range(oo.R):
                oo.F_alfa_rep[r] = oo.pkldalfas[r].real
            for c in range(oo.C):
                oo.F_alfa_rep[oo.R+2*c]   = oo.pkldalfas[oo.R+2*c]
                oo.F_alfa_rep[oo.R+2*c+1] = oo.pkldalfas[oo.R+2*c + 1]
            print(oo.F_alfa_rep)
        oo.uts          = _N.empty((iters//oo.BsmpxSkp, oo.TR, oo.R, oo.N+1+1, 1))
        oo.wts          = _N.empty((iters//oo.BsmpxSkp, oo.TR, oo.C, oo.N+2+1, 1))
        oo.ranks        = _N.empty((iters, oo.C), dtype=_N.int)
        oo.fs           = _N.empty((iters, oo.C))
        oo.amps         = _N.empty((iters, oo.C))

        oo.mnStds       = _N.empty(iters)

    def setParams(self, psth_run=False, psth_knts=10):
        oo = self
        # #generate initial values of parameters
        #oo._d = _kfardat.KFARGauObsDat(oo.TR, oo.N, oo.k)
        #oo._d.copyData(oo.y)

        oo.Ns      = _N.ones(oo.TR, dtype=_N.int)*oo.N
        oo.ks      = _N.ones(oo.TR, dtype=_N.int)*oo.k

        oo.F     = _N.zeros((oo.k, oo.k))
        _N.fill_diagonal(oo.F[1:, 0:oo.k-1], 1)
        oo.F[0] =  _N.random.randn(oo.k)/_N.arange(1, oo.k+1)**2
        oo.F[0, 0] = 0.8
        oo.Fs    = _N.zeros((oo.TR, oo.k, oo.k))
        for tr in range(oo.TR):
            oo.Fs[tr] = oo.F
        oo.Ik    = _N.identity(oo.k)
        oo.IkN   = _N.tile(oo.Ik, (oo.N+1, 1, 1))

        #  need TR
        #  pr_x[:, 0]  empty, not used
        #oo.p_x   = _N.empty((oo.TR, oo.N+1, oo.k, 1)) 
        oo.p_x   = _N.empty((oo.TR, oo.N+1, oo.k)) 
        oo.p_x[:, 0, 0] = 0
        oo.p_V   = _N.empty((oo.TR, oo.N+1, oo.k, oo.k)) 
        oo.p_Vi  = _N.empty((oo.TR, oo.N+1, oo.k, oo.k)) 
        #oo.f_x   = _N.empty((oo.TR, oo.N+1, oo.k, 1)) 
        oo.f_x   = _N.empty((oo.TR, oo.N+1, oo.k)) 
        oo.f_V   = _N.empty((oo.TR, oo.N+1, oo.k, oo.k)) 
        #oo.s_x   = _N.empty((oo.TR, oo.N+1, oo.k, 1)) 
        oo.s_x   = _N.empty((oo.TR, oo.N+1, oo.k)) 
        oo.s_V   = _N.empty((oo.TR, oo.N+1, oo.k, oo.k)) 

        _N.fill_diagonal(oo.F[1:, 0:oo.k-1], 1)
        oo.G       = _N.zeros((oo.k, 1))
        oo.G[0, 0] = 1
        oo.Q       = _N.empty(oo.TR)


        #  baseFN_inter   baseFN_comps   baseFN_comps

        print("freq_lims")
        print(oo.freq_lims)
        radians      = buildLims(0, oo.freq_lims, nzLimL=1., Fs=(1/oo.dt))
        oo.AR2lims      = 2*_N.cos(radians)

        oo.smpx        = _N.zeros((oo.TR, (oo.N + 1) + 2, oo.k))   #  start at 0 + u
        oo.ws          = _N.empty((oo.TR, oo.N+1), dtype=_N.float)

        #############   ADDED THIS FOR DEBUG
        #oo.F_alfa_rep = _N.array([-0.4       +0.j,          0.96999828+0.00182841j,  0.96999828-0.00182841j, 0.51000064+0.02405102j,  0.51000064-0.02405102j,  0.64524011+0.04059507j, 0.64524011-0.04059507j]).tolist()
        if oo.F_alfa_rep is None:
            oo.F_alfa_rep  = initF(oo.R, oo.Cs+oo.Cn, 0).tolist()   #  init F_alfa_rep
        print("F_alfa_rep*********************")
        print(oo.F_alfa_rep)

        #print(ampAngRep(oo.F_alfa_rep))
        if oo.q20 is None:
            oo.q20         = 0.00077
        oo.q2          = _N.ones(oo.TR)*oo.q20

        oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]
        oo.Fs    = _N.zeros((oo.TR, oo.k, oo.k))

        oo.F[0] = oo.F0
        _N.fill_diagonal(oo.F[1:, 0:oo.k-1], 1)

        for tr in range(oo.TR):
            oo.Fs[tr] = oo.F


        ########  Limit the amplitude to something reasonable
        xE, nul = createDataAR(oo.N, oo.F0, oo.q20, 0.1)
        mlt  = _N.std(xE) / 0.5    #  we want amplitude around 0.5
        oo.q2          /= mlt*mlt
        xE, nul = createDataAR(oo.N, oo.F0, oo.q2[0], 0.1)

        w  =  5
        wf =  gauKer(w)
        gk = _N.empty((oo.TR, oo.N+1))
        fgk= _N.empty((oo.TR, oo.N+1))
        for m in range(oo.TR):
            gk[m] =  _N.convolve(oo.y[m], wf, mode="same")
            gk[m] =  gk[m] - _N.mean(gk[m])
            gk[m] /= 5*_N.std(gk[m])
            fgk[m] = bpFilt(15, 100, 1, 135, 500, gk[m])   #  we want
            fgk[m, :] /= 3*_N.std(fgk[m, :])

            if oo.noAR:
                oo.smpx[m, 2:, 0] = 0
            else:
                oo.smpx[m, 2:, 0] = fgk[m, :]

            for n in range(2+oo.k-1, oo.N+1+2):  # CREATE square smpx
                oo.smpx[m, n, 1:] = oo.smpx[m, n-oo.k+1:n, 0][::-1]
            for n in range(2+oo.k-2, -1, -1):  # CREATE square smpx
                oo.smpx[m, n, 0:oo.k-1] = oo.smpx[m, n+1, 1:oo.k]
                oo.smpx[m, n, oo.k-1] = _N.dot(oo.F0, oo.smpx[m, n:n+oo.k, oo.k-2]) # no noise

        if oo.bpsth:
            psthKnts, apsth, aWeights = _spknts.suggestPSTHKnots(oo.dt, oo.TR, oo.N+1, oo.y.T, psth_knts=psth_knts, psth_run=psth_run)

            _N.savetxt("apsth.txt", apsth, fmt="%.4f")
            _N.savetxt("psthKnts.txt", psthKnts, fmt="%.4f")
                
            apprx_ps = _N.array(_N.abs(aWeights))
            oo.u_a   = -_N.log(1/apprx_ps - 1)

            #  For oo.u_a, use the values we get from aWeights 

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



    def stitch_Hist(self, ARo, hcrv, stsM):  # history curve
        #  this has no direct bearing on sampling of history knots
        #  however, 
        oo = self
        # print("N+1   %d" % (oo.N+1))
        for m in range(oo.TR):
            sts = stsM[m]
            for it in range(len(sts)-1):
                t0 = sts[it]
                t1 = sts[it+1]
                #ARo[m, t0+1:t1+1] = hcrv[t0-t0+1:t1-t0+1]#hcrv[t0-t0:t1-t0]
                ARo[m, t0+1:t1+1] = hcrv[t0-t0:t1-t0]#hcrv[t0-t0+1:t1-t0+1]#
            T = oo.N+1 - sts[len(sts)-1]
            t1= sts[len(sts)-1]   #  if len(sts) == 1, didn't do for loop
            
            #ARo[m, t1+1:] = hcrv[1:T]#hcrv[0:T-1]
            #print(".............")
            #print("m   %(m)d   t1=%(t1)d   T=%(T)d    len(hcrv)=%(lh)d" % {"m" : m, "t1" : t1, "T" : T, "lh" : len(hcrv)})
            #print(sts)
            #print("left %(l)d   right %(r)d" % {"l" : len(ARo[m, t1+1:]), "r" : len(hcrv[0:T-1])})
            #  will fail if hcrv is too short, ie T is > len(hcrv)
            if hcrv.shape[0] > T - 1:
                ARo[m, t1+1:] = hcrv[0:T-1]#hcrv[1:T]#
            else:
                ARo[m, t1+1:t1+hcrv.shape[0]+1] = hcrv[0:T-1]#hcrv[1:T]#
                ARo[m, t1+hcrv.shape[0]+1:] = hcrv[-1]#hcrv[1:T]#
            isiHiddenPrt = oo.t0_is_t_since_1st_spk[m] + 1

            ARo[m, 0:sts[0]+1] = hcrv[isiHiddenPrt:isiHiddenPrt + sts[0]+1]


    # def getComponents(self):
    #     oo    = self
    #     TR    = oo.TR
    #     NMC   = oo.NMC
    #     burn  = oo.burn
    #     R     = oo.R
    #     C     = oo.C
    #     ddN   = oo.N

    #     oo.rts = _N.empty((TR, burn+NMC, ddN+2, R))    #  real components   N = ddN
    #     oo.zts = _N.empty((TR, burn+NMC, ddN+2, C))    #  imag components 

    #     for tr in range(TR):
    #         for it in range(1, burn + NMC):
    #             b, c = dcmpcff(alfa=oo.allalfas[it])
    #             print(b)
    #             print(c)
    #             for r in range(R):
    #                 oo.rts[tr, it, :, r] = b[r] * oo.uts[tr, it, r, :]

    #             for z in range(C):
    #                 #print "z   %d" % z
    #                 cf1 = 2*c[2*z].real
    #                 gam = oo.allalfas[it, R+2*z]
    #                 cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
    #                 oo.zts[tr, it, 0:ddN+2, z] = cf1*oo.wts[tr, it, z, 1:ddN+3] - cf2*oo.wts[tr, it, z, 0:ddN+2]

    #     oo.zts0 = _N.array(oo.zts[:, :, 1:, 0], dtype=_N.float16)

    def CIF(self, gibbsIter=0):
        """
        us     offset                     TR x 1
        alps   psth spline weights        
        hS     history spline weights
        osc                               TR x Nm1
        """
        oo = self
        usr  = oo.smp_u[gibbsIter].reshape((oo.TR, 1))
        alps = oo.smp_aS[gibbsIter]  #  this is last
        hS   = oo.smp_hS[gibbsIter]  #  this is last
        osc  = oo.Bsmpx[gibbsIter//oo.BsmpxSkp, 2:]

        ARo   = _N.empty((oo.TR, oo.N+1))


        BaS = _N.dot(oo.B.T, alps)
        Msts = []
        for m in range(oo.TR):
            Msts.append(_N.where(oo.y[m] == 1)[0])

        oo.loghist = _N.zeros(oo.N+1)
        _N.dot(oo.Hbf, hS, out=oo.loghist)

        oo.stitch_Hist(ARo, oo.loghist, Msts)

        cif = _N.exp(usr + ARo + osc + BaS + oo.knownSig) / (1 + _N.exp(usr + ARo + osc + BaS + oo.knownSig))

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
        _plt.savefig("%(od)s/chosenFsAmps%(sI)d" % {"od" : oo.outdir, "sI" : startIt})
        _plt.close()

        indsFs = _N.where((oo.fs[startIt:, 0] >= loF) & (oo.fs[startIt:, 0] <= hiF))
        indsAs = _N.where((oo.amps[startIt:, 0] >= loA) & (oo.amps[startIt:, 0] <= hiA))

        asfsInds = _N.intersect1d(indsAs[0], indsFs[0]) + startIt
        q = _N.mean(oo.smp_q2[0, startIt:])


        #alfas = _N.mean(oo.allalfas[asfsInds], axis=0)
        pcklme = [aus, q, oo.allalfas[asfsInds], aSs, oo.B, hSs, oo.Hbf]
        
        dmp = open("%(od)s/posteriorModes%(sI)d.pkl" % {"od" : oo.outdir, "sI" : startIt}, "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()

    
    def dump_smps(self, frm, pcklme=None, dir=None, toiter=None, smpls_fn_incl_trls=False):
        oo    = self
        if pcklme is None:
            pcklme = {}

        toiter         = oo.last_iter+1 if (toiter is None) else toiter
        if oo.bpsth:
            pcklme["aS"]   = oo.smp_aS[0:toiter:oo.BsmpxSkp]  #  this is last
        pcklme["frm"]    = frm
        pcklme["evry"]    = oo.evry
        pcklme["B"]    = oo.B
        pcklme["R"]    = oo.R
        pcklme["C"]    = oo.C
        pcklme["k"]    = oo.k
        pcklme["BsmpxSkp"]    = oo.BsmpxSkp
        pcklme["rts"]    = _N.array(oo.rts, dtype=_N.float16)  #  resolution about 1e-6.  So as long as signal amplitude about 0.1, this is not a problem
        pcklme["zts"]    = _N.array(oo.zts, dtype=_N.float16)
        pcklme["toiter"]      = toiter
        pcklme["q2"]   = oo.smp_q2[0:toiter:oo.BsmpxSkp]
        pcklme["amps"] = oo.amps[0:toiter:oo.BsmpxSkp]
        pcklme["fs"]   = oo.fs[0:toiter:oo.BsmpxSkp]
        pcklme["u"]    = oo.smp_u[0:toiter:oo.BsmpxSkp]
        pcklme["mnStds"]= oo.mnStds[0:toiter:oo.BsmpxSkp]
        pcklme["allalfas"]= oo.allalfas[0:toiter:oo.BsmpxSkp]
        pcklme["smpx"] = oo.smpx
        pcklme["ws"]   = oo.ws
        pcklme["useTrials"]   = oo.useTrials
        pcklme["t0_is_t_since_1st_spk"] = oo.t0_is_t_since_1st_spk
        if oo.Hbf is not None:
            pcklme["spkhist"] = oo.smp_hist[0:toiter:oo.BsmpxSkp]
            pcklme["Hbf"]    = oo.Hbf
            pcklme["h_coeffs"]    = oo.smp_hS[0:toiter:oo.BsmpxSkp]
        if oo.doBsmpx:
            pcklme["Bsmpx"]    = _N.array(oo.Bsmpx[0:toiter//oo.BsmpxSkp], dtype=_N.float16)

        #cifs = _N.empty((oo.TR, oo.N
        #for it 
        #oo.CIF(us, alps, hS, osc
            

        print("saving state in %s" % oo.outSmplFN)
        if dir is None:
            if smpls_fn_incl_trls is None:
                dmp = open(oo.outSmplFN, "wb")
            else:
                dmp = open("%(bf)s_%(tr0)d_%(tr1)d" % {"bf" : oo.outSmplFN, "tr0" : int(_N.min(oo.useTrials)), "tr1" : int(_N.max(oo.useTrials))}, "wb")
        else:
            if smpls_fn_incl_trls is None:
                dmp = open("%(d)s/%(sfn)s" % {"d" : dir, "sfn" : oo.outSmplFN}, "wb")
            else:
                dmp = open("%(d)s/%(bf)s_%(tr0)d_%(tr1)d" % {"d" : dir, "bf" : oo.outSmplFN, "tr0" : int(_N.min(oo.useTrials)), "tr1" : int(_N.max(oo.useTrials))}, "wb")

        pickle.dump(pcklme, dmp, -1)
        dmp.close()

