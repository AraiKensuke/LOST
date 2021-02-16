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
    Cn            = None;    Cs            = None;    C             = None
    kntsPSTH      = None;    dfPSTH        = None
    ID_q2         = False
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

    #def __init__(self):
        # if (self.noAR is not None) or (self.noAR == False):
        #     self.lfc         = _lfc.logerfc()

    def loadDat(self, runDir, datfilename, trials, h0_1=None, h0_2=None, h0_3=None, h0_4=None, h0_5=None, multiply_shape_hyperparam=1, multiply_scale_hyperparam=1): #################  loadDat
        oo = self
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
                if _N.sum(y[utrl, :]) > 0:
                    oo.useTrials.append(ki)
            except ValueError:
                print("a trial requested to use will be removed %d" % utrl)

        ######  oo.y are for trials that have at least 1 spike
        #y     = _N.array(y[oo.useTrials], dtype=_N.int)
        y     = _N.array(y, dtype=_N.int)

        if oo.downsamp:
            evry, dsdat = downsamplespkdat(y, 0.01, max_evry=5)
        else:
            evry = 1
            dsdat = y
            print("NO downsamp")
        oo.evry     = evry
        oo.dt *= oo.evry

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

        cnts, bins = _N.histogram(isis, bins=_N.linspace(0.5, maxisi+0.5, maxisi+1))    #  cnts[0]   are number of ISIs of size 1
        p9 = sisis[int(len(sisis)*0.9)]

        x = bins[minisi:p9]
        y = cnts[minisi-1:p9-1]
        z = _N.polyfit(x, y, 5)
        ply = _N.poly1d(z)
        plyx= ply(x)
        #imax = _N.where(plyx == _N.max(plyx))[0][0] + minisi
        imax = _N.where(cnts == _N.max(cnts))[0][0] + minisi

        if (imax == 1):   #  no rebound excitation or pause
            print("imax == 1")
            oo.h0_1 = 1
            oo.h0_2 = int(sisis[int(Lisi*0.3)]*2)#oo.h0_2*3
            oo.h0_3 = int(sisis[int(Lisi*0.6)]*2)#oo.h0_2*3
            oo.h0_4 = int(sisis[int(Lisi*0.9)]*2)#oo.h0_2*3
            print("-----  %(1)d  %(2)d  %(3)d  %(4)d" % {"1" : oo.h0_1, "2" : oo.h0_2, "3" : oo.h0_3, "4" : oo.h0_4})
            oo.hist_max_at_0 = True
            oo.histknots = 9
        else:      #  a pause
            print("pause")
            ii = 1
            # while cnts[ii] == 0:
            #     ii += 1
            # oo.h0_1 = ii  #  firing prob is 0, oo.h0_1 ms postspike

            #pts = _N.array([imax, int(imnisi*1.5), int(sisis[int(Lisi*0.97)]*1.5), int(sisis[int(Lisi*0.97)])*2.5])   #12
            #pts = _N.array([imax, int(imnisi*0.7), int(sisis[int(Lisi*0.97)]*0.8), int(sisis[int(Lisi*0.97)])*1.5])    # 13
            pts = _N.array([sisis[int(Lisi*0.05)], sisis[int(Lisi*0.15)], sisis[int(Lisi*0.5)], sisis[int(Lisi*0.7)], sisis[int(Lisi*0.9)]])    # 14
            #pts = _N.array([imax, int(imnisi), int(sisis[int(Lisi*0.97)]), int(sisis[int(Lisi*0.97)])*2])
            #imnisi = int(_N.mean(isis)*0.9)
            #pts = _N.array([imax, imnisi, int(0.4*(sisis[int(Lisi*0.97)] - imnisi) + imnisi), int(sisis[int(Lisi*0.97)])])
            #pts = _N.array([imax, imnisi, int(0.4*(sisis[int(Lisi*0.97)] - imnisi) + imnisi)*3, int(sisis[int(Lisi*0.97)])*3])
            #pts = _N.array([imax, int(imnisi*1.2), int(imnisi*2.5), int(sisis[int(Lisi*0.97)])*3])

            #print("imax    %d" % imax)
            #pts = _N.array([int(imax*0.9), imnisi, int(0.3*(sisis[int(Lisi*0.9)] - imnisi) + imnisi), int(sisis[int(Lisi*0.9)])])
            #pts = _N.array([imax, imnisi, int(0.25*(sisis[int(Lisi*0.85)] - imnisi) + imnisi), int(sisis[int(Lisi*0.85)])])
            #pts = _N.array([imax, imnisi, int(0.5*(sisis[int(Lisi*0.995)] - imnisi) + imnisi), int(sisis[int(Lisi*0.995)])])
            #pts = _N.array([imax, imnisi, int(0.4*(sisis[int(Lisi*0.995)] - imnisi) + imnisi), int(sisis[int(Lisi*0.995)])])

            #pts = _N.array([19, 21, 23, 33])  #  quick hack for f64-1-Xaa/wp_0-60_5_1a
            spts = _N.sort(pts)
            # for i in range(3):
            #     if spts[i] == spts[i+1]:
            #         spts[i+1] += 1
            oo.h0_1 = spts[0]
            oo.h0_2 = spts[1]
            oo.h0_3 = spts[2]
            oo.h0_4 = spts[3]
            oo.h0_5 = spts[4]
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

            print("hist knot locations-----  %(1)d  %(2)d  %(3)d  %(4)d  %(5)d" % {"1" : oo.h0_1, "2" : oo.h0_2, "3" : oo.h0_3, "4" : oo.h0_4, "5" : oo.h0_5})

            oo.h0_1 = oo.h0_1 if h0_1 is None else h0_1
            oo.h0_2 = oo.h0_2 if h0_2 is None else h0_2
            oo.h0_3 = oo.h0_3 if h0_3 is None else h0_3
            oo.h0_4 = oo.h0_4 if h0_4 is None else h0_4
            oo.h0_5 = oo.h0_5 if h0_5 is None else h0_5

            print("hist knot locations -----  %(1)d  %(2)d  %(3)d  %(4)d  %(5)d   (overridden?)" % {"1" : oo.h0_1, "2" : oo.h0_2, "3" : oo.h0_3, "4" : oo.h0_4, "5" : oo.h0_5})

            oo.hist_max_at_0 = False
            oo.histknots = 10
        oo.maxISI  = int(sisis[int(Lisi*0.99)])


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

    def allocateSmp(self, iters, Bsmpx=False):
        oo = self
        print("^^^^^^   allocateSmp  %d" % iters)
        ####  initialize
        if Bsmpx:
            oo.Bsmpx        = _N.zeros((oo.TR, iters//oo.BsmpxSkp, (oo.N+1) + 2))
            #oo.Bsmpx        = _N.zeros((iters//oo.BsmpxSkp, oo.TR, (oo.N+1) + 2))
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
            for r in range(oo.R):
                oo.F_alfa_rep[r] = oo.pkldalfas[r].real
            for c in range(oo.C):
                oo.F_alfa_rep[oo.R+2*c]   = oo.pkldalfas[oo.R+2*c]
                oo.F_alfa_rep[oo.R+2*c+1] = oo.pkldalfas[oo.R+2*c + 1]
            print(oo.F_alfa_rep)
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

        radians      = buildLims(oo.Cn, oo.freq_lims, nzLimL=1.)
        oo.AR2lims      = 2*_N.cos(radians)

        oo.smpx        = _N.zeros((oo.TR, (oo.N + 1) + 2, oo.k))   #  start at 0 + u
        oo.ws          = _N.empty((oo.TR, oo.N+1), dtype=_N.float)

        if oo.F_alfa_rep is None:
            oo.F_alfa_rep  = initF(oo.R, oo.Cs, oo.Cn, ifs=oo.ifs).tolist()   #  init F_alfa_rep

        print(ampAngRep(oo.F_alfa_rep))
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
            psthKnts, apsth, aWeights = _spknts.suggestPSTHKnots(oo.dt, oo.TR, oo.N+1, oo.y.T, iknts=4)
            _N.savetxt("apsth.txt", apsth, fmt="%.4f")
            _N.savetxt("psthKnts.txt", psthKnts, fmt="%.4f")

            apprx_ps = _N.array(_N.abs(aWeights))
            oo.u_a   = -_N.log(1/apprx_ps - 1)

            #  For oo.u_a, use the values we get from aWeights 

            print(psthKnts)

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
        farknot = oo.maxISI*1.1# < (oo.t1-oo.t0) if oo.maxISI*2  else int((oo.t1-oo.t0) *0.9)
        if oo.hist_max_at_0:
            print("!!!!!!!!!!!!!!!!!!   here 1")
            print(_N.array([oo.h0_1, oo.h0_2, oo.h0_3, oo.h0_4, farknot]))
            oo.Hbf = patsy.bs(_N.linspace(0, (oo.N+1), oo.N+1, endpoint=False), knots=_N.array([oo.h0_1, oo.h0_2, oo.h0_3, oo.h0_4, farknot]), include_intercept=True)    #  spline basisp
            print(oo.Hbf)
        else:
            print("!!!!!!!!!!!!!!!!!!   here 2")
            print(_N.array([oo.h0_1, oo.h0_2, oo.h0_3, oo.h0_4, oo.h0_5, farknot]))
            oo.Hbf = patsy.bs(_N.linspace(0, (oo.N+1), oo.N+1, endpoint=False), knots=_N.array([oo.h0_1, oo.h0_2, oo.h0_3, oo.h0_4, oo.h0_5, farknot]), include_intercept=True)    #  spline basisp
            print(oo.Hbf)


    def stitch_Hist(self, ARo, hcrv, stsM):  # history curve
        #  this has no direct bearing on sampling of history knots
        #  however, 
        oo = self
        # print("ARo.shape?????????")
        # print(ARo.shape)
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
            ARo[m, t1+1:] = hcrv[0:T-1]#hcrv[1:T]#
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

        for tr in range(TR):
            for it in range(1, burn + NMC):
                b, c = dcmpcff(alfa=oo.allalfas[it])
                print(b)
                print(c)
                for r in range(R):
                    oo.rts[tr, it, :, r] = b[r] * oo.uts[tr, it, r, :]

                for z in range(C):
                    #print "z   %d" % z
                    cf1 = 2*c[2*z].real
                    gam = oo.allalfas[it, R+2*z]
                    cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
                    oo.zts[tr, it, 0:ddN+2, z] = cf1*oo.wts[tr, it, z, 1:ddN+3] - cf2*oo.wts[tr, it, z, 0:ddN+2]

        oo.zts0 = _N.array(oo.zts[:, :, 1:, 0], dtype=_N.float16)

    def readdump(self):
        oo    = self

        with open("mARp.dump", "rb") as f:
            lm = pickle.load(f)
        f.close()
        oo.F_alfa_rep = lm[0].allalfas[-1].tolist()
        oo.q20 = lm[0].q2[0]
        oo.aS  = lm[0].aS
        oo.us  = lm[0].us


    #def CIF(self, us=None, alps=None, hS=None, osc=None, smplInd0=None, smplInd1=None):

    """
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
    """
    def CIF(self, gibbsIter=0):
        """
        us     offset                     TR x 1
        alps   psth spline weights        
        hS     history spline weights
        osc                               TR x Nm1
        """
        oo = self
        usr  = oo.smp_u[:, gibbsIter].reshape((oo.TR, 1))
        alps = oo.smp_aS[gibbsIter]  #  this is last
        hS   = oo.smp_hS[:, gibbsIter]  #  this is last
        osc  = oo.Bsmpx[:, gibbsIter//oo.BsmpxSkp, 2:]

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

    
    def dump_smps(self, frm, pcklme=None, dir=None, toiter=None):
        oo    = self
        if pcklme is None:
            pcklme = {}

        toiter         = oo.last_iter if (toiter is None) else toiter
        if oo.bpsth:
            pcklme["aS"]   = oo.smp_aS[0:toiter]  #  this is last
        pcklme["frm"]    = frm
        pcklme["evry"]    = oo.evry
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
            pcklme["Bsmpx"]    = oo.Bsmpx[0:toiter//oo.BsmpxSkp]

        #cifs = _N.empty((oo.TR, oo.N
        #for it 
        #oo.CIF(us, alps, hS, osc
            

        print("saving state in %s" % oo.outSmplFN)
        if dir is None:
            dmp = open(oo.outSmplFN, "wb")
        else:
            dmp = open("%(d)s/%(sfn)s" % {"d" : dir, "sfn" : oo.outSmplFN}, "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()

        # import pickle
        # with open("smpls.dump", "rb") as f:
        # lm = pickle.load(f)
