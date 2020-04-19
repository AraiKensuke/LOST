import pickle
from kflib import createDataAR
import numpy as _N
import patsy
import re as _re
import matplotlib.pyplot as _plt

import scipy.stats as _ss
from LOSTdirs import resFN, datFN

import numpy.polynomial.polynomial as _Npp
import time as _tm
import ARlib as _arl
import kfARlibMPmv_ram2 as _kfar
import pyPG as lw
from ARcfSmplNoMCMC import ARcfSmpl
#from ARcfSmpl2 import ARcfSmpl

import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import os

import mcmcARspk as mcmcARspk
import monitor_gibbs as _mg

class mcmcARp(mcmcARspk.mcmcARspk):
    #  Description of model
    rn            = None    #  used for count data
    k             = None
    Cn            = None;    Cs            = None;    C             = None
    kntsPSTH      = None;    dfPSTH        = None
    use_prior     = _cd.__COMP_REF__
    AR2lims       = None
    F_alfa_rep    = None

    #  Sampled 
    ranks         = None
    pgs           = None
    fs            = None
    amps          = None
    dt            = None
    mcmcRunDir    = None

    ####  TEMPORARY
    Bi            = None
    rsds          = None

    bOMP          = False    #  use openMP

    #  Gibbs
    ARord         = _cd.__NF__
    x             = None   #  true latent state
    #  Current values of params and state
    B             = None;    aS            = None; us             = None;    

    #  coefficient sampling
    fSigMax       = 500.    #  fixed parameters
    #freq_lims     = [[1 / .85, fSigMax]]
    freq_lims     = [[.1, fSigMax]]
    sig_ph0L      = -1
    sig_ph0H      = 0

    #  1 offset for all trials
    bIndOffset    = True
    peek          = 400

    VIS           = None

    ignr          = 0

    def loadDat(self, trials): #################  loadDat
        oo = self
        bGetFP = False

        x_st_cnts = _N.loadtxt(resFN("lat_obs.dat", dir=oo.setname))
        y_ch      = 1   #  spike channel
        p = _re.compile("^\d{6}")   # starts like "exptDate-....."
        m = p.match(oo.setname)

        bRealDat, dch = False, 2

        TR = len(trials)

        #  If I only want to use a small portion of the data
        oo.N   = x_st_cnts.shape[0] - 1
        if oo.t1 == None:
            oo.t1 = oo.N + 1
        #  meaning of N changes here
        N   = oo.t1 - 1 - oo.t0

        vTrials = _N.array(trials)
        x   = x_st_cnts[oo.t0:oo.t1, dch*trials].T
        y   = x_st_cnts[oo.t0:oo.t1, dch*trials+1].T

        ######  oo.y are for trials that have at least 1 spike
        oo.y     = _N.array(y)
        oo.x     = _N.array(x)

        oo.TR    = TR
        oo.N     = N

        oo.smpx        = _N.zeros((oo.TR, (oo.N + 1) + 2, oo.k))   #  start at 0 + u

        
    def allocateSmp(self, iters, Bsmpx=False):
        oo = self
        print "^^^^^^   allocateSmp  %d" % iters
        ####  initialize
        if Bsmpx:
            oo.Bsmpx        = _N.zeros((oo.TR, iters/oo.BsmpxSkp, (oo.N+1) + 2))

        oo.smp_q2       = _N.zeros((oo.TR, iters))
        oo.smp_oq2       = _N.zeros((oo.TR, iters))        
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


        
    def gibbsSamp(self):  ###########################  GIBBSSAMPH
        oo          = self

        ooTR        = oo.TR
        ook         = oo.k

        ooN         = oo.N
        _kfar.init(oo.N, oo.k, oo.TR)
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))

        print "oo.mcmcRunDir    %s" % oo.mcmcRunDir
        if oo.mcmcRunDir is None:
            oo.mcmcRunDir = ""
        elif (len(oo.mcmcRunDir) > 0) and (oo.mcmcRunDir[-1] != "/"):
            oo.mcmcRunDir += "/"

        it    = -1

        #runTO = ooNMC + oo.burn - 1 if (burns is None) else (burns - 1)
        runTO = oo.burn - 1
        oo.allocateSmp(runTO+1, Bsmpx=oo.doBsmpx)
        alpR   = oo.F_alfa_rep[0:oo.R]
        alpC   = oo.F_alfa_rep[oo.R:]

        Msts = []
        for m in xrange(ooTR):
            Msts.append(_N.where(oo.y[m] == 1)[0])

        ##  ORDER OF SAMPLING
        ##  f_xx, f_V
        ##  DA:  PG, kpOws
        ##  history, build ARo
        ##  psth
        ##  offset
        ##  DA:  latent state
        ##  AR coefficients
        ##  q2

        K     = _N.empty((oo.TR, oo.N + 1, oo.k))   #  kalman gain

        iterBLOCKS  = oo.burn/oo.peek
        smpx_tmp = _N.empty((oo.TR, oo.N+1, oo.k))

        ws   = _N.ones((oo.TR, oo.N+1))*0.001
        ######  Gibbs sampling procedure
        for itrB in xrange(iterBLOCKS):
            for it in xrange(itrB*oo.peek, (itrB+1)*oo.peek):
                ttt1 = _tm.time()

                if (it % 10) == 0:
                    print it
                #  generate latent AR state
                oo.f_x[:, 0]     = oo.x00
                if it == 0:
                    for m in xrange(ooTR):
                        oo.f_V[m, 0]     = oo.s2_x00
                else:
                    oo.f_V[:, 0]     = _N.mean(oo.f_V[:, 1:], axis=1)

                #  _d.F, _d.N, _d.ks, 
                _kfar.armdl_FFBS_1itrMP(oo.y, ws, oo.Fs, _N.linalg.inv(oo.Fs), oo.q2, oo.Ns, oo.ks, oo.f_x, oo.f_V, oo.p_x, oo.p_V, smpx_tmp, K)

                oo.smpx[:, 2:]           = smpx_tmp
                oo.smpx[:, 1, 0:ook-1]   = oo.smpx[:, 2, 1:]
                oo.smpx[:, 0, 0:ook-2]   = oo.smpx[:, 2, 2:]

                if oo.doBsmpx and (it % oo.BsmpxSkp == 0):
                    oo.Bsmpx[:, it / oo.BsmpxSkp, 2:]    = oo.smpx[:, 2:, 0]
                stds = _N.std(oo.smpx[:, 2+oo.ignr:, 0], axis=1)
                oo.mnStds[it] = _N.mean(stds, axis=0)
                print "mnStd  %.3f" % oo.mnStds[it]

                ttt6 = _tm.time()
                if not oo.bFixF:   
                    ARcfSmpl(ooN+1-oo.ignr, ook, oo.AR2lims, oo.smpx[:, 1+oo.ignr:, 0:ook], oo.smpx[:, oo.ignr:, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR, prior=oo.use_prior, accepts=8, aro=oo.ARord, sig_ph0L=oo.sig_ph0L, sig_ph0H=oo.sig_ph0H)  
                    oo.F_alfa_rep = alpR + alpC   #  new constructed
                    prt, rank, f, amp = ampAngRep(oo.F_alfa_rep, f_order=True)
                    oo.amps[it, :]  = amp
                    oo.fs[it, :]    = f
                    
                oo.allalfas[it] = oo.F_alfa_rep


                oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]
                for tr in xrange(oo.TR):
                    oo.Fs[tr, 0]    = oo.F0[:]

                #  sample u     WE USED TO Do this after smpx
                #  u(it+1)    using ws(it+1), F0(it), smpx(it+1), ws(it+1)

                oo.a2 = 0.5*(ooTR*(ooN-oo.ignr) + 2)  #  N + 1 - 1
                BB2 = 0
                for m in xrange(ooTR):
                    #   set x00 
                    oo.x00[m]      = oo.smpx[m, 2]*0.1

                    #####################    sample q2
                    rsd_stp = oo.smpx[m, 3+oo.ignr:,0] - _N.dot(oo.smpx[m, 2+oo.ignr:-1], oo.F0).T
                    #oo.rsds[it, m] = _N.dot(rsd_stp, rsd_stp.T)
                    BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
                oo.q2[:] = _ss.invgamma.rvs(oo.a2, scale=BB2)

                oo.smp_q2[:, it]= oo.q2


                #  sample obs noise
                oo.a2 = 0.5*(ooTR*(ooN-oo.ignr) + 2)  #  N + 1 - 1
                BB2 = 0
                for m in xrange(ooTR):
                    #   set x00 
                    #####################    sample q2
                    rsd_stp = oo.smpx[m, 2+oo.ignr:,0] - oo.y[m]
                    BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
                oo.oq2[:] = _ss.invgamma.rvs(oo.a2, scale=BB2)
                ws[:, :] = oo.oq2[0]

                oo.smp_oq2[:, it]= oo.oq2
                

                ttt7 = _tm.time()


        oo.dump_smps(0, toiter=(it+1), dir=oo.mcmcRunDir)


    def dump(self, dir=None):
        oo    = self
        pcklme = [oo]
        #oo.Bsmpx = None
        oo.smpx  = None
        oo.wts   = None
        oo.uts   = None
        oo.lfc   = None
        oo.rts   = None
        oo.zts   = None

        if dir is None:
            dmp = open("oo.dump", "wb")
        else:
            dmp = open("%s/oo.dump" % dir, "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()


    def run(self, runDir=None, trials=None): ###########  RUN
        oo     = self    #  call self oo.  takes up less room on line

        oo.setname = os.getcwd().split("/")[-1]

        oo.Cs          = len(oo.freq_lims)
        oo.C           = oo.Cn + oo.Cs
        oo.R           = 1
        oo.k           = 2*oo.C + oo.R
        #  x0  --  Gaussian prior
        oo.u_x00        = _N.zeros(oo.k)
        oo.s2_x00       = _arl.dcyCovMat(oo.k, _N.ones(oo.k), 0.4)
        oo.restarts = 0

        oo.loadDat(trials)
        oo.setParams()

        print "readSmpls   "

        t1    = _tm.time()

        oo.gibbsSamp()
        t2    = _tm.time()
        print (t2-t1)
        print "done"

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
        for tr in xrange(oo.TR):
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

        if oo.F_alfa_rep is None:
            oo.F_alfa_rep  = initF(oo.R, oo.Cs, oo.Cn, ifs=oo.ifs).tolist()   #  init F_alfa_rep

        if oo.q20 is None:
            oo.q20         = 0.00077
        oo.q2          = _N.ones(oo.TR)*oo.q20   #
        oo.oq2          = _N.ones(oo.TR)*oo.q20   #          

        oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]
        oo.Fs    = _N.zeros((oo.TR, oo.k, oo.k))

        oo.F[0] = oo.F0
        _N.fill_diagonal(oo.F[1:, 0:oo.k-1], 1)

        for tr in xrange(oo.TR):
            oo.Fs[tr] = oo.F


        ########  Limit the amplitude to something reasonable
        for m in xrange(oo.TR):
            oo.smpx[m, 2:, 0] = oo.y[m]

            for n in xrange(2+oo.k-1, oo.N+1+2):  # CREATE square smpx
                oo.smpx[m, n, 1:] = oo.smpx[m, n-oo.k+1:n, 0][::-1]
            for n in xrange(2+oo.k-2, -1, -1):  # CREATE square smpx
                oo.smpx[m, n, 0:oo.k-1] = oo.smpx[m, n+1, 1:oo.k]
                oo.smpx[m, n, oo.k-1] = _N.dot(oo.F0, oo.smpx[m, n:n+oo.k, oo.k-2]) # no noise

    def dump_smps(self, frm, pcklme=None, dir=None, toiter=None):
        oo    = self
        if pcklme is None:
            pcklme = {}

        toiter         = oo.NMC + oo.burn if (toiter is None) else toiter
        pcklme["frm"]    = frm
        pcklme["q2"]   = oo.smp_q2[:, 0:toiter]
        pcklme["amps"] = oo.amps[0:toiter]
        pcklme["fs"]   = oo.fs[0:toiter]
        pcklme["mnStds"]= oo.mnStds[0:toiter]
        pcklme["allalfas"]= oo.allalfas[0:toiter]
        pcklme["smpx"] = oo.smpx
        if oo.doBsmpx:
            pcklme["Bsmpx"]    = oo.Bsmpx[:, 0:toiter/oo.BsmpxSkp]

        print "saving state in %s" % oo.outSmplFN
        if dir is None:
            dmp = open(oo.outSmplFN, "wb")
        else:
            dmp = open("%(d)s/%(sfn)s" % {"d" : dir, "sfn" : oo.outSmplFN}, "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()
