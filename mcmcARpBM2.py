from kflib import createDataAR, disjointSubset
#import matplotlib.pyplot as _plt
import numpy as _N
import re as _re
from filter import bpFilt, lpFilt, gauKer

import scipy.stats as _ss
from kassdirs import resFN, datFN

from   mcmcARpPlot import plotFigs, plotARcomps, plotQ2
from mcmcARpFuncs import loadL2, runNotes
import kfardat as _kfardat

import utilities as _U

import numpy.polynomial.polynomial as _Npp
import time as _tm
import ARlib as _arl
import kfARlibMPmv as _kfar
import pyPG as lw
from ARcfSmpl import ARcfSmpl, FilteredTimeseries

import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
from multiprocessing import Pool

import os
import mcmcARspk as mARspk

from multiprocessing import Pool

import matplotlib.pyplot as _plt

class mcmcARpBM2(mARspk.mcmcARspk):
    ###########  TEMP
    THR           = None
    startZ        = 0

    #  binomial states
    nStates       = 2
    s             = None   #  coupling M x 2
    Z             = None   #  state index M x 2  [(1, 0), (0, 1), (0, 1), ...]
    m             = None   #  dim 2
    sd            = None
    smp_ss        = None

    #  Do not vary these Z's
    fxdz          = None
    varz          = None

    
    #  Dirichlet priors
    alp           = None

    #  initial value
    lowStates     = None   #  

    def allocateSmp(self, iters):   ################################ INITGIBBS
        oo   = self

        if oo.processes > 1:
            os.system("taskset -p 0xff %d" % os.getpid())

        oo.__class__.__bases__[0].allocateSmp(oo, iters)

        oo.smp_zs    = _N.zeros((oo.TR, oo.burn + oo.NMC, oo.nStates))
        oo.smp_ms    = _N.zeros((oo.burn + oo.NMC, oo.nStates))

        if oo.Z is None:   #  not set externally
            oo.Z         = _N.zeros((oo.TR, oo.nStates), dtype=_N.int)
            if oo.lowStates is not None:
                for tr in xrange(oo.TR):
                    oo.Z[tr, 0] = 0;                oo.Z[tr, 1] = 1
                    try:
                        oo.lowStates.index(tr)
                        oo.Z[tr, 0] = 1;                oo.Z[tr, 1] = 0
                    except ValueError:
                        pass
            else:
                for tr in xrange(oo.TR):
                    oo.Z[tr, 1] = 1;                oo.Z[tr, 0] = 0

        oo.s         = _N.array([0.1, 1])
        oo.smp_ss       = _N.zeros(oo.burn + oo.NMC)
        oo.sd        = _N.zeros((oo.TR, oo.TR))
        oo.m         = _N.array([0.1, 0.9])

        oo.alp       = _N.array([1, 1])

        if (oo.varz is None) and (oo.fxdz is None):
            ##  
            oo.varz  = _N.arange(oo.TR)
            oo.fxdz  = _N.array([])
        elif (oo.varz is None) and (oo.fxdz is not None):
            if type(oo.fxdz) is list:
               oo.fxdz = _N.array(oo.fxdz)
            oo.varz  = _N.array(disjointSubset(range(oo.TR), oo.fxdz), dtype=_N.int)
        elif (oo.varz is not None) and (oo.fxdz is None):
            if type(oo.varz) is list:
                oo.varz = _N.array(oo.varz)
            oo.fxdz  = _N.array(disjointSubset(range(oo.TR), oo.varz), dtype=_N.int)

    def dirichletAllocate(self):  ###########################  GIBBSSAMP
        oo          = self
        ooTR        = oo.TR
        print ooTR
        ook         = oo.k
        ooNMC       = oo.NMC
        ooN         = oo.N

        oo.allocateSmp(oo.burn + oo.NMC)
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))

        ARo     = _N.empty((ooTR, oo._d.N+1))
        ARo01   = _N.empty((oo.nStates, ooTR, oo._d.N+1))
        
        kpOws = _N.empty((ooTR, ooN+1))
        lv_f     = _N.zeros((ooN+1, ooN+1))
        lv_u     = _N.zeros((ooTR, ooTR))
        Bii    = _N.zeros((ooN+1, ooN+1))
        
        #alpC.reverse()
        #  F_alfa_rep = alpR + alpC  already in right order, no?

        Wims         = _N.empty((ooTR, ooN+1, ooN+1))
        Oms          = _N.empty((ooTR, ooN+1))
        smWimOm      = _N.zeros(ooN + 1)
        smWinOn      = _N.zeros(ooTR)
        bConstPSTH = False
        D_f          = _N.diag(_N.ones(oo.B.shape[0])*oo.s2_a)   #  spline
        iD_f = _N.linalg.inv(D_f)
        D_u  = _N.diag(_N.ones(oo.TR)*oo.s2_u)   #  This should 
        iD_u = _N.linalg.inv(D_u)
        iD_u_u_u = _N.dot(iD_u, _N.ones(oo.TR)*oo.u_u)
        BDB  = _N.dot(oo.B.T, _N.dot(D_f, oo.B))
        DB   = _N.dot(D_f, oo.B)
        BTua = _N.dot(oo.B.T, oo.u_a)

        it    = 0

        oo.lrn   = _N.empty((ooTR, ooN+1))
        oo.s_lrn   = _N.empty((ooTR, ooN+1))
        oo.sprb   = _N.empty((ooTR, ooN+1))
        oo.lrn_scr1   = _N.empty(ooN+1)
        oo.lrn_iscr1   = _N.empty(ooN+1)
        oo.lrn_scr2   = _N.empty(ooN+1)
        oo.lrn_scr3   = _N.empty(ooN+1)
        oo.lrn_scld   = _N.empty(ooN+1)

        oo.lrn   = _N.empty((ooTR, ooN+1))
        if oo.l2 is None:
            oo.lrn[:] = 1
        else:
            for tr in xrange(ooTR):
                oo.lrn[tr] = oo.build_lrnLambda2(tr)

        ###############################  MCMC LOOP  ########################
        ###  need pointer to oo.us, but reshaped for broadcasting to work
        ###############################  MCMC LOOP  ########################
        oous_rs = oo.us.reshape((ooTR, 1))   #  done for broadcasting rules
        lrnBadLoc = _N.empty((oo.TR, oo.N+1), dtype=_N.bool)

        sd01   = _N.zeros((oo.nStates, oo.TR, oo.TR))
        _N.fill_diagonal(sd01[0], oo.s[0])
        _N.fill_diagonal(sd01[1], oo.s[1])

        smpx01 = _N.zeros((oo.nStates, oo.TR, oo.N+1))
        ARo01  = _N.empty((oo.nStates, oo.TR, oo.N+1))
        zsmpx  = _N.empty((oo.TR, oo.N+1))

        #  zsmpx created
        #  PG

        zd     = _N.zeros((oo.TR, oo.TR))
        izd    = _N.zeros((oo.TR, oo.TR))
        ll    = _N.zeros(oo.nStates)
        Bp    = _N.empty((oo.nStates, oo.N+1))

        for m in xrange(ooTR):
            oo._d.f_V[m, 0]     = oo.s2_x00
            oo._d.f_V[m, 1]     = oo.s2_x00

        THR = _N.empty(oo.TR)
        dirArgs = _N.empty(oo.nStates)  #  dirichlet distribution args
        expT= _N.empty(ooN+1)
        BaS = _N.dot(oo.B.T, oo.aS)

        oo.nSMP_smpxC = 0
        if oo.processes > 1:
            print oo.processes
            pool = Pool(processes=oo.processes)

        while (it < ooNMC + oo.burn - 1):
            lowsts = _N.where(oo.Z[:, 0] == 1)
            #print "lowsts   %s" % str(lowsts)
            t1 = _tm.time()
            it += 1
            print "****------------  %d" % it

            #  generate latent AR state

            ######  Z
            #print "!!!!!!!!!!!!!!!  1"

            for tryZ in xrange(oo.nStates):
                _N.dot(sd01[tryZ], oo.smpx[..., 2:, 0], out=smpx01[tryZ])
                #oo.build_addHistory(ARo01[tryZ], smpx01[tryZ, m], BaS, oo.us, lrnBadLoc)
                oo.build_addHistory(ARo01[tryZ], smpx01[tryZ], BaS, oo.us, oo.knownSig)
                """
                for m in xrange(oo.TR):
                    locs = _N.where(lrnBadLoc[m] == True)
                    if locs[0].shape[0] > 0:
                        print "found a bad loc"
                        fig = _plt.figure(figsize=(8, 5))
                        _plt.suptitle("%d" % m)
                        _plt.subplot(2, 1, 1)
                        _plt.plot(smpx01[tryZ, m])
                        _plt.subplot(2, 1, 2)
                        _plt.plot(ARo01[tryZ, m])
                """
                #print "!!!!!!!!!!!!!!!  2"
            for m in oo.varz:
                for tryZ in xrange(oo.nStates):  #  only allow certain trials to change

                    #  calculate p0, p1  p0 = m_0 x PROD_n Ber(y_n | Z_j)
                    #                       = m_0 x _N.exp(_N.log(  ))
                    #  p0, p1 not normalized
                    ll[tryZ] = 0
                    #  Ber(0 | ) and Ber(1 | )
                    _N.exp(smpx01[tryZ, m] + BaS + ARo01[tryZ, m] + oo.us[m], out=expT)
                    Bp[0]   = 1 / (1 + expT)
                    Bp[1]   = expT / (1 + expT)

                    #   z[:, 1]   is state label

                    for n in xrange(oo.N+1):
                        ll[tryZ] += _N.log(Bp[oo.y[m, n], n])

                ofs = _N.min(ll)
                ll  -= ofs
                nc = oo.m[0]*_N.exp(ll[0]) + oo.m[1]*_N.exp(ll[1])

                iARo = 1
                oo.Z[m, 0] = 0;  oo.Z[m, 1] = 1
                THR[m] = (oo.m[0]*_N.exp(ll[0]) / nc)
                if _N.random.rand() < THR[m]:
                    oo.Z[m, 0] = 1;  oo.Z[m, 1] = 0
                    iARo = 0
                oo.smp_zs[m, it] = oo.Z[m]
                ####  did we forget to do this?
                ARo[m] = ARo01[iARo, m]
            for m in oo.fxdz: #####  outside BM loop
                oo.smp_zs[m, it] = oo.Z[m]
            t2 = _tm.time()

            #  Z  set
            _N.fill_diagonal(zd, oo.s[oo.Z[:, 1]])
            _N.fill_diagonal(izd, 1./oo.s[oo.Z[:, 1]])
            #for kkk in xrange(oo.TR):
            #    print zd[kkk, kkk]
            _N.dot(zd, oo.smpx[..., 2:, 0], out=zsmpx)
            ######  sample m's
            _N.add(oo.alp, _N.sum(oo.Z[oo.varz], axis=0), out=dirArgs)
            oo.m[:] = _N.random.dirichlet(dirArgs)
            oo.smp_ms[it] = oo.m
            print oo.m

            oo.build_addHistory(ARo, zsmpx, BaS, oo.us, oo.knownSig)
            t3 = _tm.time()

            ######  PG generate
            nanLoc = _N.where(_N.isnan(BaS))

            for m in xrange(ooTR):
                lw.rpg_devroye(oo.rn, zsmpx[m] + oo.us[m] + BaS + ARo[m], out=oo.ws[m])  ######  devryoe  ####TRD change
                nanLoc = _N.where(_N.isnan(oo.ws[m]))
                if len(nanLoc[0]) > 0:
                    loc = nanLoc[0][0]
            _N.divide(oo.kp, oo.ws, out=kpOws)

            ########     per trial offset sample
            Ons  = kpOws - zsmpx - ARo - BaS
            _N.einsum("mn,mn->m", oo.ws, Ons, out=smWinOn)  #  sum over trials
            ilv_u  = _N.diag(_N.sum(oo.ws, axis=1))  #  var  of LL
            _N.fill_diagonal(lv_u, 1./_N.diagonal(ilv_u))
            lm_u  = _N.dot(lv_u, smWinOn)  #  nondiag of 1./Bi are inf, mean LL
            #  now sample
            iVAR = ilv_u + iD_u
            VAR  = _N.linalg.inv(iVAR)  #
            Mn    = _N.dot(VAR, _N.dot(ilv_u, lm_u) + iD_u_u_u)
            oo.us[:]  = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]

            oo.smp_u[:, it] = oo.us

            ########     PSTH sample  Do PSTH after we generate zs
            if oo.bpsth:
                Oms  = kpOws - zsmpx - ARo - oous_rs
                _N.einsum("mn,mn->n", oo.ws, Oms, out=smWimOm)   #  sum over 
                ilv_f  = _N.diag(_N.sum(oo.ws, axis=0))
                _N.fill_diagonal(lv_f, 1./_N.diagonal(ilv_f))
                lm_f  = _N.dot(lv_f, smWimOm)  #  nondiag of 1./Bi are inf
                #  now sample
                iVAR = _N.dot(oo.B, _N.dot(ilv_f, oo.B.T)) + iD_f
                VAR  = _N.linalg.inv(iVAR)  #  knots x knots
                #iBDBW = _N.linalg.inv(BDB + lv_f)   # BDB not diag
                #Mn    = oo.u_a + _N.dot(DB, _N.dot(iBDBW, lm_f - BTua))

                Mn = oo.u_a + _N.dot(DB, _N.linalg.solve(BDB + lv_f, lm_f - BTua))
                oo.aS   = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                oo.smp_aS[it, :] = oo.aS

                #iBDBW = _N.linalg.inv(BDB + lv_f)   # BDB not diag
                #Mn    = oo.u_a + _N.dot(DB, _N.dot(iBDBW, lm_f - BTua))
                #oo.aS   = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                #oo.smp_aS[it, :] = oo.aS
            else:
                oo.aS[:]   = 0
            BaS = _N.dot(oo.B.T, oo.aS)

            t4 = _tm.time()
            ####  Sample latent state
            oo._d.y = _N.dot(izd, kpOws - BaS - ARo - oous_rs)
            oo._d.copyParams(oo.F0, oo.q2)
            #  (MxM)  (MxN) = (MxN)  (Rv is MxN)
            _N.dot(_N.dot(izd, izd), 1. / oo.ws, out=oo._d.Rv)

            oo._d.f_x[:, 0, :, 0]     = oo.x00
            #if it == 1:
            for m in xrange(ooTR):
                oo._d.f_V[m, 0]     = oo.s2_x00
            else:
                oo._d.f_V[:, 0]     = _N.mean(oo._d.f_V[:, 1:], axis=1)

            tpl_args = zip(oo._d.y, oo._d.Rv, oo._d.Fs, oo.q2, oo._d.Ns, oo._d.ks, oo._d.f_x[:, 0], oo._d.f_V[:, 0])

            t5 = _tm.time()
            if oo.processes == 1:
                for m in xrange(ooTR):
                    oo.smpx[m, 2:], oo._d.f_x[m], oo._d.f_V[m] = _kfar.armdl_FFBS_1itrMP(tpl_args[m])
                    oo.smpx[m, 1, 0:ook-1]   = oo.smpx[m, 2, 1:]
                    oo.smpx[m, 0, 0:ook-2]   = oo.smpx[m, 2, 2:]
                    oo.smp_q2[m, it]= oo.q2[m]

            else:
                sxv = pool.map(_kfar.armdl_FFBS_1itrMP, tpl_args)
                for m in xrange(ooTR):
                    oo.smpx[m, 2:] = sxv[m][0]
                    oo._d.f_x[m] = sxv[m][1]
                    oo._d.f_V[m] = sxv[m][2]
                    oo.smpx[m, 1, 0:ook-1]   = oo.smpx[m, 2, 1:]
                    oo.smpx[m, 0, 0:ook-2]   = oo.smpx[m, 2, 2:]
                    #oo.Bsmpx[m, it, 2:]    = oo.smpx[m, 2:, 0]

            stds = _N.std(oo.smpx[:, 2:, 0], axis=1)
            oo.mnStds[it] = _N.mean(stds, axis=0)
            print "mnStd  %.3f" % oo.mnStds[it]
            ###  

            lwsts = _N.where(oo.Z[:, 0] == 1)[0]
            hists = _N.where(oo.Z[:, 1] == 1)[0]

            sts2chg = hists
            if (it > oo.startZ) and (len(sts2chg) > 0):
                AL = 0.5*_N.sum(oo.smpx[sts2chg, 2:, 0]*oo.smpx[sts2chg, 2:, 0]*oo.ws[sts2chg])
                BRL = kpOws[sts2chg] - BaS - oous_rs[sts2chg] - ARo[sts2chg] 
                BL = _N.sum(oo.ws[sts2chg]*BRL*oo.smpx[sts2chg, 2:, 0])
                UL = BL / (2*AL)
                sgL= 1/_N.sqrt(2*AL)
                U = UL
                sg= sgL

                print "U  %(U).3f    s  %(s).3f" % {"U" : U, "s" : sg}

                oo.s[1] = U + sg*_N.random.randn()

                _N.fill_diagonal(sd01[0], oo.s[0])
                _N.fill_diagonal(sd01[1], oo.s[1])
                print oo.s[1]
                oo.smp_ss[it] = oo.s[1] 

            oo.a2 = oo.a_q2 + 0.5*(ooTR*ooN + 2)  #  N + 1 - 1
            BB2 = oo.B_q2
            for m in xrange(ooTR):
                #   set x00 
                #oo.x00[m]      = oo.smpx[m, 2]*0.1
                oo.x00[m]      = oo.smpx[m, 2]*0.001

                #####################    sample q2
                rsd_stp = oo.smpx[m, 3:,0] - _N.dot(oo.smpx[m, 2:-1], oo.F0).T
                BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
            oo.q2[:] = _ss.invgamma.rvs(oo.a2, scale=BB2)

            oo.smp_q2[:, it]= oo.q2
            t7 = _tm.time()
            print "gibbs iter %.3f" % (t7-t1)

    def runDirAlloc(self, pckl, trials=None): ###########  RUN
        """
        """
        oo     = self    #  call self oo.  takes up less room on line
        oo.setname = os.getcwd().split("/")[-1]

        oo.Cs          = len(oo.freq_lims)
        oo.C           = oo.Cn + oo.Cs
        oo.R           = 1
        oo.k           = 2*oo.C + oo.R
        #  x0  --  Gaussian prior
        oo.u_x00        = _N.zeros(oo.k)
        oo.s2_x00       = _arl.dcyCovMat(oo.k, _N.ones(oo.k), 0.4)

        oo.loadDat(trials)
        oo.setParams()
        oo.us = pckl[0]
        oo.q2 = _N.ones(oo.TR)*pckl[1]
        oo.F0 = _N.zeros(oo.k)
        print len(pckl[2])
        for l in xrange(len(pckl[2])):
            oo.F0 += (-1*_Npp.polyfromroots(pckl[2][l])[::-1].real)[1:]
        oo.F0 /= len(pckl[2])
        oo.aS = pckl[3]
        
        oo.dirichletAllocate()

    def run(self, runDir=None, trials=None, minSpkCnt=0): ###########  RUN
        oo     = self    #  call self oo.  takes up less room on line
        if oo.processes > 1:
            os.system("taskset -p 0xff %d" % os.getpid())
        oo.setname = os.getcwd().split("/")[-1]

        oo.Cs          = len(oo.freq_lims)
        oo.C           = oo.Cn + oo.Cs
        oo.R           = 1
        oo.k           = 2*oo.C + oo.R
        #  x0  --  Gaussian prior
        oo.u_x00        = _N.zeros(oo.k)
        oo.s2_x00       = _arl.dcyCovMat(oo.k, _N.ones(oo.k), 0.4)

        oo.rs=-1
        if runDir == None:
            runDir="%(sn)s/AR%(k)d_[%(t0)d-%(t1)d]" % \
                {"sn" : oo.setname, "ar" : oo.k, "t0" : oo.t0, "t1" : oo.t1}

        if oo.rs >= 0:
            unpickle(runDir, oo.rs)
        else:   #  First run
            oo.restarts = 0

        oo.loadDat(trials)
        oo.setParams()
        t1    = _tm.time()
        tmpNOAR = oo.noAR
        oo.noAR = True
        oo.gibbsSamp()

        # oo.noAR = tmpNOAR
        # oo.gibbsSamp()
        # t2    = _tm.time()
        # print (t2-t1)

    def dumpSamples(self, dir=None):
        pcklme     = {}
        
        pcklme["ss"]  = oo.smp_ss
        pcklme["zs"]  = oo.smp_zs
        pcklme["ms"]  = oo.smp_ms
        oo.__class__.__bases__[0].dumpSamples(pcklme=pcklme, dir=dir)
