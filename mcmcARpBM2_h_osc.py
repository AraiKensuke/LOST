"""
timing
1  0.00005
2  0.00733
2a 0.08150
2b 0.03315
3  0.11470
FFBS  1.02443
5  0.03322
"""
import pickle
from kflib import createDataAR
import numpy as _N
import patsy
import re as _re
import matplotlib.pyplot as _plt

import scipy.stats as _ss
from kassdirs import resFN, datFN

import utilities as _U

import numpy.polynomial.polynomial as _Npp
import time as _tm
import ARlib as _arl
import kfARlibMPmv as _kfar
import pyPG as lw
from ARcfSmpl import ARcfSmpl, FilteredTimeseries

import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import os

import mcmcARspk as mcmcARspk

from multiprocessing import Pool

class mcmcARpBM2(mcmcARspk.mcmcARspk):
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
    loghist       = None
    doS           = True

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
        oo.m         = _N.array([0, 1.])

        oo.alp       = _N.array([1., 1.])

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

        oo.loghist = _N.zeros(oo.N+1)

        ARo     = _N.empty((ooTR, oo._d.N+1))
        
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

        ###############################  MCMC LOOP  ########################
        ###  need pointer to oo.us, but reshaped for broadcasting to work
        ###############################  MCMC LOOP  ########################
        oous_rs = oo.us.reshape((ooTR, 1))   #  done for broadcasting rules

        sd01   = _N.zeros((oo.nStates, oo.TR, oo.TR))
        _N.fill_diagonal(sd01[0], oo.s[0])
        _N.fill_diagonal(sd01[1], oo.s[1])

        smpx01 = _N.zeros((oo.nStates, oo.TR, oo.N+1))
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

        alpR   = oo.F_alfa_rep[0:oo.R]
        alpC   = oo.F_alfa_rep[oo.R:]

        oo.nSMP_smpxC = 0
        if oo.processes > 1:
            print oo.processes
            pool = Pool(processes=oo.processes)
        print "oo.mcmcRunDir    %s" % oo.mcmcRunDir
        if oo.mcmcRunDir is None:
            oo.mcmcRunDir = ""
        elif (len(oo.mcmcRunDir) > 0) and (oo.mcmcRunDir[-1] != "/"):
            oo.mcmcRunDir += "/"


        #  H shape    100 x 9
        Hbf = oo.Hbf

        RHS = _N.empty((oo.histknots, 1))

        if oo.h0_1 > 1:   #  first few are 0s   
            #cInds = _N.array([0, 1, 5, 6, 7, 8, 9, 10])
            cInds = _N.array([0, 4, 5, 6, 7, 8, 9])
            #vInds = _N.array([2, 3, 4])
            vInds = _N.array([1, 2, 3,])
            RHS[cInds, 0] = 0
            RHS[0, 0] = -5
        else:
            #cInds = _N.array([5, 6, 7, 8, 9, 10])
            cInds = _N.array([4, 5, 6, 7, 8, 9,])
            vInds = _N.array([0, 1, 2, 3, ])
            #vInds = _N.array([0, 1, 2, 3, 4])
            RHS[cInds, 0] = 0

        Msts = []
        for m in xrange(ooTR):
            Msts.append(_N.where(oo.y[m] == 1)[0])
        HcM  = _N.empty((len(vInds), len(vInds)))

        HbfExpd = _N.empty((oo.histknots, ooTR, oo.N+1))
        #  HbfExpd is 11 x M x 1200
        #  find the mean.  For the HISTORY TERM
        for i in xrange(oo.histknots):
            for m in xrange(oo.TR):
                sts = Msts[m]
                HbfExpd[i, m, 0:sts[0]] = 0
                for iss in xrange(len(sts)-1):
                    t0  = sts[iss]
                    t1  = sts[iss+1]
                    HbfExpd[i, m, t0+1:t1+1] = Hbf[0:t1-t0, i]
                HbfExpd[i, m, sts[-1]+1:] = 0

        _N.dot(oo.B.T, oo.aS, out=BaS)
        if oo.hS is None:
            oo.hS = _N.zeros(oo.histknots)

        _N.dot(Hbf, oo.hS, out=oo.loghist)
        oo.stitch_Hist(ARo, oo.loghist, Msts)

        ##  ORDER OF SAMPLING
        ##  f_xx, f_V
        ##  BINARY state
        ##  DA:  PG, kpOws
        ##  history, build ARo
        ##  psth
        ##  offset
        ##  DA:  latent state
        ##  AR coefficients
        ##  q2

        while (it < ooNMC + oo.burn - 1):
            lowsts = _N.where(oo.Z[:, 0] == 1)
            #print "lowsts   %s" % str(lowsts)
            t1 = _tm.time()
            it += 1
            print "****------------  %d" % it
            oo._d.f_x[:, 0, :, 0]     = oo.x00
            if it == 0:
                for m in xrange(ooTR):
                    oo._d.f_V[m, 0]     = oo.s2_x00
            else:
                oo._d.f_V[:, 0]     = _N.mean(oo._d.f_V[:, 1:], axis=1)


            #  generate latent AR state

            if it > oo.startZ:
                for tryZ in xrange(oo.nStates):
                    _N.dot(sd01[tryZ], oo.smpx[..., 2:, 0], out=smpx01[tryZ])

                for m in oo.varz:
                    for tryZ in xrange(oo.nStates):  #  only allow certain trials to change

                        #  calculate p0, p1  p0 = m_0 x PROD_n Ber(y_n | Z_j)
                        #                       = m_0 x _N.exp(_N.log(  ))
                        #  p0, p1 not normalized
                        ll[tryZ] = 0
                        #  Ber(0 | ) and Ber(1 | )
                        _N.exp(smpx01[tryZ, m] + BaS + ARo[m] + oo.us[m] + oo.knownSig[m], out=expT)
                        Bp[0]   = 1 / (1 + expT)
                        Bp[1]   = expT / (1 + expT)

                        #   z[:, 1]   is state label

                        for n in xrange(oo.N+1):
                            ll[tryZ] += _N.log(Bp[oo.y[m, n], n])

                    ofs = _N.min(ll)
                    ll  -= ofs
                    nc = oo.m[0]*_N.exp(ll[0]) + oo.m[1]*_N.exp(ll[1])

                    oo.Z[m, 0] = 0;  oo.Z[m, 1] = 1
                    THR[m] = (oo.m[0]*_N.exp(ll[0]) / nc)
                    if _N.random.rand() < THR[m]:
                        oo.Z[m, 0] = 1;  oo.Z[m, 1] = 0
                    oo.smp_zs[m, it] = oo.Z[m]

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
            else:
                _N.fill_diagonal(zd, oo.s[oo.Z[:, 1]])
                _N.fill_diagonal(izd, 1./oo.s[oo.Z[:, 1]])

                _N.dot(zd, oo.smpx[..., 2:, 0], out=zsmpx)
                ######  sample m's
                oo.smp_ms[it] = oo.m
                oo.smp_zs[:, it, 1] = 1
                oo.smp_zs[:, it, 0] = 0
            print oo.m

            t3 = _tm.time()

            ######  PG generate
            for m in xrange(ooTR):
                lw.rpg_devroye(oo.rn, zsmpx[m] + oo.us[m] + BaS + ARo[m] + oo.knownSig[m], out=oo.ws[m])  ######  devryoe  ####TRD change
            _N.divide(oo.kp, oo.ws, out=kpOws)


            if not oo.bFixH:
                O = kpOws - zsmpx - oo.us.reshape((ooTR, 1)) - BaS -  oo.knownSig

                iOf = vInds[0]   #  offset HcM index with RHS index.
                for i in vInds:
                    for j in vInds:
                        HcM[i-iOf, j-iOf] = _N.sum(oo.ws*HbfExpd[i]*HbfExpd[j])

                    RHS[i, 0] = _N.sum(oo.ws*HbfExpd[i]*O)
                    for cj in cInds:
                        RHS[i, 0] -= _N.sum(oo.ws*HbfExpd[i]*HbfExpd[cj])*RHS[cj, 0]

                # print HbfExpd
                # print HcM
                # print RHS[vInds]
                vm = _N.linalg.solve(HcM, RHS[vInds])
                Cov = _N.linalg.inv(HcM)
                print vm
                cfs = _N.random.multivariate_normal(vm[:, 0], Cov, size=1)

                RHS[vInds,0] = cfs[0]
                oo.smp_hS[:, it] = RHS[:, 0]

                #RHS[2:6, 0] = vm[:, 0]
                #print HcM
                #vv = _N.dot(Hbf, RHS)
                #print vv.shape
                #print oo.loghist.shape
                _N.dot(Hbf, RHS[:, 0], out=oo.loghist)
                oo.smp_hist[:, it] = oo.loghist
                oo.stitch_Hist(ARo, oo.loghist, Msts)

            ########     PSTH sample  Do PSTH after we generate zs
            if oo.bpsth:
                Oms  = kpOws - zsmpx - ARo - oous_rs - oo.knownSig
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

            ########     per trial offset sample
            Ons  = kpOws - zsmpx - ARo - BaS - oo.knownSig

            #  solve for the mean of the distribution
            H    = _N.ones((oo.TR-1, oo.TR-1)) * _N.sum(oo.ws[0])
            uRHS = _N.empty(oo.TR-1)
            for dd in xrange(1, oo.TR):
                H[dd-1, dd-1] += _N.sum(oo.ws[dd])
                uRHS[dd-1] = _N.sum(oo.ws[dd]*Ons[dd] - oo.ws[0]*Ons[0])

            MM  = _N.linalg.solve(H, uRHS)
            Cov = _N.linalg.inv(H)

            oo.us[1:] = _N.random.multivariate_normal(MM, Cov, size=1)
            oo.us[0]  = -_N.sum(oo.us[1:])
            oo.smp_u[:, it] = oo.us

            t4 = _tm.time()
            ####  Sample latent state
            oo._d.y = _N.dot(izd, kpOws - BaS - ARo - oous_rs - oo.knownSig)
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
            if not oo.bFixF:   
                ARcfSmpl(oo.lfc, ooN+1, ook, oo.AR2lims, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR, prior=oo.use_prior, accepts=30, aro=oo.ARord, sig_ph0L=oo.sig_ph0L, sig_ph0H=oo.sig_ph0H)  
                oo.F_alfa_rep = alpR + alpC   #  new constructed
                prt, rank, f, amp = ampAngRep(oo.F_alfa_rep, f_order=True)
                print prt
            #ut, wt = FilteredTimeseries(ooN+1, ook, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR)
            #ranks[it]    = rank
            oo.allalfas[it] = oo.F_alfa_rep

            for m in xrange(ooTR):
                #oo.wts[m, it, :, :]   = wt[m, :, :, 0]
                #oo.uts[m, it, :, :]   = ut[m, :, :, 0]
                if not oo.bFixF:
                    oo.amps[it, :]  = amp
                    oo.fs[it, :]    = f

            oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]

            lwsts = _N.where(oo.Z[:, 0] == 1)[0]
            hists = _N.where(oo.Z[:, 1] == 1)[0]

            sts2chg = hists
            if (it > oo.startZ) and oo.doS:
                AL = 0.5*_N.sum(oo.smpx[sts2chg, 2:, 0]*oo.smpx[sts2chg, 2:, 0]*oo.ws[sts2chg])
                BRL = kpOws[sts2chg] - BaS - oous_rs[sts2chg] - ARo[sts2chg] - oo.knownSig[sts2chg]
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
            if (it > 1) and (it % oo.peek == 0):
                fig = _plt.figure(figsize=(12, 8))
                fig.add_subplot(4, 1, 1)
                _plt.plot(oo.amps[1:it, 0])
                fig.add_subplot(4, 1, 2)
                _plt.plot(oo.fs[1:it, 0])
                fig.add_subplot(4, 1, 3)
                _plt.plot(oo.mnStds[1:it])
                fig.add_subplot(4, 1, 4)
                _plt.plot(oo.smp_ms[1:it, 0])

                _plt.savefig("%(dir)s/tmp-fsamps%(it)d" % {"dir" : oo.mcmcRunDir, "it" : it})
                _plt.close()

                oo.dump_smpsS(toiter=it, dir=oo.mcmcRunDir)
        oo.dump_smpsS(dir=oo.mcmcRunDir)

    def runDirAlloc(self, runDir=None, trials=None, minSpkCnt=0, pckl=None, runlatent=False, dontrun=False): ###########  RUN
        oo     = self    #  call self oo.  takes up less room on line
        if oo.processes > 1:
            os.system("taskset -p 0xffffffff %d" % os.getpid())
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
        if pckl is not None:
            oo.restarts = 1
            oo.F0 = _N.zeros(oo.k)
            if type(pckl) == list:   #  format for posterior mode
                oo.bFixF  = True
                #oo.bFixH  = True
                oo.us = pckl[0]
                oo.B  = pckl[4]
                oo.hS = pckl[5]
                oo.Hbf = pckl[6]
                oo.q2 = _N.ones(oo.TR)*pckl[1]

                # for l in xrange(len(pckl[2])):
                #     oo.F0 += (-1*_Npp.polyfromroots(pckl[2][l])[::-1].real)[1:]
                # oo.F0 /= len(pckl[2]) # mean
                oo.F_alfa_rep  = _N.mean(pckl[2], axis=0)
                print oo.F_alfa_rep
                oo.aS = pckl[3]
            else:   #  format for last
                pckldITERS = pckl["allalfas"].shape[0]
                print "found %d Gibbs iterations samples" % pckldITERS
                oo.pkldalfas  = pckl["allalfas"][pckldITERS-1]
                oo.aS     = pckl["aS"][pckldITERS-1]
                oo.B      = pckl["B"]
                oo.hS     = pckl["h_coeffs"][:, pckldITERS-1]
                oo.q2     = pckl["q2"][:, pckldITERS-1]
                oo.us     = pckl["u"][:, pckldITERS-1]
                oo.smpx   = pckl["smpx"]
                if pckl.has_key("ws"):
                    oo.ws     = pckl["ws"]
                if pckl.has_key("m"):
                    oo.m      = pckl["m"]
                    oo.Z      = pckl["Z"]
                oo.Hbf    = oo.Hbf
                oo.F0     = (-1*_Npp.polyfromroots(oo.pkldalfas)[::-1].real)[1:]
                oo.F_alfa_rep  = _N.mean(oo.pkldalfas, axis=0)
        if not dontrun:
            t1 = _tm.time()
            oo.dirichletAllocate()
            t2    = _tm.time()
            print (t2-t1)

    def dump_smpsS(self, dir=None, toiter=None):
        oo         = self
        pcklme     = {}
        
        toiter         = oo.NMC + oo.burn if (toiter is None) else toiter
        pcklme["ss"]  = oo.smp_ss[0:toiter]
        pcklme["zs"]  = oo.smp_zs[:, 0:toiter]
        pcklme["ms"]  = oo.smp_ms[0:toiter]
        #oo.__class__.__bases__[0].dump_smps(pcklme=pcklme, dir=dir, toiter=toiter)
        oo.dump_smps(pcklme=pcklme, dir=dir, toiter=toiter)
