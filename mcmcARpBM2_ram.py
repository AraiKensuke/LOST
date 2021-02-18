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
from LOST.kflib import createDataAR
import numpy as _N
import patsy
import re as _re
import matplotlib.pyplot as _plt

import scipy.stats as _ss

import utilities as _U

import numpy.polynomial.polynomial as _Npp
import time as _tm
import LOST.ARlib as _arl

import LOST.kfARlibMPmv_ram2 as _kfar
import pyPG as lw
#from ARcfSmpl import ARcfSmpl, FilteredTimeseries
#import LOST.ARcfSmplNoMCMC as _arcfs
from ARcfSmplNoMCMC import ARcfSmpl

import commdefs as _cd

from LOST.ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import os
import signal

import LOST.mcmcARspk as mcmcARspk

from multiprocessing import Pool

interrupted = False
def signal_handler(signal, frame):
    global interrupted
    print("******  INTERRUPT")
    interrupted = True

class mcmcARpBM2(mcmcARspk.mcmcARspk):
    ###########  TEMP
    THR           = None
    startZ        = 2000

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

    def allocateSmp(self, iters, Bsmpx=False):   ################################ INITGIBBS
        oo   = self

        oo.__class__.__bases__[0].allocateSmp(oo, iters, Bsmpx=Bsmpx)

        oo.smp_zs    = _N.zeros((oo.TR, oo.ITERS, oo.nStates))
        oo.smp_ms    = _N.zeros((oo.ITERS, oo.nStates))

        if oo.Z is None:   #  not set externally
            oo.Z         = _N.zeros((oo.TR, oo.nStates), dtype=_N.int)
            if oo.lowStates is not None:
                for tr in range(oo.TR):
                    oo.Z[tr, 0] = 0;                oo.Z[tr, 1] = 1
                    try:
                        oo.lowStates.index(tr)
                        oo.Z[tr, 0] = 1;                oo.Z[tr, 1] = 0
                    except ValueError:
                        pass
            else:
                for tr in range(oo.TR):
                    oo.Z[tr, 1] = 1;                oo.Z[tr, 0] = 0


        oo.s         = _N.array([0.01, 1])
        oo.smp_ss       = _N.zeros(oo.ITERS)
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

        signal.signal(signal.SIGINT, signal_handler)

        ooTR        = oo.TR
        ook         = oo.k
        ooN         = oo.N

        runTO = oo.ITERS - 1
        oo.allocateSmp(runTO+1, Bsmpx=oo.doBsmpx)
        #oo.allocateSmp(oo.burn + oo.NMC)
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))

        _kfar.init(oo.N, oo.k, oo.TR)

        if oo.dohist:
            oo.loghist = _N.zeros(oo.Hbf.shape[0])
        else:
            print("fixed hist is")
            print(oo.loghist)

        ARo   = _N.zeros((ooTR, ooN+1))
        
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

        for m in range(ooTR):
            oo.f_V[m, 0]     = oo.s2_x00
            oo.f_V[m, 1]     = oo.s2_x00

        THR = _N.empty(oo.TR)
        dirArgs = _N.empty(oo.nStates)  #  dirichlet distribution args
        expT= _N.empty(ooN+1)
        BaS = _N.dot(oo.B.T, oo.aS)

        alpR   = oo.F_alfa_rep[0:oo.R]
        alpC   = oo.F_alfa_rep[oo.R:]

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(oo.F_alfa_rep)
        print("*****************************")
        print(alpR)
        print(alpC)


        oo.nSMP_smpxC = 0

        if oo.mcmcRunDir is None:
            oo.mcmcRunDir = ""
        elif (len(oo.mcmcRunDir) > 0) and (oo.mcmcRunDir[-1] != "/"):
            oo.mcmcRunDir += "/"


        #  H shape    100 x 9
        Hbf = oo.Hbf

        RHS = _N.empty((oo.histknots, 1))

        cInds = _N.arange(oo.iHistKnotBeginFixed, oo.histknots)
        vInds = _N.arange(0, oo.iHistKnotBeginFixed)
        RHS[cInds, 0] = 0

        Msts = []
        for m in range(ooTR):
            Msts.append(_N.where(oo.y[m] == 1)[0])
        HcM  = _N.empty((len(vInds), len(vInds)))

        HbfExpd = _N.empty((oo.histknots, ooTR, oo.N+1))
        #  HbfExpd is 11 x M x 1200
        #  find the mean.  For the HISTORY TERM
        for i in range(oo.histknots):
            for m in range(oo.TR):
                sts = Msts[m]
                HbfExpd[i, m, 0:sts[0]] = 0
                for iss in range(len(sts)-1):
                    t0  = sts[iss]
                    t1  = sts[iss+1]
                    HbfExpd[i, m, t0+1:t1+1] = Hbf[0:t1-t0, i]
                HbfExpd[i, m, sts[-1]+1:] = 0

        _N.dot(oo.B.T, oo.aS, out=BaS)
        if oo.hS is None:
            oo.hS = _N.zeros(oo.histknots)

        _N.dot(Hbf, oo.hS, out=oo.loghist)
        oo.stitch_Hist(ARo, oo.loghist, Msts)

        K     = _N.empty((oo.TR, oo.N + 1, oo.k))   #  kalman gain

        iterBLOCKS  = oo.ITERS//oo.peek
        smpx_tmp = _N.empty((oo.TR, oo.N+1, oo.k))

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
        oo.gau_var = _N.array(oo.ws)

        #iterBLOCKS = 1
        #oo.peek = 1

        arangeNp1 = _N.arange(oo.N+1)
        for itrB in range(iterBLOCKS):
            it = itrB*oo.peek
            if it > 0:
                print("it: %(it)d    mnStd  %(mnstd).3f   fs  %(fs).3f    m %(m).3f    [%(0).2f,%(1).2f]" % {"it" : itrB*oo.peek, "mnstd" : oo.mnStds[it-1], "fs" : oo.fs[it-1, 0], "m" : oo.m[0], "0" : oo.s[0], "1" : oo.s[1]})

            #tttA = _tm.time()
            if interrupted:
                break
            for it in range(itrB*oo.peek, (itrB+1)*oo.peek):

                lowsts = _N.where(oo.Z[:, 0] == 1)
                #print "lowsts   %s" % str(lowsts)
                t1 = _tm.time()
                oo.f_x[:, 0]     = oo.x00
                if it == 0:
                    for m in range(ooTR):
                        oo.f_V[m, 0]     = oo.s2_x00
                else:
                    oo.f_V[:, 0]     = _N.mean(oo.f_V[:, 1:], axis=1)


                #  generate latent AR state

                if it > oo.startZ:
                    for tryZ in range(oo.nStates):
                        _N.dot(sd01[tryZ], oo.smpx[:, 2:, 0], out=smpx01[tryZ])

                    for m in range(oo.TR):
                        for tryZ in range(oo.nStates):  #  only allow certain trials to change

                            #  calculate p0, p1  p0 = m_0 x PROD_n Ber(y_n | Z_j)
                            #                       = m_0 x _N.exp(_N.log(  ))
                            #  p0, p1 not normalized
                            #  Ber(0 | ) and Ber(1 | )
                            _N.exp(smpx01[tryZ, m] + BaS + ARo[m] + oo.us[m] + oo.knownSig[m], out=expT)
                            Bp[0]   = 1 / (1 + expT)
                            Bp[1]   = expT / (1 + expT)

                            #   z[:, 1]   is state label
                            #ll[tryZ] = 0
                            ll[tryZ] = _N.sum(_N.log(Bp[oo.y[m, arangeNp1], arangeNp1]))

                        ofs = _N.min(ll)
                        ll  -= ofs
                        #nc = oo.m[0]*_N.exp(ll[0]) + oo.m[1]*_N.exp(ll[1])
                        nc = oo.m[0] + oo.m[1]*_N.exp(ll[1] - ll[0])

                        oo.Z[m, 0] = 0;  oo.Z[m, 1] = 1
                        #THR[m] = (oo.m[0]*_N.exp(ll[0]) / nc)
                        THR[m] = (oo.m[0] / nc)
                        if _N.random.rand() < THR[m]:
                            oo.Z[m, 0] = 1;  oo.Z[m, 1] = 0
                        oo.smp_zs[m, it] = oo.Z[m]
                    for m in oo.fxdz: #####  outside BM loop
                        oo.smp_zs[m, it] = oo.Z[m]
                    #  Z  set
                    _N.fill_diagonal(zd, oo.s[oo.Z[:, 1]])
                    _N.fill_diagonal(izd, 1./oo.s[oo.Z[:, 1]])

                    _N.dot(zd, oo.smpx[..., 2:, 0], out=zsmpx)
                    ######  sample m's
                    _N.add(oo.alp, _N.sum(oo.Z[oo.varz], axis=0), out=dirArgs)
                    oo.m[:] = _N.random.dirichlet(dirArgs)
                    oo.smp_ms[it] = oo.m

                else:   #  turned off dirichlet, always allocate to low state
                    _N.fill_diagonal(zd, oo.s[oo.Z[:, 1]])
                    _N.fill_diagonal(izd, 1./oo.s[oo.Z[:, 1]])

                    _N.dot(zd, oo.smpx[:, 2:, 0], out=zsmpx)
                    ######  sample m's
                    oo.smp_ms[it] = oo.m
                    oo.smp_zs[:, it, 1] = 1
                    oo.smp_zs[:, it, 0] = 0

                lwsts = _N.where(oo.Z[:, 0] == 1)[0]
                hists = _N.where(oo.Z[:, 1] == 1)[0]

                #print(zsmpx[0, 0:20])
                #print(oo.smpx[0, 2:22, 0])
                t3 = _tm.time()

                ######  PG generate
                for m in range(ooTR):
                    ###  CHANGE 1
                    #lw.rpg_devroye(oo.rn, oo.smpx[m, 2:, 0] + oo.us[m] + BaS + ARo[m] + oo.knownSig[m], out=oo.ws[m])  ######  devryoe
                    lw.rpg_devroye(oo.rn, zsmpx[m] + oo.us[m] + BaS + ARo[m] + oo.knownSig[m], out=oo.ws[m])  ######  devryoe  ####TRD change

                _N.divide(oo.kp, oo.ws, out=kpOws)


                if oo.dohist:
                    #O = kpOws - oo.smpx[..., 2:, 0] - oo.us.reshape((ooTR, 1)) - BaS -  oo.knownSig
                    O = kpOws - zsmpx - oo.us.reshape((ooTR, 1)) - BaS -  oo.knownSig

                    for ii in range(len(vInds)):
                        #print("i   %d" % i)
                        #print(_N.sum(HbfExpd[i]))
                        i = vInds[ii]
                        for jj in range(ii, len(vInds)):
                            j = vInds[jj]
                            #print("j   %d" % j)
                            #print(_N.sum(HbfExpd[j]))
                            HcM[ii, jj] = _N.sum(oo.ws*HbfExpd[i]*HbfExpd[j])
                            HcM[jj, ii] = HcM[ii, jj]

                        RHS[ii, 0] = _N.sum(oo.ws*HbfExpd[i]*O)
                        for cj in cInds:
                            RHS[ii, 0] -= _N.sum(oo.ws*HbfExpd[i]*HbfExpd[cj])*RHS[cj, 0]
 
                    vm = _N.linalg.solve(HcM, RHS[vInds])
                    Cov = _N.linalg.inv(HcM)
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
                    #Oms  = kpOws - oo.smpx[..., 2:, 0] - ARo - oous_rs - oo.knownSig
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
                _N.dot(oo.B.T, oo.aS, out=BaS)

                ########     per trial offset sample
                #Ons  = kpOws - zsmpx - ARo - BaS - oo.knownSig
                Ons  = kpOws - oo.smpx[..., 2:, 0] - ARo - BaS - oo.knownSig

                #  solve for the mean of the distribution
                H    = _N.ones((oo.TR-1, oo.TR-1)) * _N.sum(oo.ws[0])
                uRHS = _N.empty(oo.TR-1)
                for dd in range(1, oo.TR):
                    H[dd-1, dd-1] += _N.sum(oo.ws[dd])
                    uRHS[dd-1] = _N.sum(oo.ws[dd]*Ons[dd] - oo.ws[0]*Ons[0])

                MM  = _N.linalg.solve(H, uRHS)
                Cov = _N.linalg.inv(H)

                oo.us[1:] = _N.random.multivariate_normal(MM, Cov, size=1)
                oo.us[0]  = -_N.sum(oo.us[1:])
                oo.smp_u[:, it] = oo.us

                t4 = _tm.time()
                ####  Sample latent state
                #oo.gau_obs = kpOws - BaS - ARo - oous_rs - oo.knownSig
                oo.gau_obs = _N.dot(izd, kpOws - BaS - ARo - oous_rs - oo.knownSig)
                #oo.copyParams(oo.F0, oo.q2)
                #  (MxM)  (MxN) = (MxN)  (Rv is MxN)
                _N.dot(_N.dot(izd, izd), 1. / oo.ws, out=oo.gau_var)
                #oo.gau_var =1 / oo.ws

                t5 = _tm.time()

                _kfar.armdl_FFBS_1itrMP(oo.gau_obs, oo.gau_var, oo.Fs, _N.linalg.inv(oo.Fs), oo.q2, oo.Ns, oo.ks, oo.f_x, oo.f_V, oo.p_x, oo.p_V, smpx_tmp, K)

                oo.smpx[:, 2:]           = smpx_tmp
                oo.smpx[:, 1, 0:ook-1]   = oo.smpx[:, 2, 1:]
                oo.smpx[:, 0, 0:ook-2]   = oo.smpx[:, 2, 2:]

                if oo.doBsmpx and (it % oo.BsmpxSkp == 0):
                    oo.Bsmpx[:, it // oo.BsmpxSkp, 2:]    = oo.smpx[:, 2:, 0]

                stds = _N.std(oo.smpx[:, 2:, 0], axis=1)
                oo.mnStds[it] = _N.mean(stds, axis=0)
                if len(hists) == 0:
                    print("!!!!!!  length hists is 0 before ARcfSmpl")
                ###  
                #_arcfs.ARcfSmpl(ooN+1, ook, oo.AR2lims, oo.smpx[:, 1:, 0:ook], oo.smpx[:, 0:, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR, prior=oo.use_prior, accepts=8, aro=oo.ARord, sig_ph0L=oo.sig_ph0L, sig_ph0H=oo.sig_ph0H)
                _arcfs.ARcfSmpl(ooN+1, ook, oo.AR2lims, oo.smpx[hists, 1:, 0:ook], oo.smpx[hists, 0:, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, len(hists), prior=oo.use_prior, accepts=8, aro=oo.ARord, sig_ph0L=oo.sig_ph0L, sig_ph0H=oo.sig_ph0H)
                oo.F_alfa_rep = alpR + alpC   #  new constructed
                prt, rank, f, amp = ampAngRep(oo.F_alfa_rep, f_order=True)
                #ut, wt = FilteredTimeseries(ooN+1, ook, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR)
                #ranks[it]    = rank
                oo.allalfas[it] = oo.F_alfa_rep

                for m in range(ooTR):
                    #oo.wts[m, it, :, :]   = wt[m, :, :, 0]
                    #oo.uts[m, it, :, :]   = ut[m, :, :, 0]
                    if not oo.bFixF:
                        oo.amps[it, :]  = amp
                        oo.fs[it, :]    = f

                oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]
                for tr in range(oo.TR):
                    oo.Fs[tr, 0]    = oo.F0[:]


                #print "len(lwsts) %(l)d   len(hists) %(h)d" % {"l" : len(lwsts), "h" : len(hists)}
                # sts2chg = hists
                # #sts2chg = lwsts
                # #if (it > oo.startZ) and oo.doS and len(sts2chg) > 0:
                # if oo.doS and len(sts2chg) > 0:
                #     AL = 0.5*_N.sum(oo.smpx[sts2chg, 2:, 0]*oo.smpx[sts2chg, 2:, 0]*oo.ws[sts2chg])
                #     #AL = 0.5*_N.sum(oo.smpx[sts2chg, 2:, 0]*oo.smpx[sts2chg, 2:, 0])
                #     BRL = kpOws[sts2chg] - BaS - oous_rs[sts2chg] - ARo[sts2chg] - oo.knownSig[sts2chg]
                #     BL = 0.5*_N.sum(oo.ws[sts2chg]*BRL*oo.smpx[sts2chg, 2:, 0])
                #     UL = BL / (2*AL)
                #     #sgL= 1/_N.sqrt(2*AL)
                #     sg2= 1./(2*AL)
                #     if it % 50 == 0:
                #         print("u  %(u).3f  %(s).3f" % {"u" : UL, "s" : _N.sqrt(sg2)})

                #     q2_pr = 0.25  # 0.05**2
                #     u_pr  = 1.
                #     #u_pr  = 0
                #     U = (u_pr * sg2 + UL * q2_pr) / (sg2 + q2_pr)
                #     sg= _N.sqrt((sg2*q2_pr) / (sg2 + q2_pr))

                #     #print "U  %(U).4f    UL %(UL).4f s  %(s).3f" % {"U" : U, "s" : sg, "UL" : UL}
                #     if _N.isnan(U):
                #         print("U is nan  UL %.4f" % UL)
                #         print("U is nan  AL %.4f" % AL)
                #         print("U is nan  BL %.4f" % BL)
                #         print("U is nan  BaS ")
                #         print("hists")
                #         print(hists)
                #         print("lwsts")
                #         print(lwsts)

                #     oo.s[1] = U + sg*_N.random.randn()
                #     #oo.s[0] = U + sg*_N.random.randn()

                #     _N.fill_diagonal(sd01[0], oo.s[0])
                #     _N.fill_diagonal(sd01[1], oo.s[1])
                #     #print oo.s[1]
                #     oo.smp_ss[it] = oo.s[1] 
                #     #oo.smp_ss[it] = oo.s[0] 

                #oo.a2 = oo.a_q2 + 0.5*(ooTR*ooN + 2)  #  N + 1 - 1
                oo.a2 = oo.a_q2 + 0.5*(len(hists)*ooN + 2)  #  N + 1 - 1
                BB2 = oo.B_q2
                #for m in range(ooTR):
                for m in hists:
                    #   set x00 
                    #oo.x00[m]      = oo.smpx[m, 2]*0.1
                    oo.x00[m]      = oo.smpx[m, 2]*0.1

                    #####################    sample q2
                    rsd_stp = oo.smpx[m, 3:,0] - _N.dot(oo.smpx[m, 2:-1], oo.F0).T
                    BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
                oo.q2[:] = _ss.invgamma.rvs(oo.a2, scale=BB2)

                oo.smp_q2[:, it]= oo.q2
                t7 = _tm.time()
                #print "gibbs iter %.3f" % (t7-t1)
                # if (it > 1) and (it % oo.peek == 0):
                #     fig = _plt.figure(figsize=(12, 8))
                #     fig.add_subplot(4, 1, 1)
                #     _plt.plot(oo.amps[1:it, 0])
                #     fig.add_subplot(4, 1, 2)
                #     _plt.plot(oo.fs[1:it, 0])
                #     fig.add_subplot(4, 1, 3)
                #     _plt.plot(oo.mnStds[1:it])
                #     fig.add_subplot(4, 1, 4)
                #     _plt.plot(oo.smp_ms[1:it, 0])

                #     _plt.savefig("%(dir)s/tmp-fsamps%(it)d" % {"dir" : oo.mcmcRunDir, "it" : it})
                #     _plt.close()

                #     oo.dump_smpsS(toiter=it, dir=oo.mcmcRunDir)
        #oo.dump_smpsS(dir=oo.mcmcRunDir)

    def run(self, datfilename, runDir, trials=None, minSpkCnt=0, pckl=None, runlatent=False, dontrun=False, h0_1=None, h0_2=None, h0_3=None, h0_4=None, h0_5=None, readSmpls=False, multiply_shape_hyperparam=1, multiply_scale_hyperparam=1, hist_timescale_ms=70, n_interior_knots=8):  
        oo     = self#

        global interrupted
        interrupted = False

        oo.Cs          = len(oo.freq_lims)
        oo.C           = oo.Cn + oo.Cs
        oo.R           = 1
        oo.k           = 2*oo.C + oo.R

        oo.u_x00        = _N.zeros(oo.k)
        oo.s2_x00       = _arl.dcyCovMat(oo.k, _N.ones(oo.k), 0.4)
        oo.restarts = 0

        oo.loadDat(runDir, datfilename, trials, h0_1=h0_1, h0_2=h0_2, h0_3=h0_3, h0_4=h0_4, h0_5=h0_5, multiply_shape_hyperparam=multiply_shape_hyperparam, multiply_scale_hyperparam=multiply_scale_hyperparam, hist_timescale_ms=hist_timescale_ms, n_interior_knots=n_interior_knots)
        oo.setParams()

        t1 = _tm.time()
        oo.dirichletAllocate()
        t2    = _tm.time()
        print (t2-t1)

    def dump_smpsS(self, dir=None, toiter=None):
        oo         = self
        pcklme     = {}
        
        toiter         = oo.ITERS if (toiter is None) else toiter
        pcklme["ss"]  = oo.smp_ss[0:toiter]
        pcklme["zs"]  = oo.smp_zs[:, 0:toiter]
        pcklme["ms"]  = oo.smp_ms[0:toiter]
        oo.__class__.__bases__[0].dump_smps(pcklme=pcklme, dir=dir, toiter=toiter)
        oo.dump_smps(pcklme=pcklme, dir=dir, toiter=toiter)
