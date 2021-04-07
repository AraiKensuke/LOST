import pickle
from LOST.kflib import createDataAR
import numpy as _N
import patsy
import re as _re
import matplotlib.pyplot as _plt

import scipy.stats as _ss

import numpy.polynomial.polynomial as _Npp
import time as _tm
import LOST.ARlib as _arl

cython_inv_v = 5  #  
if cython_inv_v == 2:
    import LOST.kfARlibMPmv_ram2 as _kfar
elif cython_inv_v == 3:
    import LOST.kfARlibMPmv_ram3 as _kfar
elif cython_inv_v == 5:
    import LOST.kfARlibMPmv_ram5 as _kfar

import pyPG as lw

cython_arc = True
if cython_arc:
    import LOST.ARcfSmplNoMCMC_ram as _arcfs
else:
    import LOST.ARcfSmplNoMCMC as _arcfs
#
#from ARcfSmpl2 import ARcfSmpl

import commdefs as _cd

from LOST.ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import os
import signal

import LOST.mcmcARspk as mcmcARspk
import LOST.monitor_gibbs as _mg

interrupted = False
def signal_handler(signal, frame):
    global interrupted
    print("******  INTERRUPT")
    interrupted = True

class mcmcARp(mcmcARspk.mcmcARspk):
    #  Description of model
    rn            = None    #  used for count data
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

    #  coefficient samplingF
    fSigMax       = 500.    #  fixed parameters
    #freq_lims     = [[1 / .85, fSigMax]]
    freq_lims     = [[.1, fSigMax]]
    sig_ph0L      = -1.
    sig_ph0H      = 0.

    #  1 offset for all trials
    bIndOffset    = True
    peek          = 400

    VIS           = None

    ignr          = 0

    

    def getComponents(self):
        """
        it0, it1 are gibbs iterations skipped, so should be like ITER//skp
        """
        oo = self
        skpdITER = oo.wts.shape[0]
        N  = oo.smpx.shape[1] - 2
        _rts = _N.empty((skpdITER, oo.TR, N+1, oo.R, 1))    #  real components   N = ddN
        _zts = _N.empty((skpdITER, oo.TR, N+1, oo.C, 1))    #  imag components 

        for it in range(skpdITER):
            if len(_N.where(_N.abs(oo.allalfas[it*oo.BsmpxSkp]) == 0)[0]) == 0:
                b, c = dcmpcff(alfa=oo.allalfas[it*oo.BsmpxSkp])
                for tr in range(oo.TR):
                    for r in range(oo.R):
                        _rts[it, tr, :, r] = b[r] * oo.uts[it, tr, r]

                    for z in range(oo.C):
                        #print "z   %d" % z
                        cf1 = 2*c[2*z].real
                        gam = oo.allalfas[it, oo.R+2*z]
                        cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
                        _zts[it, tr, 0:N+1, z] = cf1*oo.wts[it, tr, z, 1:N+2] - cf2*oo.wts[it, tr, z, 0:N+1]

        oo.rts = _N.array(_rts[:, :, 1:, :, 0])
        oo.zts = _N.array(_zts[:, :, 1:, :, 0])

        zts_stds = _N.mean(_N.std(oo.zts, axis=2), axis=1)  #  iter x C

        srtd     = _N.argsort(zts_stds, axis=1)     #  ITER x C
        for it in range(skpdITER):
            oo.fs[it] = oo.fs[it, srtd[it, ::-1]]
            oo.amps[it] = oo.amps[it, srtd[it, ::-1]]
            for tr in range(oo.TR):
                oo.zts[it, tr] = oo.zts[it, tr, :, srtd[it, ::-1]].T

    def gibbsSamp(self, smpls_fn_incl_trls=False):  ###########################  GIBBSSAMPH
        global interrupted
        oo          = self

        signal.signal(signal.SIGINT, signal_handler)

        print("****!!!!!!!!!!!!!!!!  dohist  %s" % str(oo.dohist))

        ooTR        = oo.TR
        ook         = oo.k

        ooN         = oo.N
        _kfar.init(oo.N, oo.k, oo.TR)
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))
        if oo.dohist:
            oo.loghist = _N.zeros(oo.Hbf.shape[0])
        else:
            print("fixed hist is")
            print(oo.loghist)

        print("oo.mcmcRunDir    %s" % oo.mcmcRunDir)
        if oo.mcmcRunDir is None:
            oo.mcmcRunDir = ""
        elif (len(oo.mcmcRunDir) > 0) and (oo.mcmcRunDir[-1] != "/"):
            oo.mcmcRunDir += "/"

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
        bConstPSTH   = False

        D_f          = _N.diag(_N.ones(oo.B.shape[0])*oo.s2_a)   #  spline
        iD_f = _N.linalg.inv(D_f)
        D_u  = _N.diag(_N.ones(oo.TR)*oo.s2_u)   #  This should 
        iD_u = _N.linalg.inv(D_u)
        iD_u_u_u = _N.dot(iD_u, _N.ones(oo.TR)*oo.u_u)

        if oo.bpsth:
            BDB  = _N.dot(oo.B.T, _N.dot(D_f, oo.B))
            DB   = _N.dot(D_f, oo.B)
            BTua = _N.dot(oo.B.T, oo.u_a)

        it    = -1

        oous_rs = oo.us.reshape((ooTR, 1))
        #runTO = ooNMC + oo.burn - 1 if (burns is None) else (burns - 1)
        runTO = oo.ITERS - 1
        oo.allocateSmp(runTO+1, Bsmpx=oo.doBsmpx)
        if cython_arc:
            _arcfs.init(ooN+1-oo.ignr, oo.k, oo.TR, oo.R, oo.Cs, oo.Cn, aro=_cd.__NF__)
            alpR   = _N.array(oo.F_alfa_rep[0:oo.R])
            alpC   = _N.array(oo.F_alfa_rep[oo.R:])
        else:
            alpR   = oo.F_alfa_rep[0:oo.R]
            alpC   = oo.F_alfa_rep[oo.R:]

        BaS = _N.zeros(oo.N+1)#_N.empty(oo.N+1)

        #  H shape    100 x 9
        Hbf = oo.Hbf

        RHS = _N.empty((oo.histknots, 1))

        print("-----------    histknots %d" % oo.histknots)

        cInds = _N.arange(oo.iHistKnotBeginFixed, oo.histknots)
        vInds = _N.arange(0, oo.iHistKnotBeginFixed)
        #cInds = _N.array([4, 12, 13])
        #vInds = _N.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, ])
        #vInds = _N.arange(0, oo.iHistKnotBeginFixed)

        RHS[cInds, 0] = 0

        Msts = []
        for m in range(ooTR):
            Msts.append(_N.where(oo.y[m] == 1)[0])
        HcM  = _N.ones((len(vInds), len(vInds)))

        HbfExpd = _N.zeros((oo.histknots, ooTR, oo.N+1))


        #HbfExpd = _N.zeros((oo.histknots, ooTR, oo.Hbf.shape[0]))
        #  HbfExpd is 11 x M x 1200
        #  find the mean.  For the HISTORY TERM
        for i in range(oo.histknots):
            for m in range(oo.TR):
                sts = Msts[m]
                HbfExpd[i, m, 0:sts[0]] = 0
                for iss in range(len(sts)-1):
                    t0  = sts[iss]
                    t1  = sts[iss+1]
                    #HbfExpd[i, m, t0+1:t1+1] = Hbf[1:t1-t0+1, i]#Hbf[0:t1-t0, i]
                    HbfExpd[i, m, t0+1:t1+1] = Hbf[0:t1-t0, i]
                HbfExpd[i, m, sts[-1]+1:] = 0

        _N.dot(oo.B.T, oo.aS, out=BaS)
        if oo.hS is None:
            oo.hS = _N.zeros(oo.histknots)

        if oo.dohist:
            _N.dot(Hbf, oo.hS, out=oo.loghist)
        oo.stitch_Hist(ARo, oo.loghist, Msts)

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

        iterBLOCKS  = oo.ITERS//oo.peek
        smpx_C_cont = _N.empty((oo.TR, oo.N+1, oo.k))   #  need C contiguous

        #  oo.smpx[:, 1+oo.ignr:, 0:ook], oo.smpx[:, oo.ignr:, 0:ook-1]
        smpx_contiguous1        = _N.zeros((oo.TR, oo.N + 2, oo.k))
        smpx_contiguous2        = _N.zeros((oo.TR, (oo.N + 1) + 2, oo.k-1))
        if (cython_inv_v == 3) or (cython_inv_v == 5):
            oo.if_V = _N.array(oo.f_V)
            oo.chol_L_fV = _N.array(oo.f_V)
        ######  Gibbs sampling procedure
        ttts = _N.zeros((oo.ITERS, 9))
        for itrB in range(iterBLOCKS):
            it = itrB*oo.peek
            if it > 0:
                #  0.5*oo.fs  because (dt*2)  ->  1 corresponds to Fs/2

                print("---------it: %(it)d    mnStd  %(mnstd).3f" % {"it" : itrB*oo.peek, "mnstd" : oo.mnStds[it-1]})
                if not oo.noAR:
                    print(prt)
                mnttt = _N.mean(ttts[0:it-1], axis=0)
                for ti in range(9):
                    print("t%(2)d-t%(1)d  %(ttt).4f" % {"1" : ti+1, "2" : ti+2, "ttt" : mnttt[ti]})

            if interrupted:
                break
            for it in range(itrB*oo.peek, (itrB+1)*oo.peek):
                ttt1 = _tm.time()

                itstore = it // oo.BsmpxSkp

                #  generate latent AR state
                oo.f_x[:, 0]     = oo.x00
                if it == 0:
                    for m in range(ooTR):
                        oo.f_V[m, 0]     = oo.s2_x00
                else:
                    oo.f_V[:, 0]     = _N.mean(oo.f_V[:, 1:], axis=1)

                ###  PG latent variable sample
                ttt2 = _tm.time()

                for m in range(ooTR):
                    lw.rpg_devroye(oo.rn, oo.smpx[m, 2:, 0] + oo.us[m] + BaS + ARo[m] + oo.knownSig[m], out=oo.ws[m])  ######  devryoe
                ttt3 = _tm.time()

                if ooTR == 1:
                    oo.ws   = oo.ws.reshape(1, ooN+1)
                _N.divide(oo.kp, oo.ws, out=kpOws)

                if oo.dohist:
                    O = kpOws - oo.smpx[..., 2:, 0] - oo.us.reshape((ooTR, 1)) - BaS -  oo.knownSig

                    #print(oo.ws)

                    # for i in vInds:
                    #     #print("i   %d" % i)
                    #     #print(_N.sum(HbfExpd[i]))
                    #     for j in vInds:
                    #         #print("j   %d" % j)
                    #         #print(_N.sum(HbfExpd[j]))
                    #         HcM[i-iOf, j-iOf] = _N.sum(oo.ws*HbfExpd[i]*HbfExpd[j])

                    #     RHS[i, 0] = _N.sum(oo.ws*HbfExpd[i]*O)
                    #     for cj in cInds:
                    #         RHS[i, 0] -= _N.sum(oo.ws*HbfExpd[i]*HbfExpd[cj])*RHS[cj, 0]
                    for ii in range(len(vInds)):
                        #print("i   %d" % i)
                        #print(_N.sum(HbfExpd[i]))
                        i = vInds[ii]
                        for jj in range(len(vInds)):
                            j = vInds[jj]
                            #print("j   %d" % j)
                            #print(_N.sum(HbfExpd[j]))
                            HcM[ii, jj] = _N.sum(oo.ws*HbfExpd[i]*HbfExpd[j])

                        RHS[ii, 0] = _N.sum(oo.ws*HbfExpd[i]*O)
                        for cj in cInds:
                            RHS[ii, 0] -= _N.sum(oo.ws*HbfExpd[i]*HbfExpd[cj])*RHS[cj, 0]
                        

                    # print("HbfExpd..............................")
                    # for i in range(oo.histknots):
                    #     print(_N.sum(HbfExpd[i]))
                    # print("HcM..................................")
                    # print(HcM)
                    # print("RHS..................................")
                    # print(RHS[vInds])
                    vm = _N.linalg.solve(HcM, RHS[vInds])
                    Cov = _N.linalg.inv(HcM)
                    #print vm
                    #print(Cov)
                    #print(vm[:, 0])
                    cfs = _N.random.multivariate_normal(vm[:, 0], Cov, size=1)

                    RHS[vInds,0] = cfs[0]
                    oo.smp_hS[it] = RHS[:, 0]

                    #RHS[2:6, 0] = vm[:, 0]
                    #vv = _N.dot(Hbf, RHS)
                    #print vv.shape
                    #print oo.loghist.shape
                    _N.dot(Hbf, RHS[:, 0], out=oo.loghist)
                    oo.smp_hist[it] = oo.loghist
                    oo.stitch_Hist(ARo, oo.loghist, Msts)
                else:
                    oo.smp_hist[it] = oo.loghist
                    oo.stitch_Hist(ARo, oo.loghist, Msts)

                #  Now that we have PG variables, construct Gaussian timeseries
                #  ws(it+1)    using u(it), F0(it), smpx(it)

                #  cov matrix, prior of aS 

                # oo.gau_obs = kpOws - BaS - ARo - oous_rs - oo.knownSig
                # oo.gau_var =1 / oo.ws   #  time dependent noise
                ttt4 = _tm.time()
                if oo.bpsth:
                    Oms  = kpOws - oo.smpx[..., 2:, 0] - ARo - oous_rs - oo.knownSig
                    _N.einsum("mn,mn->n", oo.ws, Oms, out=smWimOm)   #  sum over
                    ilv_f  = _N.diag(_N.sum(oo.ws, axis=0))
                    #  diag(_N.linalg.inv(Bi)) == diag(1./Bi).  Bii = inv(Bi)
                    _N.fill_diagonal(lv_f, 1./_N.diagonal(ilv_f))
                    lm_f  = _N.dot(lv_f, smWimOm)  #  nondiag of 1./Bi are inf
                    #  now sample
                    iVAR = _N.dot(oo.B, _N.dot(ilv_f, oo.B.T)) + iD_f
                    ttt4a = _tm.time()
                    VAR  = _N.linalg.inv(iVAR)  #  knots x knots
                    ttt4b = _tm.time()
                    #iBDBW = _N.linalg.inv(BDB + lv_f)   # BDB not diag
                    #Mn    = oo.u_a + _N.dot(DB, _N.dot(iBDBW, lm_f - BTua))

                    #  BDB + lv_f     (N+1 x N+1)
                    #  lm_f - BTua    (N+1)
                    Mn = oo.u_a + _N.dot(DB, _N.linalg.solve(BDB + lv_f, lm_f - BTua))

                    #t4c = _tm.time()

                    oo.aS   = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                    oo.smp_aS[it] = oo.aS
                    _N.dot(oo.B.T, oo.aS, out=BaS)

                ttt5 = _tm.time()
                ########     per trial offset sample  burns==None, only psth fit
                Ons  = kpOws - oo.smpx[..., 2:, 0] - ARo - BaS - oo.knownSig

                #  solve for the mean of the distribution


                if not oo.bpsth:  # if not doing PSTH, don't constrain offset, as there are no confounds controlling offset
                    _N.einsum("mn,mn->m", oo.ws, Ons, out=smWinOn)  #  sum over trials
                    ilv_u  = _N.diag(_N.sum(oo.ws, axis=1))  #  var  of LL
                    #  diag(_N.linalg.inv(Bi)) == diag(1./Bi).  Bii = inv(Bi)
                    _N.fill_diagonal(lv_u, 1./_N.diagonal(ilv_u))
                    lm_u  = _N.dot(lv_u, smWinOn)  #  nondiag of 1./Bi are inf, mean LL
                    #  now sample
                    iVAR = ilv_u + iD_u
                    VAR  = _N.linalg.inv(iVAR)  #
                    Mn    = _N.dot(VAR, _N.dot(ilv_u, lm_u) + iD_u_u_u)
                    oo.us[:]  = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                    if not oo.bIndOffset:
                        oo.us[:] = _N.mean(oo.us)
                    oo.smp_u[it] = oo.us
                else:
                    H    = _N.ones((oo.TR-1, oo.TR-1)) * _N.sum(oo.ws[0])
                    uRHS = _N.empty(oo.TR-1)
                    for dd in range(1, oo.TR):
                        H[dd-1, dd-1] += _N.sum(oo.ws[dd])
                        uRHS[dd-1] = _N.sum(oo.ws[dd]*Ons[dd] - oo.ws[0]*Ons[0])

                    MM  = _N.linalg.solve(H, uRHS)
                    Cov = _N.linalg.inv(H)

                    oo.us[1:] = _N.random.multivariate_normal(MM, Cov, size=1)
                    oo.us[0]  = -_N.sum(oo.us[1:])
                    if not oo.bIndOffset:
                        oo.us[:] = _N.mean(oo.us)
                    oo.smp_u[it] = oo.us

                # Ons  = kpOws - ARo
                # _N.einsum("mn,mn->m", oo.ws, Ons, out=smWinOn)  #  sum over trials
                # ilv_u  = _N.diag(_N.sum(oo.ws, axis=1))  #  var  of LL
                # #  diag(_N.linalg.inv(Bi)) == diag(1./Bi).  Bii = inv(Bi)
                # _N.fill_diagonal(lv_u, 1./_N.diagonal(ilv_u))
                # lm_u  = _N.dot(lv_u, smWinOn)  #  nondiag of 1./Bi are inf, mean LL
                # #  now sample
                # iVAR = ilv_u + iD_u
                # VAR  = _N.linalg.inv(iVAR)  #
                # Mn    = _N.dot(VAR, _N.dot(ilv_u, lm_u) + iD_u_u_u)
                # oo.us[:]  = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                # if not oo.bIndOffset:
                #     oo.us[:] = _N.mean(oo.us)
                # oo.smp_u[:, it] = oo.us

                ttt6 = _tm.time()
                if not oo.noAR:
                #  _d.F, _d.N, _d.ks, 
                    #_kfar.armdl_FFBS_1itrMP(oo.gau_obs, oo.gau_var, oo.Fs, _N.linalg.inv(oo.Fs), oo.q2, oo.Ns, oo.ks, oo.f_x, oo.f_V, oo.p_x, oo.p_V, oo.smpx, K)

                    oo.gau_obs = kpOws - BaS - ARo - oous_rs - oo.knownSig
                    oo.gau_var =1 / oo.ws   #  time dependent noise

                    #print(oo.Fs)
                    #print(_N.linalg.inv(oo.Fs))
                    if (cython_inv_v == 2):
                        _kfar.armdl_FFBS_1itrMP(oo.gau_obs, oo.gau_var, oo.Fs, _N.linalg.inv(oo.Fs), oo.q2, oo.Ns, oo.ks, oo.f_x, oo.f_V, oo.p_x, oo.p_V, smpx_C_cont, K)
                    else:
                        _kfar.armdl_FFBS_1itrMP(oo.gau_obs, oo.gau_var, oo.Fs, _N.linalg.inv(oo.Fs), oo.q2, oo.Ns, oo.ks, oo.f_x, oo.f_V, oo.chol_L_fV, oo.if_V, oo.p_x, oo.p_V, smpx_C_cont, K)

                    oo.smpx[:, 2:]           = smpx_C_cont
                    oo.smpx[:, 1, 0:ook-1]   = oo.smpx[:, 2, 1:]
                    oo.smpx[:, 0, 0:ook-2]   = oo.smpx[:, 2, 2:]

                    if oo.doBsmpx and (it % oo.BsmpxSkp == 0):
                        oo.Bsmpx[it // oo.BsmpxSkp, :, 2:]    = oo.smpx[:, 2:, 0]
                        #oo.Bsmpx[it // oo.BsmpxSkp, :, 2:]    = oo.smpx[:, 2:, 0]
                    stds = _N.std(oo.smpx[:, 2+oo.ignr:, 0], axis=1)
                    oo.mnStds[it] = _N.mean(stds, axis=0)

                    ttt7 = _tm.time()
                    #print("..................................")
                    #print(alpR)
                    #print(alpC)

                    #print(alpR)
                    #print(alpC)

                    # print(oo.smpx[0, 0:20, 0])
                    # print(oo.q2)

                    if cython_arc:
                        _N.copyto(smpx_contiguous1, 
                                  oo.smpx[:, 1+oo.ignr:])
                        _N.copyto(smpx_contiguous2, 
                                  oo.smpx[:, oo.ignr:, 0:ook-1])

                        #ARcfSmpl(int N, int k, int TR, AR2lims_nmpy, smpxU, smpxW, double[::1] q2, int R, int Cs, int Cn, complex[::1] valpR, complex[::1] valpC, double sig_ph0L, double sig_ph0H, double prR_s2)

                        oo.uts[itstore], oo.wts[itstore] = _arcfs.ARcfSmpl(ooN+1-oo.ignr, ook, oo.TR, oo.AR2lims, smpx_contiguous1, smpx_contiguous2, oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.sig_ph0L, oo.sig_ph0H, 0.2*0.2)
                    else:
                        oo.uts[itstore], oo.wts[itstore] = _arcfs.ARcfSmpl(ooN+1-oo.ignr, ook, oo.AR2lims, oo.smpx[:, 1+oo.ignr:, 0:ook], oo.smpx[:, oo.ignr:, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR, aro=oo.ARord, sig_ph0L=oo.sig_ph0L, sig_ph0H=oo.sig_ph0H)
                    #oo.F_alfa_rep = alpR + alpC   #  new constructed
                    oo.F_alfa_rep[0:oo.R] = alpR
                    oo.F_alfa_rep[oo.R:]  = alpC
                    
                    prt, rank, f, amp = ampAngRep(oo.F_alfa_rep, oo.dt, f_order=True)
                    #print(f)
                    #print(amp)
                    ttt8 = _tm.time()
                        #print prt
                    #ut, wt = FilteredTimeseries(ooN+1, ook, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR)
                    #ranks[it]    = rank
                    oo.allalfas[it] = oo.F_alfa_rep

                    for m in range(ooTR):
                        #oo.wts[m, it, :, :]   = wt[m, :, :, 0]
                        #oo.uts[m, it, :, :]   = ut[m, :, :, 0]
                        if not oo.bFixF:
                            oo.amps[it, :]  = amp
                            oo.fs[it, :]    = f

                    ttt9 = _tm.time()
                    oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]
                    for tr in range(oo.TR):
                        oo.Fs[tr, 0]    = oo.F0[:]

                    #  sample u     WE USED TO Do this after smpx
                    #  u(it+1)    using ws(it+1), F0(it), smpx(it+1), ws(it+1)

                    oo.a2 = oo.a_q2 + 0.5*(ooTR*ooN + 2)  #  N + 1 - 1
                    #oo.a2 = 0.5*(ooTR*(ooN-oo.ignr) + 2)  #  N + 1 - 1
                    BB2 = oo.B_q2
                    #BB2 = 0
                    for m in range(ooTR):
                        #   set x00 
                        oo.x00[m]      = oo.smpx[m, 2]*0.1

                        #####################    sample q2
                        rsd_stp = oo.smpx[m, 3+oo.ignr:,0] - _N.dot(oo.smpx[m, 2+oo.ignr:-1], oo.F0).T
                        #oo.rsds[it, m] = _N.dot(rsd_stp, rsd_stp.T)
                        BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
                        
                        
                    oo.q2[:] = _ss.invgamma.rvs(oo.a2, scale=BB2)
                    oo.smp_q2[it]= oo.q2
                    ttt10 = _tm.time()
                else:
                    ttt7 = ttt8 = ttt9 = ttt10 = ttt6

                ttt10 = _tm.time()
                ttts[it, 0] = ttt2-ttt1
                ttts[it, 1] = ttt3-ttt2
                ttts[it, 2] = ttt4-ttt3
                ttts[it, 3] = ttt5-ttt4
                ttts[it, 4] = ttt6-ttt5
                ttts[it, 5] = ttt7-ttt6
                ttts[it, 6] = ttt8-ttt7
                ttts[it, 7] = ttt9-ttt8
                ttts[it, 8] = ttt10-ttt9

            oo.last_iter = it
            if it > oo.minITERS:
                smps = _N.empty((3, it+1))
                smps[0, :it+1] = oo.amps[:it+1, 0]

                smps[1, :it+1] = oo.fs[:it+1, 0]
                smps[2, :it+1] = oo.mnStds[:it+1]

                #frms = _mg.stationary_from_Z_bckwd(smps, blksz=oo.peek)
                if _mg.stationary_test(oo.amps[:it+1, 0], oo.fs[:it+1, 0], oo.mnStds[:it+1], it+1, blocksize=oo.mg_blocksize, points=oo.mg_points):
                    break

                """
                fig = _plt.figure(figsize=(8, 8))
                fig.add_subplot(3, 1, 1)
                _plt.plot(range(1, it), oo.amps[1:it, 0], color="grey", lw=1.5)
                _plt.plot(range(0, it), oo.amps[0:it, 0], color="black", lw=3)
                _plt.ylabel("amp")
                fig.add_subplot(3, 1, 2)
                _plt.plot(range(1, it), oo.fs[1:it, 0]/(2*oo.dt), color="grey", lw=1.5)
                _plt.plot(range(0, it), oo.fs[0:it, 0]/(2*oo.dt), color="black", lw=3)
                _plt.ylabel("f")
                fig.add_subplot(3, 1, 3)
                _plt.plot(range(1, it), oo.mnStds[1:it], color="grey", lw=1.5)
                _plt.plot(range(0, it), oo.mnStds[0:it], color="black", lw=3)
                _plt.ylabel("amp")
                _plt.xlabel("iter")
                _plt.savefig("%(dir)stmp-fsamps%(it)d" % {"dir" : oo.mcmcRunDir, "it" : it+1})
                fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
                _plt.close()
                """
                #if it - frms > oo.stationaryDuration:
                #   break

        oo.getComponents()
        oo.dump_smps(0, toiter=(it+1), dir=oo.mcmcRunDir, smpls_fn_incl_trls=smpls_fn_incl_trls)
        #oo.VIS = ARo   #  to examine this from outside




    def run(self, datfilename, runDir, trials=None, minSpkCnt=0, pckl=None, readSmpls=False, multiply_shape_hyperparam=1, multiply_scale_hyperparam=1, hist_timescale_ms=70, n_interior_knots=8, smpls_fn_incl_trls=False, psth_run=False, psth_knts=10): ###########  RUN
        """
        hist_timescale_ms  rough timescale of the post-spike history, counted from 10th percentile ISI
        """
        global interrupted
        oo     = self    #  call self oo.  takes up less room on line
        interrupted = False

        oo.C           = oo.Cn + oo.Cs
        oo.Cn          = 0
        oo.Cs          = oo.C

        #oo.Cs          = len(oo.freq_lims)


        # oo.Cn           = 0
        # oo.Cs           = oo.C
        oo.k           = 2*oo.C + oo.R

        #  x0  --  Gaussian prior
        oo.u_x00        = _N.zeros(oo.k)
        oo.s2_x00       = _arl.dcyCovMat(oo.k, _N.ones(oo.k), 0.4)
        oo.restarts = 0

        oo.loadDat(runDir, datfilename, trials, multiply_shape_hyperparam=multiply_shape_hyperparam, multiply_scale_hyperparam=multiply_scale_hyperparam, hist_timescale_ms=hist_timescale_ms, n_interior_knots=n_interior_knots)

        oo.setParams(psth_run=psth_run, psth_knts=psth_knts)

        if not psth_run:
            t1    = _tm.time()
            oo.gibbsSamp(smpls_fn_incl_trls=smpls_fn_incl_trls)
            t2    = _tm.time()
            print("time:  %.3f" % (t2-t1))
