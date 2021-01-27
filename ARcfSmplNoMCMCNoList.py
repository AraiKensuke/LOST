import numpy.polynomial.polynomial as _Npp
import scipy.stats as _ss
import LOST.kdist as _kd
import LOST.ARlib as _arl
import warnings
#import logerfc as _lfc
import LOST.commdefs as _cd
import numpy as _N
#from ARcfSmplFuncs import ampAngRep, randomF, dcmpcff, betterProposal
from LOST.ARcfSmplFuncs import ampAngRep, dcmpcff
#import ARcfSmplFuncsCy as ac
import matplotlib.pyplot as _plt
import time as _tm


def ARcfSmpl(N, k, AR2lims, smpxU, smpxW, q2, R, Cs, Cn, alpR, alpC, TR, accepts=1, prior=_cd.__COMP_REF__, aro=_cd.__NF__, sig_ph0L=-1, sig_ph0H=0):
    """
    Cs   - number of signal roots
    Cn   - number of noise roots
    """
    ttt1 = _tm.time()

    valpR = _N.array(alpR)
    valpC = _N.array(alpC)
    maskR = _N.ones(len(alpR))
    maskC = _N.ones(len(alpC))

    C = Cs + Cn

    #  I return F and Eigvals

    ujs     = _N.empty((TR, R, N + 1, 1))
    wjs     = _N.empty((TR, C, N + 2, 1))
    
    #  CONVENTIONS, DATA FORMAT
    #  x1...xN  (observations)   size N-1
    #  x{-p}...x{0}  (initial values, will be guessed)
    #  smpx
    #  ujt      depends on x_t ... x_{t-p-1}  (p-1 backstep operators)
    #  1 backstep operator operates on ujt
    #  wjt      depends on x_t ... x_{t-p-2}  (p-2 backstep operators)
    #  2 backstep operator operates on wjt
    #  
    #  smpx is N x p.  The first element has x_0...x_{-p} in it
    #  For real filter
    #  prod_{i neq j} (1 - alpi B) x_t    operates on x_t...x_{t-p+1}
    #  For imag filter
    #  prod_{i neq j,j+1} (1 - alpi B) x_t    operates on x_t...x_{t-p+2}

    ######  COMPLEX ROOTS.  Cannot directly sample the conditional posterior

    Ff  = _N.zeros((1, k-1))
    F0  = _N.zeros(2)
    F1  = _N.zeros(2)
    A   = _N.empty(2)

    Xs     = _N.empty((TR, N-2, 2))
    Ys     = _N.empty((N-2, 1))
    H      = _N.empty((TR, 2, 2))
    iH     = _N.empty((TR, 2, 2))
    mu     = _N.empty((TR, 2, 1))
    J      = _N.empty((2, 2))
    Mj     = _N.empty(TR)
    Mji    = _N.empty(TR)
    mj     = _N.empty(TR)
    filtrootsC = _N.empty(2*C-2+R, dtype=_N.complex)
    filtrootsR = _N.empty(2*C+R-1, dtype=_N.complex)

    #  r = sqrt(-1*phi_1)   0.25 = -1*phi_1   -->  phi_1 >= -0.25   gives r >= 0.5 for signal components   
    if aro == _cd.__SF__:    #  signal first
        arInd = range(C)
    else:                    #  noise first
        arInd = range(C-1, -1, -1)

    ttt2 = _tm.time()
    ttt2a = 0
    ttt2b = 0
    for c in arInd:   #  Filtering signal root first drastically alters the strength of the signal root upon update sometimes.  
        # rprior = prior
        # if c >= Cs:
        #    rprior = _cd.__COMP_REF__
        tttA = _tm.time()
        if c >= Cs:
            ph0L = -1
            ph0H = 0
        else:
            ph0L = sig_ph0L   # 
            ph0H = sig_ph0H #  R=0.97, reasonably oscillatory
            
        j = 2*c + 1
        p1a =  AR2lims[c, 0]
        p1b =  AR2lims[c, 1]
        
        iif = -1
        for ii in range(C):
            #if (ii != j) and (ii != j-1):
            if (ii != c):
                iif += 1
                filtrootsC[iif] = valpC[2*ii]
                iif += 1
                filtrootsC[iif] = valpC[2*ii+1]
        for ii in range(R):
            filtrootsC[ii+2*C-2] = valpR[ii]
        # given all other roots except the jth.  This is PHI0
            
        jth_r1 = alpC.pop(j)    #  negative root   #  nothing is done with these
        jth_r2 = alpC.pop(j-1)  #  positive root

        #  exp(-(Y - FX)^2 / 2q^2)

        #Frmj = _Npp.polyfromroots(alpR + alpC).real
        Frmj = _Npp.polyfromroots(filtrootsC).real
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(_N.array(alpR + alpC))
        # print(filtrootsC)
        
        #print "ampAngRep"
        #print ampAngRep(alpC)

        Ff[0, :]   = Frmj[::-1]
        #  Ff first element k-delay,  Prod{i neq j} (1 - alfa_i B)

        ##########  CREATE FILTERED TIMESERIES   ###########
        ##########  Frmj is k x k, smpxW is (N+2) x k ######

        for m in range(TR):
            _N.dot(smpxW[m], Ff.T, out=wjs[m, c])
            #_N.savetxt("%(2)d-%(1)d" % {"2" : aro, "1" : c}, wjs[m, c])
            #_plt.figure(figsize=(15, 4))
            #_plt.plot(wjs[m, c, 1000:1300])

            ####   Needed for proposal density calculation
            #Ys[:, 0]    = wj[2:N, 0]
            Ys[:]    = wjs[m, c, 2:N]
            Xs[m, :, 0] = wjs[m, c, 1:N-1, 0]   # new old
            Xs[m, :, 1] = wjs[m, c, 0:N-2, 0]
            iH[m]       = _N.dot(Xs[m].T, Xs[m]) / q2[m]
            #H[m]        = _N.linalg.inv(iH[m])   #  aka A
            H[m,0,0]=iH[m,1,1]; H[m,1,1]=iH[m,0,0];
            H[m,1,0]=-iH[m,1,0];H[m,0,1]=-iH[m,0,1];
            H[m]         /= (iH[m,0,0]*iH[m,1,1]-iH[m,0,1]*iH[m,1,0])
            mu[m]        = _N.dot(H[m], _N.dot(Xs[m].T, Ys))/q2[m]

        #  
        Ji  = _N.sum(iH, axis=0)
        #J   = _N.linalg.inv(Ji)
        J[0,0]=Ji[1,1]; J[1,1]=Ji[0,0];
        J[1,0]=-Ji[1,0];J[0,1]=-Ji[0,1];
        J         /= (Ji[0,0]*Ji[1,1]-Ji[0,1]*Ji[1,0])

        U   = _N.dot(J, _N.einsum("tij,tjk->ik", iH, mu))

        #bBdd, iBdd, mags, vals = _arl.ARevals(U[:, 0])
        #print "U"
        #print U[:, 0]
        #print ":::::: *****"
        #print ampAngRep(vals)

        ##########  Sample *PROPOSED* parameters 

        # #  If posterior valid Gaussian    q2 x H - oNZ * prH00

        ###  This case is still fairly inexpensive.  
        vPr1  = J[0, 0] - (J[0, 1]*J[0, 1])   / J[1, 1]   # known as Mj
        vPr2  = J[1, 1]
        svPr2 = _N.sqrt(vPr2)     ####  VECTORIZE
        svPr1 = _N.sqrt(vPr1)

        #b2Real = (U[1, 0] + 0.25*U[0, 0]*U[0, 0] > 0)
        ######  Initialize F0

        mu1prp = U[1, 0]
        mu0prp = U[0, 0]

        tttB = _tm.time()
        #print "ph0L  %(L).4f  ph1L  %(H).4f   %(u).4f   %(s).4f" % {"L" : ph0L, "H" : ph0H, "u" : mu1prp, "s" : svPr2}
        #print "vPr1  %(1).4e  vPr2 %(2).4e" % {"1" : vPr1, "2" : vPr2}
        ph0j2 = _kd.truncnorm(a=ph0L, b=ph0H, u=mu1prp, std=svPr2)
        r1    = _N.sqrt(-1*ph0j2)
        mj0   = mu0prp + (J[0, 1] * (ph0j2 - mu1prp)) / J[1, 1]
        ph0j1 = _kd.truncnorm(a=p1a*r1, b=p1b*r1, u=mj0, std=svPr1)
        tttC = _tm.time()
        A[0] = ph0j1; A[1] = ph0j2

        #F0[0] = ph0j1; F0[1] = ph0j2

        #  F1 +/- sqrt(F1^2 + 4F1) / 
        img        = _N.sqrt(-(A[0]*A[0] + 4*A[1]))*1j
        #vals, vecs = _N.linalg.eig(_N.array([A, [1, 0]]))
            
        #  the positive root comes first
        valpC[j-1] = (A[0] - img)*0.5
        valpC[j]   = (A[0] + img)*0.5
        alpC.insert(j-1, (A[0] - img)*0.5)     #alpC.insert(j - 1, jth_r1)
        alpC.insert(j-1, (A[0] + img)*0.5)     #alpC.insert(j - 1, jth_r2)
        tttC = _tm.time()
        ttt2a += tttB - tttA
        ttt2b += tttC - tttB

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print(alpC)
    # print(valpC)
    Ff  = _N.zeros((1, k))
    ######     REAL ROOTS.  Directly sample the conditional posterior
    for j in range(R - 1, -1, -1):
        # given all other roots except the jth
        #jth_r = alpR.pop(j)

        for ii in range(C):
            filtrootsR[2*ii] = valpC[2*ii]
            filtrootsR[2*ii+1] = valpC[2*ii+1]
        iif = -1
        for ii in range(R):
            if ii != j:
                iif += 1
                filtrootsR[iif+2*C] = valpR[ii]

        #print("R !!!!")
        #print(alpR + alpC)
        #print(filtrootsR)
        Frmj = _Npp.polyfromroots(filtrootsR).real #  Ff first element k-delay
        Ff[0, :] = Frmj[::-1]   #  Prod{i neq j} (1 - alfa_i B)

        ##########  CREATE FILTERED TIMESERIES   ###########
        ##########  Frmj is k x k, smpxU is (N+1) x k ######

        for m in range(TR):
            #uj  = _N.dot(Ff, smpxU[m].T).T
            #_N.dot(Ff, smpxU[m].T, out=ujs[m, j])
            _N.dot(smpxU[m], Ff.T, out=ujs[m, j])
            #ujs.append(uj)

            ####   Needed for proposal density calculation
            
            Mji[m] = _N.dot(ujs[m, j, 0:N, 0], ujs[m, j, 0:N, 0]) / q2[m]
            Mj[m] = 1 / Mji[m]
            mj[m] = _N.dot(ujs[m, j, 1:, 0], ujs[m, j, 0:N, 0]) / (q2[m]*Mji[m])

        #  truncated Gaussian to [-1, 1]
        Ji = _N.sum(Mji)
        J  = 1 / Ji
        U  = J * _N.dot(Mji, mj)

        #  we only want 
        rj = _kd.truncnorm(a=-1, b=1, u=U, std=_N.sqrt(J))

        #alpR.insert(j, rj)
        valpR[j] = rj
    alpR[:] = valpR
    alpC[:] = valpC
    ttt3 = _tm.time()

    # print("--------------------------------")
    # print("t2-t1  %.4f" % (ttt2-ttt1))
    # print("t3-t2  %.4f" % (ttt3-ttt2))
    # print("ttt2a  %.4f" % ttt2a)
    # print("ttt2b  %.4f" % ttt2b)


    #return ujs, wjs#, lsmpld

