import numpy.polynomial.polynomial as _Npp
import scipy.stats as _ss
import kdist as _kd
import ARlib as _arl
import warnings
#import logerfc as _lfc
import commdefs as _cd
import numpy as _N
#from ARcfSmplFuncs import ampAngRep, randomF, dcmpcff, betterProposal
from ARcfSmplFuncs import ampAngRep, dcmpcff, betterProposal
#import ARcfSmplFuncsCy as ac
import matplotlib.pyplot as _plt


def ARcfSmpl(N, k, AR2lims, smpxU, smpxW, q2, R, Cs, Cn, alpR, alpC, TR, accepts=1, prior=_cd.__COMP_REF__, aro=_cd.__NF__, sig_ph0L=-1, sig_ph0H=0):
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

    Ff  = _N.zeros((1, k-1))  #  1 - a_1 x_1 - a_2 x_2   (why k-1)
    F0  = _N.zeros(2)
    F1  = _N.zeros(2)

    #  r = sqrt(-1*phi_1)   0.25 = -1*phi_1   -->  phi_1 >= -0.25   gives r >= 0.5 for signal components   
    if aro == _cd.__SF__:    #  signal first
        arInd = range(C)
    else:                    #  noise first
        arInd = range(C-1, -1, -1)

    for c in arInd:   #  Filtering signal root first drastically alters the strength of the signal root upon update sometimes.  
        # rprior = prior
        # if c >= Cs:
        #    rprior = _cd.__COMP_REF__
            
        j = 2*c + 1 #  1, 3, 5, ...
        # given all other roots except the jth.  This is PHI0
        jth_r1 = alpC.pop(j)    #  negative root   #  nothing is done with these
        jth_r2 = alpC.pop(j-1)  #  positive root

        #  exp(-(Y - FX)^2 / 2q^2)
        Frmj = _Npp.polyfromroots(alpR + alpC).real   #    jth removed
        #print "ampAngRep"
        #print ampAngRep(alpC)

        Ff[0, :]   = Frmj[::-1]   

        #  Ff first element k-delay,  Prod{i neq j} (1 - alfa_i B)

        ##########  CREATE FILTERED TIMESERIES   ###########
        ##########  Frmj is k x k, smpxW is (N+2) x k ######

        for m in xrange(TR):
            #  Ff[0]*smpxW[m, 0] + Ff[1]*smpxW[m, 1] + Ff[2]*smpxW[m, 2]
            _N.dot(smpxW[m], Ff.T, out=wjs[m, c])  #  Eq. 20 Huerta West

        alpC.insert(j-1, jth_r1)     #alpC.insert(j - 1, jth_r1)
        alpC.insert(j-1, jth_r2)     #alpC.insert(j - 1, jth_r2)
        

    Ff  = _N.zeros((1, k))
    ######     REAL ROOTS.  Directly sample the conditional posterior
    for j in xrange(R - 1, -1, -1):
        # given all other roots except the jth
        jth_r = alpR.pop(j)

        Frmj = _Npp.polyfromroots(alpR + alpC).real #  Ff first element k-delay
        Ff[0, :] = Frmj[::-1]   #  Prod{i neq j} (1 - alfa_i B)

        ##########  CREATE FILTERED TIMESERIES   ###########
        ##########  Frmj is k x k, smpxU is (N+1) x k ######

        for m in xrange(TR):
            #uj  = _N.dot(Ff, smpxU[m].T).T
            #_N.dot(Ff, smpxU[m].T, out=ujs[m, j])
            _N.dot(smpxU[m], Ff.T, out=ujs[m, j])

        alpR.insert(j, jth_r)

    return ujs, wjs#, lsmpld


def timeseries_decompose(R, C, allalfas, TR, it, N, ignr, rt, zt, uts, wts):
    """
    uts, wts   filtered time series
    rt, zt     decomposed, additive components
    """
    for tr in xrange(TR):
        b, c = dcmpcff(alfa=allalfas[it])

        for r in xrange(R):
            rt[it, tr, :, r] = b[r] * uts[tr, r, :, 0]

        for z in xrange(C):
            #print("%dth comp" % z)
            cf1 = 2*c[2*z].real
            gam = allalfas[it, R+2*z]
            #print(gam)
            cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
            #print("%(1).3f  %(2).3f" % {"1" : cf1, "2" : cf2})
            zt[it, tr, 0:N-ignr-2, z] = \
                cf1*wts[tr, z, 1:N-ignr-1, 0] - \
                cf2*wts[tr, z, 0:N-ignr-2, 0]
