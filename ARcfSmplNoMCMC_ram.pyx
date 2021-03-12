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
from libc.math cimport sqrt
cimport cython


ujs = None
wjs = None
Ff  = None
F0  = None
Xs     = None
Ys     = None
H      = None
iH     = None
mu     = None
J      = None
Ji      = None
Mj     = None
Mji    = None
mj     = None
filtrootsC = None
filtrootsR = None
arInd  = None
ES     = None
U      = None
XsYs   = None

cdef double *p_Xs
cdef double *p_Ys
cdef double*p_H 
cdef double *p_iH
cdef double *p_mu
cdef double *p_J
cdef double *p_Ji
cdef double *p_Mji
cdef double *p_mj
cdef double  *p_ujs
cdef double  *p_wjs
cdef double *p_ES
cdef double *p_U
cdef double *p_Ff
cdef double *p_F0
cdef double *p_XsYs
cdef complex *p_vfiltrootsC
cdef complex *p_vfiltrootsR

@cython.boundscheck(False)
@cython.wraparound(False)
def init(int N, int k, int TR, int R, int Cs, int Cn, aro=_cd.__NF__):
    global ujs, wjs, Ff, F0, A, Xs, Ys, XsYs, H, iH, mu, J, Ji, Mj, Mji, mj, filtrootsC, filtrootsR, arInd, ES, U
    global p_Xs, p_Ys, p_XsYs, p_H, p_iH, p_mu, p_J, p_Ji, p_Mji, p_mj, p_ujs, p_wjs, p_ES, p_U, p_Ff, p_F0
    global p_vfiltrootsC, p_vfiltrootsR
    print("INIT ARcfSmplNoMCMC_ram")
    cdef int C = Cs + Cn

    ujs     = _N.zeros((TR, R, N + 1, 1))
    wjs     = _N.zeros((TR, C, N + 2, 1))
    cdef double[:, :, :, ::1] v_ujs = ujs
    p_ujs = &v_ujs[0, 0, 0, 0]
    cdef double[:, :, :, ::1] v_wjs = wjs
    p_wjs = &v_wjs[0, 0, 0, 0]

    Ff  = _N.zeros((1, k-1))
    F0  = _N.zeros(2)
    cdef double[:, ::1] vFf  = Ff
    p_Ff  = &vFf[0, 0]
    cdef double[::1] vF0  = F0
    p_F0  = &vF0[0]

    Xs     = _N.zeros((N-2, 2))
    Ys     = _N.zeros((N-2, 1))
    XsYs   = _N.zeros((2, 1))
    H      = _N.zeros((TR, 2, 2))
    iH     = _N.zeros((TR, 2, 2))
    mu     = _N.zeros((TR, 2, 1))

    cdef double[:, ::1] vXs = Xs
    p_Xs = &vXs[0, 0]
    cdef double[:, ::1] vYs = Ys
    p_Ys = &vYs[0, 0]
    cdef double[:, ::1] vXsYs = XsYs
    p_XsYs = &vXsYs[0, 0]

    cdef double[:, :, ::1] vH = H
    p_H = &vH[0, 0, 0]
    cdef double[:, :, ::1] viH = iH
    p_iH = &viH[0, 0, 0]
    cdef double[:, :, ::1] vmu = mu
    p_mu = &vmu[0, 0, 0]

    J      = _N.zeros((2, 2))
    Ji      = _N.zeros((2, 2))
    Mji    = _N.zeros(TR)
    mj     = _N.zeros(TR)

    cdef double[:, ::1] vJ = J
    p_J = &vJ[0, 0]
    cdef double[:, ::1] vJi = Ji
    p_Ji = &vJi[0, 0]
    cdef double[::1] vMji = Mji
    p_Mji = &vMji[0]
    cdef double[::1] vmj = mj
    p_mj = &vmj[0]

    filtrootsC = _N.zeros(2*C-2+R, dtype=_N.complex)
    filtrootsR = _N.zeros(2*C+R-1, dtype=_N.complex)
    cdef complex[::1] vfiltrootsC = filtrootsC
    p_vfiltrootsC = &vfiltrootsC[0]
    cdef complex[::1] vfiltrootsR = filtrootsR
    p_vfiltrootsR = &vfiltrootsR[0]


    if aro == _cd.__SF__:    #  signal first
        arInd = _N.arange(C)
    else:                    #  noise first
        arInd = _N.arange(C-1, -1, -1)

    ES = _N.zeros((2, 1))     #  einsum  output is shape (2, 1)
    cdef double[:, ::1]    vES = ES
    p_ES       = &vES[0, 0]
    U  = _N.zeros((2, 1))     #  einsum  output is shape (2, 1)
    cdef double[:, ::1]    vU = U
    p_U       = &vU[0, 0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
#def ARcfSmpl(int N, int k, int TR, double[:, ::1] AR2lims, smpxU, smpxW, double[::1] q2, int R, int Cs, int Cn, complex[::1] valpR, complex[::1] valpC, int sig_ph0L, int sig_ph0H):
def ARcfSmpl(int N, int k, int TR, AR2lims_nmpy, smpxU, smpxW, double[::1] q2, int R, int Cs, int Cn, complex[::1] valpR, complex[::1] valpC, double sig_ph0L, double sig_ph0H, double prR_s2):
    global ujs, wjs, Ff, F0, Xs, Ys, XsYs, H, iH, mu, J, Ji, Mj, Mji, mj, filtrootsC, filtrootsR, arInd
    global p_vfiltrootsC, p_vfiltrootsR

    global p_Xs, p_Ys, p_H, p_iH, p_mu, p_J, p_Ji, p_Mji, p_mj, p_ujs, p_wjs, p_ES, p_U, p_Ff, p_F0

    ##ttt1 = _tm.time()
    cdef int C = Cs + Cn

    #  I return F and Eigvals
    
    cdef double[:, :, ::1] v_smpxU = smpxU
    cdef double[:, :, ::1] v_smpxW = smpxW
    cdef double* p_smpxW = &v_smpxW[0, 0, 0]
    cdef double* p_smpxU = &v_smpxU[0, 0, 0]

    # ES = _N.zeros(2)     #  einsum  output is shape (2, 1)
    # cdef double[::1]    vES = ES
    # cdef double* p_ES       = &vES[0]
    # U  = _N.zeros(2)     #  einsum  output is shape (2, 1)
    # cdef double[::1]    vU = U
    # cdef double* p_U       = &vU[0]

    
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

    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    cdef double* p_q2 = &q2[0]
    cdef double[:, ::1] AR2lims = AR2lims_nmpy
    cdef double* p_AR2lims = &AR2lims[0, 0]

    cdef complex* p_valpC = &valpC[0]
    cdef complex* p_valpR = &valpR[0]


    cdef int c, j, ti, ni, ii, jj, iif, m
    cdef double iH00, iH01, iH10, iH11, div, idiv
    cdef double svPr1, svPr2, vPr1, vPr2
    cdef double JR, JiR, UR, tot, iq2, r1, ph0j1, ph0j2, rj
    cdef double upos, spos
    cdef complex img

    iq2 = 1./p_q2[0]

    #  r = sqrt(-1*phi_1)   0.25 = -1*phi_1   -->  phi_1 >= -0.25   gives r >= 0.5 for signal components   

    #ttt2 = _tm.time()
    #ttt2a = 0
    #ttt2b = 0

    # print("AR2lims    Cs %d" % Cs)
    # print(AR2lims_nmpy)
    # print("sig_ph0L  %(sL).4e   sig_ph0H  %(sH).4e" % {"sL" : sig_ph0L, "sH" : sig_ph0H})

    #print("ARcfSmpNoMCMC_ram  Starting q2:  %.4e" % p_q2[0])
    #for 0 <= c < C:
    for C-1 >= c > -1:
        #if c >= Cs:
        ph0L = -1
        ph0H = 0
        # else:
        #    ph0L = sig_ph0L   # 
        #    ph0H = sig_ph0H #  R=0.97, reasonably oscillatory
            
        j = 2*c + 1
        p1a =  p_AR2lims[2*c]
        p1b =  p_AR2lims[2*c+1]
        
        iif = -1
        for 0 <= ii < C:
            if (ii != c):
                iif += 1
                p_vfiltrootsC[iif] = p_valpC[2*ii]
                iif += 1
                p_vfiltrootsC[iif] = p_valpC[2*ii+1]
        for ii in range(R):
            p_vfiltrootsC[ii+2*C-2] = p_valpR[ii]
        # given all other roots except the jth.  This is PHI0
            
        #     jth_r1 = alpC.pop(j)    #  negative root   #  nothing is done with these
        #     jth_r2 = alpC.pop(j-1)  #  positive root

        #     #  exp(-(Y - FX)^2 / 2q^2)

        #     #Frmj = _Npp.polyfromroots(alpR + alpC).real
        Frmj = _Npp.polyfromroots(filtrootsC).real
        #print("Frmj   ")
        #print(Frmj)

        Ff[0, :]   = Frmj[::-1]
        #for 0 <= ti < k - 1:m
        #    Ff[0, ti]   = Frmj[k-2-ti]#Frmj[::-1]
        #     #  Ff first element k-delay,  Prod{i neq j} (1 - alfa_i B)

        #     ##########  CREATE FILTERED TIMESERIES   ###########
        #     ##########  Frmj is k x k, smpxW is (N+2) x k ######

        p_Ji[0] = 0
        p_Ji[1] = 0
        p_Ji[2] = 0
        p_Ji[3] = 0
        for 0 <= m < TR:
            # print("SHAPES SHAPES SHAPES SHAPES")
            # print(smpxW[m].shape)
            # print(Ff.T.shape)
            # print(wjs[m, c].shape)
            # for 0 <= ti < N:
            #     tot = 0
            #     for 0 <= ki < k-1:
            #         tot += p_Ff[k]*p_smpxW[m*N+ k]
            #     wjs[m, c, ti] = tot
            _N.dot(smpxW[m], Ff.T, out=wjs[m, c])

            # Ys[:]    = wjs[m, c, 2:N]
            # Xs[:, 0] = wjs[m, c, 1:N-1, 0]   # This block
            # Xs[:, 1] = wjs[m, c, 0:N-2, 0]

            #Xs     = _N.zeros((N-2, 2))
            for 0 <= ni < N-2:
                # wjs      = _N.zeros((TR, C, N + 2, 1))
                p_Ys[ni]    = p_wjs[(N+2)*C*m + (N+2)*c + 2+ni]
                p_Xs[2*ni]    = p_wjs[(N+2)*C*m + (N+2)*c + ni+1]
                p_Xs[2*ni+1]    = p_wjs[(N+2)*C*m + (N+2)*c + ni]

            iH00 = 0
            iH01 = 0
            iH11 = 0

            for 0 <= ni < N-2:
                iH00 += p_Xs[2*ni] * p_Xs[2*ni]
                iH01 += p_Xs[2*ni] * p_Xs[2*ni+1]
                iH11 += p_Xs[2*ni+1] * p_Xs[2*ni+1]
            p_iH[m*4] = iH00 * iq2
            p_iH[m*4+1] = iH01 * iq2
            p_iH[m*4+2] = iH01 * iq2
            p_iH[m*4+3] = iH11 * iq2

            p_Ji[0] += p_iH[m*4]
            p_Ji[1] += p_iH[m*4+1]
            p_Ji[2] += p_iH[m*4+2]
            p_Ji[3] += p_iH[m*4+3]
            ####  H[m]        = _N.linalg.inv(iH[m])   #  aka A
            idiv       = 1./(p_iH[4*m]*p_iH[4*m+3]-p_iH[4*m+1]*p_iH[4*m+2])
            p_H[4*m]  =p_iH[4*m+3] * idiv; 
            p_H[4*m+3]=p_iH[4*m] * idiv;
            p_H[4*m+1]=-p_iH[4*m+1] * idiv;
            p_H[4*m+2]=-p_iH[4*m+1] * idiv;

            #Xs     = _N.zeros((N-2, 2))
            #Ys     = _N.zeros((N-2, 1))
            p_XsYs[0] = 0
            p_XsYs[1] = 0
            for 0 <= ti < N-2:
                p_XsYs[0] += p_Xs[2*ti]*p_Ys[ti]
                p_XsYs[1] += p_Xs[2*ti+1]*p_Ys[ti]
            #mu     = _N.zeros((TR, 2, 1))
            p_mu[2*m]     = (p_H[4*m]  *p_XsYs[0] + p_H[4*m+2]*p_XsYs[1])*iq2
            p_mu[2*m+1]   = (p_H[4*m+1]*p_XsYs[0] + p_H[4*m+3]*p_XsYs[1])*iq2
            #mu[m]        = _N.dot(H[m], _N.dot(Xs.T, Ys)) * iq2
            #mu[m]        = _N.dot(H[m], XsYs) * iq2
            
            
        #  

        #iH     = _N.zeros((TR, 2, 2))
        #_N.copyto(Ji, _N.sum(iH, axis=0))
        idiv = 1./(p_Ji[0]*p_Ji[3]-p_Ji[1]*p_Ji[2])
        p_J[0] = p_Ji[3] * idiv
        p_J[3] = p_Ji[0] * idiv
        p_J[1] = -p_Ji[1] * idiv
        p_J[2] = -p_Ji[2] * idiv

        # for 0 <= ii < 2:
        p_ES[0] = 0
        p_ES[1] = 0
        for 0 <= ii < 2:
            for 0 <= m < TR:
                for 0 <= jj < 2:
                    p_ES[ii] += p_iH[4*m+2*ii+jj]*p_mu[2*m+jj]
        #ES  = _N.einsum("tij,tjk->ik", iH, mu)

        #U   = _N.dot(J, _N.einsum("tij,tjk->ik", iH, mu))
        #  dot(2 x 2    2 x 1)
        p_U[0] = p_J[0]*p_ES[0] + p_J[1]*p_ES[1]
        p_U[1] = p_J[2]*p_ES[0] + p_J[3]*p_ES[1]
        #U   = _N.dot(J, ES)


        ##########  Sample *PROPOSED* parameters 

        # #  If posterior valid Gaussian    q2 x H - oNZ * prH00

        ###  This case is still fairly inexpensive.  
        vPr1   = p_J[0] - (p_J[1]*p_J[1]) / p_J[3]
        vPr2   = p_J[3]
        svPr2  = sqrt(vPr2)
        svPr1  = sqrt(vPr1)

        #b2Real = (U[1, 0] + 0.25*U[0, 0]*U[0, 0] > 0)
        ######  Initialize F0

        mu1prp = p_U[1]
        mu0prp = p_U[0]

        #tttB = _tm.time()
        # print("%d  +++++++++++++++++++++++++++++" % c)
        # print "ph0L  %(L).4f  ph0H  %(H).4f   %(u).4f   %(s1).4f   %(s2).4f   idiv  %(id).4e" % {"L" : ph0L, "H" : ph0H, "u" : mu1prp, "s1" : svPr1, "s2" : svPr2, "id" : idiv}
        # #print "vPr1  %(1).4e  vPr2 %(2).4e" % {"1" : vPr1, "2" : vPr2}
        # print(J)
        # print(Ji)
        # print(Xs[::50, 0])
        # print(Xs[::50, 1])
        # print(Ys[::50])

        ph0j2 = _kd.truncnormC(a=ph0L, b=ph0H, u=mu1prp, std=svPr2)
        r1    = sqrt(-1*ph0j2)
        mj0   = mu0prp + (p_J[1] * (ph0j2 - mu1prp)) / p_J[3]
        # print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
        # print(p1a*r1)
        # print(p1b*r1)
        # print(mj0)
        # print(svPr1)

        ph0j1 = _kd.truncnormC(a=p1a*r1, b=p1b*r1, u=mj0, std=svPr1)
        #tttC = _tm.time()
        #p_A[0] = ph0j1; p_A[1] = ph0j2

        #F0[0] = ph0j1; F0[1] = ph0j2

        #  F1 +/- sqrt(F1^2 + 4F1) / 
        #img        = sqrt(-(p_A[0]*p_A[0] + 4*p_A[1]))*1j
        img        = sqrt(-(ph0j1*ph0j1 + 4*ph0j2))*1j
        #vals, vecs = _N.linalg.eig(_N.array([A, [1, 0]]))
            
        #  the positive root comes first
        p_valpC[j-1] = (ph0j1 - img)*0.5
        p_valpC[j]   = (ph0j1 + img)*0.5
    FfR  = _N.zeros((1, k))
    ######     REAL ROOTS.  Directly sample the conditional posterior
    #for j in range(R - 1, -1, -1):
    #print("rj - roots")
    for R-1 >= j > -1:    # range(R - 1, -1, -1):
        # given all other roots except the jth

        for 0 <= ii < C:
            p_vfiltrootsR[2*ii] = p_valpC[2*ii]
            p_vfiltrootsR[2*ii+1] = p_valpC[2*ii+1]
        iif = -1

        for 0 <= ii < R:
            if ii != j:
                iif += 1
                p_vfiltrootsR[iif+2*C] = p_valpR[ii]

        Frmj = _Npp.polyfromroots(filtrootsR).real #  Ff first element k-delay
        FfR[0, :] = Frmj[::-1]   #  Prod{i neq j} (1 - alfa_i B)

        ##########  CREATE FILTERED TIMESERIES   ###########
        ##########  Frmj is k x k, smpxU is (N+1) x k ######

        JiR = 0
        tot = 0
        for 0 <= m < TR:#for m in range(TR):
            #uj  = _N.dot(Ff, smpxU[m].T).T
            #_N.dot(Ff, smpxU[m].T, out=ujs[m, j])
            _N.dot(smpxU[m], FfR.T, out=ujs[m, j])
            #ujs.append(uj)

            ####   Needed for proposal density calculation

            #  Mji  size TR,    ujs size (TR, R, N + 1, 1)

            p_Mji[m] = 0
            p_mj[m]  = 0
            #_N.dot(ujs[m, j, 0:N, 0], ujs[m, j, 0:N, 0]) / q2[m]
            #mj[m] = _N.dot(ujs[m, j, 1:, 0], ujs[m, j, 0:N, 0])// (q2[m]*Mji[m])
            for 0 <= ii < N:
                p_Mji[m] += p_ujs[m*R*(N+1) + j*(N+1) + ii]*\
                            p_ujs[m*R*(N+1) + j*(N+1) + ii]
                p_mj[m]  += p_ujs[m*R*(N+1) + j*(N+1) + ii+1]*\
                            p_ujs[m*R*(N+1) + j*(N+1) + ii]
            p_Mji[m] *= iq2
            p_mj[m]  /= (p_q2[m]*p_Mji[m])

            JiR += p_Mji[m]
            tot += p_Mji[m]*p_mj[m]
        #  truncated Gaussian to [-1, 1]

        # print("R  %d    .................."  % j)
        # print(Frmj)
        # print("UR   %(ur).4e   JR  %(jr).4e" % {"ur" : UR, "jr" : JR})
        JR  = 1 / JiR
        UR  = JR * tot

        #  We sometimes get this.  Repeat execution of mcmcARcontinuous2.py
        #  starting with same seed.  
        #   File "ARcfSmplNoMCMC_ram.pyx", line 435, in ARcfSmplNoMCMC_ram.ARcfSmpl
        #   File "kdist.pyx", line 206, in kdist.truncnormC
        #     a = (a - u) / std
        # ZeroDivisionError: float division


        # 0  +++++++++++++++++++++++++++++
        # ph0L  -1.0000  ph0H  0.0000   -0.0519   0.0010   0.0012   idiv  1.4814e-12
        # [[ 1.35449271e-06 -5.98894384e-07]
        #  [-5.98894384e-07  1.35847269e-06]]
        # [[917040.01093497 404284.98595776]
        #  [404284.98595776 914353.315131  ]]
        # [ 1.03847005  3.89664149  2.20783531  1.11250282  7.42264765  3.7202803
        #  -7.2515943  -2.69113341  1.16868503  0.92892936  2.15100709]
        # [-2.64953923  1.61031552  1.30450315  1.19671467  0.90730243  8.0812135
        #  -1.2202872   1.457269    0.51048828  0.45051488 -2.53918853]
        # [[-0.53240586]
        #  [-3.2872491 ]
        #  [-0.63618095]
        #  [ 2.14652302]
        #  [ 2.67011545]
        #  [-2.3844769 ]
        #  [-3.70115082]
        #  [ 1.20983983]
        #  [-2.92045352]
        #  [ 2.35508279]
        #  [-4.20680826]]
        # Traceback (most recent call last):
        ##   Above traceback identical between run when there is an error and when there isn't.  Xs[::50, 0] and Xs[::50, 1] and Ys[::50]
        #  (0 x 1/prR_s2) + (UR/JR) / (1/prR_s2 + 1/JR)
        #  prR_s2*JR / (prR_s2 + JR)
        upos = (UR/JR) / (1/prR_s2 + 1/JR)
        spos = (prR_s2*JR) / (prR_s2 + JR)
        #rj = _kd.truncnormC(-1., 1., UR, sqrt(JR))
        rj = _kd.truncnormC(-1., 1., upos, sqrt(spos))

        #alpR.insert(j, rj)
        p_valpR[j] = rj
        #print(rj)
    #alpR[:] = valpR
    #alpC[:] = valpC
    #ttt3 = _tm.time()

    # print("--------------------------------")
    # print("t2-t1  %.4f" % (#ttt2-#ttt1))
    # print("t3-t2  %.4f" % (#ttt3-#ttt2))
    # print("#ttt2a  %.4f" % #ttt2a)
    # print("#ttt2b  %.4f" % #ttt2b)


    return ujs, wjs#, lsmpld

