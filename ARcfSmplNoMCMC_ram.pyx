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

ujs = None
wjs = None
Ff  = None
F0  = None
F1  = None
A   = None
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

def init(int N, int k, int TR, int R, int Cs, int Cn, aro=_cd.__NF__):
    cdef int C = Cs + Cn
    global ujs, wjs, Ff, F0, F1, A, Xs, Ys, H, iH, mu, J, Ji, Mj, Mji, mj, filtrootsC, filtrootsR, arInd
    print("INIT ARcfSmplNoMCMC_ram")
    ujs     = _N.empty((TR, R, N + 1, 1))
    wjs     = _N.empty((TR, C, N + 2, 1))
    Ff  = _N.zeros((1, k-1))
    F0  = _N.zeros(2)
    F1  = _N.zeros(2)
    A   = _N.empty(2)
    Xs     = _N.empty((N-2, 2))
    Ys     = _N.empty((N-2, 1))
    H      = _N.empty((TR, 2, 2))
    iH     = _N.empty((TR, 2, 2))
    mu     = _N.empty((TR, 2, 1))
    J      = _N.empty((2, 2))
    Ji      = _N.empty((2, 2))
    Mj     = _N.empty(TR)
    Mji    = _N.empty(TR)
    mj     = _N.empty(TR)
    filtrootsC = _N.empty(2*C-2+R, dtype=_N.complex)
    filtrootsR = _N.empty(2*C+R-1, dtype=_N.complex)

    if aro == _cd.__SF__:    #  signal first
        arInd = _N.arange(C)
    else:                    #  noise first
        arInd = _N.arange(C-1, -1, -1)


def testtest():
    global Ji

    cdef double[:, ::1] vJi = Ji
    cdef double* p_Ji = &vJi[0, 0]

    hhh = _N.random.randn(2, 2, 2)

    _N.copyto(Ji, _N.sum(hhh, axis=0))

    print(Ji)
    div = (p_Ji[0]*p_Ji[3]-p_Ji[1]*p_Ji[2])
    #J   = _N.linalg.inv(Ji)
    print("the divs")
    print(div)
    print((Ji[0,0]*Ji[1,1]-Ji[0,1]*Ji[1,0]))

    
def ARcfSmpl(int N, int k, int TR, double[:, ::1] AR2lims, smpxU, smpxW, double[::1] q2, int R, int Cs, int Cn, complex[::1] valpR, complex[::1] valpC, int sig_ph0L, int sig_ph0H):
    global ujs, wjs, Ff, F0, F1, A, Xs, Ys, H, iH, mu, J, Ji, Mj, Mji, mj, filtrootsC, filtrootsR, arInd

    ##ttt1 = _tm.time()
    cdef int C = Cs + Cn

    #  I return F and Eigvals

    cdef double[:, :, ::1] v_smpxW = smpxW
    cdef double* p_smpxW = &v_smpxW[0, 0, 0]
    cdef double[:, :, ::1] v_smpxU = smpxU
    cdef double* p_smpxU = &v_smpxU[0, 0, 0]

    cdef double[:, :, :, ::1] v_ujs = ujs
    cdef double* p_ujs = &v_ujs[0, 0, 0, 0]
    cdef double[:, :, :, ::1] v_wjs = wjs
    cdef double* p_wjs = &v_wjs[0, 0, 0, 0]

    # ES = _N.zeros(2)     #  einsum  output is shape (2, 1)
    # cdef double[::1]    vES = ES
    # cdef double* p_ES       = &vES[0]
    # U  = _N.zeros(2)     #  einsum  output is shape (2, 1)
    # cdef double[::1]    vU = U
    # cdef double* p_U       = &vU[0]
    ES = _N.zeros((2, 1))     #  einsum  output is shape (2, 1)
    cdef double[:, ::1]    vES = ES
    cdef double* p_ES       = &vES[0, 0]
    U  = _N.zeros((2, 1))     #  einsum  output is shape (2, 1)
    cdef double[:, ::1]    vU = U
    cdef double* p_U       = &vU[0, 0]

    
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
    cdef double[:, ::1] vFf  = Ff
    cdef double* p_vFf  = &vFf[0, 0]
    cdef double[::1] vF0  = F0
    cdef double* p_vF0  = &vF0[0]

    cdef double[::1] vF1  = F1
    cdef double* p_vF1  = &vF1[0]

    cdef double[::1] vA  = A
    cdef double* p_vA  = &vA[0]

    cdef double[:, ::1] vXs = Xs
    cdef double* p_Xs = &vXs[0, 0]
    cdef double[:, ::1] vYs = Ys
    cdef double* p_Ys = &vYs[0, 0]
    cdef double[:, :, ::1] vH = H
    cdef double* p_H = &vH[0, 0, 0]
    cdef double[:, :, ::1] viH = iH
    cdef double* p_iH = &viH[0, 0, 0]
    cdef double[:, :, ::1] vmu = mu
    cdef double* p_mu = &vmu[0, 0, 0]
    cdef double[:, ::1] vJ = J
    cdef double* p_J = &vJ[0, 0]
    cdef double[:, ::1] vJi = Ji
    cdef double* p_Ji = &vJi[0, 0]
    cdef double[::1] vMj = Mj
    cdef double* p_Mj = &vMj[0]
    cdef double[::1] vMji = Mji
    cdef double* p_Mji = &vMji[0]
    cdef double[::1] vmj = mj
    cdef double* p_mj = &vmj[0]
    cdef double* p_q2 = &q2[0]
    cdef double* p_AR2lims = &AR2lims[0, 0]

    cdef complex* p_valpC = &valpC[0]
    cdef complex* p_valpR = &valpR[0]

    cdef complex[::1] vfiltrootsC = filtrootsC
    cdef complex* p_vfiltrootsC = &vfiltrootsC[0]
    cdef complex[::1] vfiltrootsR = filtrootsR
    cdef complex* p_vfiltrootsR = &vfiltrootsR[0]

    cdef int c, j, ti, ni, ii, jj, iif, m
    cdef double iH00, iH01, iH10, iH11, div, idiv
    cdef double svPr1, svPr2, vPr1, vPr2
    cdef double JR, JiR, UR, tot, iq2, r1, ph0j1, ph0j2, rj
    cdef complex img

    iq2 = 1./p_q2[0]

    #  r = sqrt(-1*phi_1)   0.25 = -1*phi_1   -->  phi_1 >= -0.25   gives r >= 0.5 for signal components   

    #ttt2 = _tm.time()
    #ttt2a = 0
    #ttt2b = 0

    #for 0 <= c < C:
    for C-1 >= c > -1:
        if c >= Cs:
            ph0L = -1
            ph0H = 0
        else:
            ph0L = sig_ph0L   # 
            ph0H = sig_ph0H #  R=0.97, reasonably oscillatory
            
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
        

        Ff[0, :]   = Frmj[::-1]
        #for 0 <= ti < k - 1:m
        #    Ff[0, ti]   = Frmj[k-2-ti]#Frmj[::-1]
        #     #  Ff first element k-delay,  Prod{i neq j} (1 - alfa_i B)

        #     ##########  CREATE FILTERED TIMESERIES   ###########
        #     ##########  Frmj is k x k, smpxW is (N+2) x k ######

        for 0 <= m < TR:
            # print("SHAPES SHAPES SHAPES SHAPES")
            # print(smpxW[m].shape)
            # print(Ff.T.shape)
            # print(wjs[m, c].shape)
            # for 0 <= ti < N:
            #     tot = 0
            #     for 0 <= ki < k:
            #         tot += p_Ff[k]*p_smpxW[m*N+ k]
            #     wjs[m, c, ti] = tot
            _N.dot(smpxW[m], Ff.T, out=wjs[m, c])

            # Ys[:]    = wjs[m, c, 2:N]
            # Xs[:, 0] = wjs[m, c, 1:N-1, 0]   # This block
            # Xs[:, 1] = wjs[m, c, 0:N-2, 0]

            for 0 <= ni < N-2:
                # wjs 
                p_Ys[ni]    = p_wjs[(N+2)*C*m + (N+2)*c + 2+ni]
                p_Xs[2*ni]    = p_wjs[(N+2)*C*m + (N+2)*c + ni+1]
                p_Xs[2*ni+1]    = p_wjs[(N+2)*C*m + (N+2)*c + ni]
            # print("compare 2222")
            # print(Ys[0:10])
            # print(Xs[0:10, 0])
            # print(Xs[0:10, 1])

            #iH[m]       = _N.dot(Xs.T, Xs) / q2[m]
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

            ####  H[m]        = _N.linalg.inv(iH[m])   #  aka A
            idiv       = 1./(p_iH[4*m]*p_iH[4*m+3]-p_iH[4*m+1]*p_iH[4*m+2])
            p_H[4*m]  =p_iH[4*m+3] * idiv; 
            p_H[4*m+3]=p_iH[4*m] * idiv;
            p_H[4*m+1]=-p_iH[4*m+1] * idiv;
            p_H[4*m+2]=-p_iH[4*m+1] * idiv;
            mu[m]        = _N.dot(H[m], _N.dot(Xs.T, Ys)) * iq2

        #  
        _N.copyto(Ji, _N.sum(iH, axis=0))
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

        #bBdd, iBdd, mags, vals = _arl.ARevals(U[:, 0])
        #print "U"
        #print U[:, 0]
        #print ":::::: *****"
        #print ampAngRep(vals)

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
        #print "ph0L  %(L).4f  ph1L  %(H).4f   %(u).4f   %(s).4f" % {"L" : ph0L, "H" : ph0H, "u" : mu1prp, "s" : svPr2}
        #print "vPr1  %(1).4e  vPr2 %(2).4e" % {"1" : vPr1, "2" : vPr2}
        # print("+++++++++++++++++++++++++++++")
        # print(ph0L)
        # print(ph0H)
        # print(mu1prp)
        # print(svPr2)
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
        #alpC.insert(j-1, (A[0] - img)*0.5)     #alpC.insert(j - 1, jth_r1)
        #alpC.insert(j-1, (A[0] + img)*0.5)     #alpC.insert(j - 1, jth_r2)
        #tttC = _tm.time()
        #ttt2a += #tttB - #tttA
        #ttt2b += #tttC - #tttB

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print(alpC)
    # print(valpC)
    FfR  = _N.zeros((1, k))
    ######     REAL ROOTS.  Directly sample the conditional posterior
    #for j in range(R - 1, -1, -1):
    for R-1 >= j > -1:    # range(R - 1, -1, -1):
        # given all other roots except the jth
        #jth_r = alpR.pop(j)

        #for ii in range(C):
        for 0 <= ii < C:
            p_vfiltrootsR[2*ii] = p_valpC[2*ii]
            p_vfiltrootsR[2*ii+1] = p_valpC[2*ii+1]
        iif = -1
        #for ii in range(R):
        for 0 <= ii < R:
            if ii != j:
                iif += 1
                p_vfiltrootsR[iif+2*C] = p_valpR[ii]

        #print("R !!!!")
        #print(alpR + alpC)
        #print(filtrootsR)
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

        JR  = 1 / JiR
        UR  = JR * tot

        #  we only want 
        rj = _kd.truncnormC(-1., 1., UR, sqrt(JR))

        #alpR.insert(j, rj)
        p_valpR[j] = rj
    #alpR[:] = valpR
    #alpC[:] = valpC
    #ttt3 = _tm.time()

    # print("--------------------------------")
    # print("t2-t1  %.4f" % (#ttt2-#ttt1))
    # print("t3-t2  %.4f" % (#ttt3-#ttt2))
    # print("#ttt2a  %.4f" % #ttt2a)
    # print("#ttt2b  %.4f" % #ttt2b)


    #return ujs, wjs#, lsmpld

