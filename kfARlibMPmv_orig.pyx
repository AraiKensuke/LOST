import numpy as _N
cimport numpy as _N
import kfcomMPmv as _kfcom
import time as _tm
cimport cython

import warnings
warnings.filterwarnings("error")

dDTYPE = _N.double
ctypedef _N.double_t dDTYPE_t

"""
c functions
"""
cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double log(double)
    double abs(double)
"""
p        AR order
Ftrgt    Ftrgt[0]  noise amp.  Ftrgt[1:]  AR(p) coeffs
f        freq.
f0       bandpass
f1
zr       amp. at band stop
"""

########################   FFBS
#def armdl_FFBS_1itrMP(y, Rv, F, q2, N, k, fx00, fV00):   #  approximation
@cython.boundscheck(False)
@cython.wraparound(False)
def armdl_FFBS_1itrMP(double[:, ::1] y, double[:, ::1] Rv, double[:, :, ::1] F, double[::1] q2, long[::1] N, long[::1] ks, double[:, :, ::1] fx, double[:, :, :, ::1] fV, double[:, :, ::1] px, double[:, :, :, ::1] pV, smpx, double[:, :, ::1] K):   #  approximation

def armdl_FFBS_1itrMP(args):   #  approximation
    """
    for Multiprocessor, aguments need to be put into a list.
    """
    y  = args[0]
    Rv = args[1]
    F  = args[2]
    q2 = args[3]
    N  = args[4] 
    cdef _N.intp_t k  = args[5]
    fx00 = args[6]
    fV00 = args[7]

    fx = _N.empty((N + 1, k, 1))
    fV = _N.empty((N + 1, k, k))
    #fx[0, :, 0] = fx00
    fx[0] = fx00
    fV[0] = fV00
    
    GQGT   = _N.zeros((k, k))
    GQGT[0, 0] = q2

    ##########  FF
    #t1 = _tm.time()
    FFdv(y, Rv, N, k, F, GQGT, fx, fV)
    #t2 = _tm.time()
    #print "FFdv  %.3f" % (t2-t1)
    ##########  BS
    smXN = _N.random.multivariate_normal(fx[N,:,0], fV[N], size=1)
    #t1 = _tm.time()
    #smpls = _kfcom.BSvecChol(F, N, k, GQGT, fx, fV, smXN)
    #smpls = _kfcom.BSvecSVD_new(F, N, k, GQGT, fx, fV, smXN)
    smpls = _kfcom.BSvec(F, N, k, GQGT, fx, fV, smXN)
    #t2 = _tm.time()
    #print (t2-t1)
    return [smpls, fx, fV]

def FFdv(y, Rv, N, k, F, GQGT, fx, fV):   #  approximate KF    #  k==1,dynamic variance
    #print "FFdv"
    #  do this until p_V has settled into stable values
    H       = _N.zeros((1, k))          #  row vector
    H[0, 0] = 1
    cdef double q2 = GQGT[0, 0]

    Ik      = _N.identity(k)
    px = _N.empty((N + 1, k, 1))
    pV = _N.empty((N + 1, k, k))

    K     = _N.empty((N + 1, k, 1))
    """
    temporary storage
    """
    IKH   = _N.eye(k)        #  only contents of first column modified
    VFT   = _N.empty((k, k))
    FVFT  = _N.empty((k, k))
    KyHpx = _N.empty((k, 1))

    #  need memory views for these
    #  F, fx, px need memory views
    #  K, KH
    #  IKH
    
    cdef double[:, ::1] Fmv       = F
    cdef double[:, :, ::1] fxmv   = fx
    cdef double[:, :, ::1] pxmv   = px
    cdef double[:, :, ::1] pVmv   = pV
    cdef double[::1] Rvmv   = Rv
    cdef double[:, :, ::1] Kmv    = K
    cdef double[:, ::1] IKHmv     = IKH

    cdef _N.intp_t n, i

    cdef double dd = 0
    for n from 1 <= n < N + 1:
        dd = 0
        for i in xrange(1, k):#  use same loop to copy and do dot product
            dd += Fmv[0, i]*fxmv[n-1, i, 0]
            pxmv[n, i, 0] = fxmv[n-1, i-1, 0]
        pxmv[n, 0, 0] = dd + Fmv[0, 0]*fxmv[n-1, 0, 0]

        _N.dot(fV[n - 1], F.T, out=VFT)
        _N.dot(F, VFT, out=pV[n])          #  prediction
        pVmv[n, 0, 0]    += q2
        mat  = 1 / (pVmv[n, 0, 0] + Rvmv[n])  #  scalar

        K[n, :, 0] = pV[n, :, 0] * mat

        _N.multiply(K[n], y[n] - pxmv[n, 0, 0], out=KyHpx)
        _N.add(px[n], KyHpx, out=fx[n])

        # (I - KH), KH is zeros except first column
        IKHmv[0, 0] = 1 - Kmv[n, 0, 0]
        for i in xrange(1, k):
            IKHmv[i, 0] = -Kmv[n, i, 0]
        # (I - KH)
        _N.dot(IKH, pV[n], out=fV[n])

def FF1dv(_d, offset=0):   #  approximate KF    #  k==1,dynamic variance
    GQGT    = _d.G[0,0]*_d.G[0, 0] * _d.Q
    k     = _d.k
    px    = _d.p_x
    pV    = _d.p_V
    fx    = _d.f_x
    fV    = _d.f_V
    Rv    = _d.Rv
    K     = _d.K

    #  do this until p_V has settled into stable values

    for n from 1 <= n < _d.N + 1:
        px[n,0,0] = _d.F[0,0] * fx[n - 1,0,0]
#        pV[n,0,0] = _d.F[0,0] * fV[n - 1,0,0] * _d.F.T[0,0] + GQGT
        pV[n,0,0] = _d.F[0,0] * fV[n - 1,0,0] * _d.F[0,0] + GQGT
        #_d.p_Vi[n,0,0] = 1/pV[n,0,0]

#        mat  = 1 / (_d.H[0,0]*pV[n,0,0]*_d.H[0,0] + Rv[n])
        mat  = 1 / (pV[n,0,0] + Rv[n])
#        K[n,0,0] = pV[n]*_d.H[0,0]*mat
        K[n,0,0] = pV[n,0,0]*mat
#        fx[n,0,0]    = px[n,0,0] + K[n,0,0]*(_d.y[n] - offset[n] - _d.H[0,0]* px[n,0,0])
#        fx[n,0,0]    = px[n,0,0] + K[n,0,0]*(_d.y[n] - _d.H[0,0]* px[n,0,0])
        fx[n,0,0]    = px[n,0,0] + K[n,0,0]*(_d.y[n] - px[n,0,0])
#        fV[n,0,0] = (1 - K[n,0,0]* _d.H[0,0])* pV[n,0,0]
fV[n,0,0] = (1 - K[n,0,0])* pV[n,0,0]
