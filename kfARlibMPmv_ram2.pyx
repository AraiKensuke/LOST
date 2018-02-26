import numpy as _N
cimport numpy as _N
#import kfcomMPmv_ram as _kfcom
#import ram as _kfcom
import kfcomMPmv_ram as _kfcom
#import kfcomMPmv as _kfcom_slow
import time as _tm
cimport cython

import warnings
warnings.filterwarnings("error")

dDTYPE = _N.double
ctypedef _N.double_t dDTYPE_t

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
def armdl_FFBS_1itrMP(args):   #  approximation
    """
    for Multiprocessor, aguments need to be put into a list.
    """
    y  = args[0]
    Rv = args[1]
    F  = args[2]
    iF  = args[3]
    q2 = args[4]
    N  = args[5] 
    cdef int k  = args[6]
    fx00 = args[7]
    fV00 = args[8]

    fx = _N.empty((N + 1, k, 1))
    fV = _N.empty((N + 1, k, k))
    fx[0] = fx00
    fV[0] = fV00
    GQGT   = _N.zeros((k, k))
    GQGT[0, 0] = q2


    ##########  FF
    #t1 = _tm.time()
    FFdv(y, Rv, N, k, F, q2, fx, fV)
    #t2 = _tm.time()
    #FFdv_slow(y, Rv, N, k, F, GQGT, fx, fV)
    #t3 = _tm.time()

    # print "FFdv"
    # print (t2-t1)
    # print "FFdv-slow"
    # print (t3-t2)

    ##########  BS
    smXN = _N.random.multivariate_normal(fx[N,:,0], fV[N], size=1)
    smpls = _kfcom.BSvec(iF, N, k, q2, fx, fV, smXN)
    #_kfcom_slow.BSvec(F, N, k, GQGT, fx, fV, smXN)

    return [smpls, fx, fV]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def FFdv(double[::1] y, double[::1] Rv, N, long k, double[:, ::1] F, double q2, _fx, _fV):   #  approximate KF    #  k==1,dynamic variance
    #  do this until p_V has settled into stable values

    px = _N.empty((N + 1, k, 1))    #  naive and analytic calculated same way
    pV = _N.empty((N + 1, k, k))

    cdef double[:, :, ::1] fx = _fx
    cdef double[:, :, ::1] fV = _fV
    cdef double* p_y  = &y[0]
    cdef double* p_Rv  = &Rv[0]
    K     = _N.empty((N + 1, k, 1))
    cdef double[:, :, ::1] Kmv   = K  # forward filter
    cdef double* p_K              = &Kmv[0, 0, 0]

    #  need memory views for these
    #  F, fx, px need memory views
    #  K, KH
    #  IKH
    
    cdef double* p_F              = &F[0, 0]
    cdef double* p_fx              = &fx[0, 0, 0]
    cdef double* p_fV              = &fV[0, 0, 0]

    cdef double[:, :, ::1] pxmv   = px
    cdef double* p_px             = &pxmv[0, 0, 0]
    cdef double[:, :, ::1] pVmv   = pV
    cdef double* p_pV             = &pVmv[0, 0, 0]
    cdef int n, i, j, ii, jj, nKK, nK, ik, n_m1_KK, n_m1_K, i_m1_K, iik

    cdef double dd = 0, val, Kfac, pKnKi

    for n from 1 <= n < N + 1:
        nKK = n * k * k
        nK  = n*k
        n_m1_KK = (n-1) * k * k
        n_m1_K = (n-1) * k
        dd = 0
        #  prediction mean  (naive and analytic method are the same)
        for i in xrange(1, k):#  use same loop to copy and do dot product
            dd             += p_F[i]*p_fx[n_m1_K + i]
            p_px[nK + i] = p_fx[n_m1_K + (i-1)] # shift older state
        p_px[nK]          = dd + p_F[0]*p_fx[n_m1_K]  #  1-step prediction 


        #####  covariance, 1-step prediction
        ####  upper 1x1
        val = 0
        for ii in xrange(k):   
            iik = ii*k
            val += p_F[ii]*p_F[ii]*p_fV[n_m1_KK + iik + ii]
            for jj in xrange(ii+1, k):
                val += 2*p_F[ii]*p_F[jj]*p_fV[n_m1_KK + iik+jj]
        p_pV[nKK]  = val + q2
        ####  lower k-1 x k-1
        for ii in xrange(1, k):
            for jj in xrange(ii, k):
                p_pV[nKK+ ii*k+ jj] = p_pV[nKK+ jj*k+ ii] = p_fV[n_m1_KK + (ii-1)*k + jj-1]
        ####  (1 x k-1) and (k-1 x 1)
        #for ii in xrange(1, k):    #  get rid of 1 loop
            val = 0
            for jj in xrange(k):
                val += p_F[jj]*p_fV[n_m1_KK+ jj*k + ii-1]
            p_pV[nKK + ii] = val
            p_pV[nKK + ii*k] = val
        ######  Kalman gain
        Kfac  = 1. / (p_pV[nKK] + p_Rv[n])  #  scalar
        for i in xrange(k):
            #p_K[nK + i] = p_pV[nKK + i*k] * Kfac
            pKnKi = p_pV[nKK + i*k] * Kfac

            p_fx[nK+i] = p_px[nK+ i] + pKnKi*(p_y[n] - p_px[nK])

            for j in xrange(i, k):
                p_fV[nKK+i*k+ j] = p_pV[nKK+ i*k+ j] - p_pV[nKK+j]*pKnKi
                p_fV[nKK+j*k + i] = p_fV[nKK+i*k+ j]
            p_K[nK+i] = pKnKi
            
    
    dat = _N.empty((N+1, 2))
    dat[:, 0] = fx[:, 0, 0]
    dat[:, 1] = fV[:, 0, 0]

    #_N.savetxt("fxfV-fast", dat, fmt="%.4f")
    



def FFdv_slow(y, Rv, N, k, F, GQGT, fx, fV):   #  approximate KF    #  k==1,dynamic variance
    #print "FFdv"
    #  do this until p_V has settled into stable values
    H       = _N.zeros((1, k))          #  row vector
    H[0, 0] = 1
    q2 = GQGT[0, 0]

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

    dat = _N.empty((N+1, 2))
    dat[:, 0] = fx[:, 0, 0]
    dat[:, 1] = fV[:, 0, 0]

    _N.savetxt("fxfV-slow", dat, fmt="%.4f")


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


