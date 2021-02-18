import numpy as _N
cimport numpy as _N
from cython.parallel import prange
#import kfcomMPmv_ram as _kfcom
#import ram as _kfcom
#import kfcomMPmv_ram as _kfcom
#import kfcomMPmv as _kfcom_slow
import time as _tm
cimport cython
cimport openmp
from libc.math cimport sqrt
from libc.stdio cimport printf

#cython: boundscheck=False, wraparound=False, nonecheck=False

"""
p        AR order
Ftrgt    Ftrgt[0]  noise amp.  Ftrgt[1:]  AR(p) coeffs
f        freq.
f0       bandpass
f1
zr       amp. at band stop
"""

cdef long g__N
cdef long g_Np1
cdef long g_k
cdef long g_kk
cdef long g_TR

def init(long N, long k, long TR):
    global g__N, g_Np1, g_k, g_TR, g_kk
    g_TR = TR
    g__N  =  N
    g_Np1= N+1
    g_k  = k
    g_kk = k*k

########################   FFBS
#def armdl_FFBS_1itrMP(y, Rv, F, q2, N, k, fx00, fV00):   #  approximation
@cython.boundscheck(False)
@cython.wraparound(False)
def armdl_FFBS_1itrMP(double[:, ::1] gau_obs, double[:, ::1] gau_var, double[:, :, ::1] F, double[:, :, ::1] iF, double[::1] q2, long[::1] Ns, long[::1] ks, double[:, :, ::1] fx, double[:, :, :, ::1] fV, double[:, :, ::1] px, double[:, :, :, ::1] pV, smpx, double[:, :, ::1] K):   #  approximation
    global g__N, g_Np1, g_k, g_kk, g_TR
    #ttt1 = _tm.time()
    cdef double* p_gau_obs  = &gau_obs[0, 0]
    cdef double* p_gau_var  = &gau_var[0, 0]
    cdef double* p_F        = &F[0, 0, 0]
    cdef double* p_iF        = &iF[0, 0, 0]
    cdef double* p_q2        = &q2[0]
    #cdef double* p_fx        = &fx[0, 0, 0, 0]
    cdef double* p_fx        = &fx[0, 0, 0]
    cdef double* p_fV        = &fV[0, 0, 0, 0]
    #cdef double* p_px        = &fx[0, 0, 0, 0]
    cdef double* p_px        = &px[0, 0, 0]
    cdef double* p_pV        = &pV[0, 0, 0, 0]
    cdef double* p_K        = &K[0, 0, 0]
    cdef double[:, :, ::1] smpx_mv = smpx
    cdef double* p_smpx            = &smpx_mv[0, 0, 0]
    cdef long __N = g__N
    cdef long _Np1 = g_Np1
    cdef long _k = g_k
    cdef long _kk = g_kk
    cdef long _TR = g_TR

    sx_nz_vars  = _N.empty((_TR, _Np1))
    sx_norms = _N.random.randn(_TR, _Np1)

    cdef int nthreads

    cdef double[:, ::1] sx_nz_vars_mv = sx_nz_vars
    cdef double[:, ::1] sx_norms_mv = sx_norms
    cdef double* p_sx_nz_vars = &sx_nz_vars_mv[0, 0]
    cdef double* p_sx_norms   = &sx_norms_mv[0, 0]

    #  fx   TR x N x k
    #  fV   TR x N x k x k

    cdef long tr, i


    #openmp.omp_set_num_threads(2)
    #cdef int nthreads = openmp.omp_get_num_threads()
    #cdef int nthread_lim = openmp.omp_get_num_procs()
    #print("nthreads %d    threadlim %(tl)d\n" % {"nt" : nthreads, "tl" : nthread_lim})

    #ttt2 = _tm.time()
    for tr in prange(_TR, nogil=True):
        FFdv_new(__N, _Np1, _k, _kk, _TR, &p_gau_obs[tr*_Np1], &p_gau_var[tr*_Np1], &p_F[tr*_k*_k], p_q2[tr], &p_fx[tr*_Np1*_k], &p_fV[tr*_Np1*_k*_k], &p_px[tr*_Np1*_k], &p_pV[tr*_Np1*_k*_k], &p_K[tr*_Np1*_k])
            # ##########  BS
    #ttt3 = _tm.time()
    ifV    = _N.linalg.inv(fV)    #  This is the bottle neck.  OMPing other loops won't help this function.  TODO:  use lapack functions.  cimported, so should be compatible with OMP?
    cdef double[:, :, :, ::1] ifV_mv = ifV
    cdef double* p_ifV            = &ifV_mv[0, 0, 0, 0]

    #ttt4 = _tm.time()
    ucmvnrms = _N.random.randn(_TR, _k)

    C       = _N.linalg.cholesky(fV[:, __N])

    #ttt5 = _tm.time()
    #  smpx   is TR x (N+1)+2 x k.   we sample smpx[:, 2:] and fill the 0,1st with what whas in 3rd time bin.
    smXN       = _N.einsum("njk,nk->nj", C, ucmvnrms) + fx[:, _Np1]

    #smpx[:, _Np1+1] = smXN   #  not as a memview
    smpx[:, __N] = smXN   #  not as a memview
    #ttt6 = _tm.time()

    for tr in prange(_TR, nogil=True):
        BSvec(__N, _Np1, _k, _kk, _TR, &p_iF[tr*_k*_k], &p_ifV[tr*_Np1*_k*_k], p_q2[tr], &p_fx[tr*_Np1*_k], &p_fV[tr*_Np1*_k*_k], &p_smpx[tr*_Np1*_k], &p_sx_nz_vars[tr*_Np1], &p_sx_norms[tr*_Np1])

    #ttt7 = _tm.time()

    # print "t2-t1   %.3e" % (#ttt2-#ttt1)
    # print "t3-t2   %.3e" % (#ttt3-#ttt2)
    # print "t4-t3   %.3e" % (#ttt4-#ttt3)
    # print "t5-t4   %.3e" % (#ttt5-#ttt4)
    # print "t6-t5   %.3e" % (#ttt6-#ttt5)
    # print "t7-t6   %.3e" % (#ttt7-#ttt6)

    #smpx[:, 1, 0:_k-1]   = smpx[:, 2, 1:]
    #smpx[:, 0, 0:_k-2]   = smpx[:, 2, 2:]

    # return [smpls, fx, fV]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void FFdv_new(long __N, long _Np1, long _k, long _kk, long _TR, double* p_gau_obs, double* p_gau_var, double* p_F, double q2, double* p_fx, double* p_fV, double* p_px, double* p_pV, double* p_K) nogil:   #  approximate KF    #  k==1,dynamic variance
    #global __N, _k, _kk, _Np1
    #  do this until p_V has settled into stable values

    cdef long n, i, j, ii, jj, nKK, nK, ik, n_m1_KK, n_m1_K, i_m1_K, iik

    cdef double dd = 0, val, Kfac, pKnKi

    for n from 1 <= n < _Np1:
        nKK = n * _kk
        nK  = n*_k
        n_m1_KK = (n-1) * _kk
        n_m1_K = (n-1) * _k
        dd = 0
        #  prediction mean  (naive and analytic method are the same)
        for i in xrange(1, _k):#  use same loop to copy and do dot product
            dd             += p_F[i]*p_fx[n_m1_K + i]
            p_px[nK + i] = p_fx[n_m1_K + (i-1)] # shift older state
        p_px[nK]          = dd + p_F[0]*p_fx[n_m1_K]  #  1-step prediction 


        #####  covariance, 1-step prediction
        ####  upper 1x1
        val = 0
        for ii in xrange(_k):   
            iik = ii*_k
            val += p_F[ii]*p_F[ii]*p_fV[n_m1_KK + iik + ii]
            for jj in xrange(ii+1, _k):
                val += 2*p_F[ii]*p_F[jj]*p_fV[n_m1_KK + iik+jj]
        p_pV[nKK]  = val + q2


        ####  lower k-1 x k-1
        for ii in xrange(1, _k):
            for jj in xrange(ii, _k):
                p_pV[nKK+ ii*_k+ jj] = p_pV[nKK+ jj*_k+ ii] = p_fV[n_m1_KK + (ii-1)*_k + jj-1]
        ####  (1 x k-1) and (k-1 x 1)
        #for ii in xrange(1, k):    #  get rid of 1 loop
            val = 0
            for jj in xrange(_k):
                val += p_F[jj]*p_fV[n_m1_KK+ jj*_k + ii-1]
            p_pV[nKK + ii] = val
            p_pV[nKK + ii*_k] = val


        ######  Kalman gain
        Kfac  = 1. / (p_pV[nKK] + p_gau_var[n])  #  scalar
        for i in xrange(_k):
            #p_K[nK + i] = p_pV[nKK + i*k] * Kfac
            pKnKi = p_pV[nKK + i*_k] * Kfac

            p_fx[nK+i] = p_px[nK+ i] + pKnKi*(p_gau_obs[n] - p_px[nK])

            for j in xrange(i, _k):
                p_fV[nKK+i*_k+ j] = p_pV[nKK+ i*_k+ j] - p_pV[nKK+j]*pKnKi
                p_fV[nKK+j*_k + i] = p_fV[nKK+i*_k+ j]
            p_K[nK+i] = pKnKi
            

###  Most expensive operation here is the SVD
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void BSvec(long __N, long _Np1, long _k, long _kk, long l_TR, double* p_iF, double *p_ifV, double q2, double* p_fx, double* p_fV, double* p_smpx, double* p_sx_nz_vars, double* p_sx_norms) nogil:
    #  Backward sampling.
    #
    #  1)  find covs - only requires values calculated from filtering step
    #  2)  Only p,p-th element in cov mat is != 0.
    #  3)  genearte 0-mean normals from variances computed in 2)
    #  4)  calculate means, add to 0-mean norms (backwds samp) to p-th component
    # 
    #global __N, _k, _Np1
    cdef int n, i, j, ii, jj, nKK, nK, ik, n_m1_KK, i_m1_K, iik, kmk, km1, kp1, np1k, kk
    cdef double trm1, trm2, trm3, c, Fs

    kmk = (_k-1)*_k
    km1 = _k-1
    kp1 = _k+1
    kk  = _k*_k

    ####   ANALYTICAL.  
    cdef double iF_p1_2     = p_iF[kmk]*p_iF[kmk]

    for j in xrange(__N):
        p_sx_nz_vars[j] = sqrt((q2*iF_p1_2)/(1+q2*p_ifV[j*kk + kmk + km1]*iF_p1_2))*p_sx_norms[j]

    for n from __N > n >= 0:
        nKK = n*kk
        nK  = n*_k
        np1k = (n+1)*_k

        c = 1 + q2*p_ifV[nKK + kmk + km1]*iF_p1_2

        Fs = 0
        trm2 = 0
        trm3 = 0

        for ik in xrange(km1):  #  shift
            p_smpx[nK + ik] = p_smpx[np1k + ik+1]
            trm2 += p_smpx[np1k + ik+1]*p_ifV[nKK + kmk + ik]
            Fs += p_iF[kmk + ik]*p_smpx[np1k+ ik]
            trm3 += p_fx[nK + ik]*p_ifV[nKK + kmk + ik]
        Fs += p_iF[kmk + km1]*p_smpx[np1k+ km1]
        trm3 += p_fx[nK + km1]*p_ifV[nKK + kmk + km1]

        trm1 = Fs*p_ifV[nKK + kmk+ km1]

        p_smpx[nK + km1]= Fs - q2*iF_p1_2*(trm1 + trm2 - trm3)/c + p_sx_nz_vars[n]
