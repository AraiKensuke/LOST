import numpy as _N
cimport numpy as _N
from cython.parallel import parallel, prange
#import kfcomMPmv_ram as _kfcom
#import ram as _kfcom
#import kfcomMPmv_ram as _kfcom
#import kfcomMPmv as _kfcom_slow
import time as _tm
cimport cython
from libc.math cimport sqrt
from libc.stdio cimport printf
#import LOST.inv_cov as _invc
#cimport LOST.inv_cov as _invc
cimport scipy.linalg.cython_lapack as _cll
from libc.stdlib cimport malloc, free

"""
p        AR order
Ftrgt    Ftrgt[0]  noise amp.  Ftrgt[1:]  AR(p) coeffs
f        freq.
f0       bandpass
f1
zr       amp. at band stop
"""

cdef long __N
cdef long _Np1
cdef long _k
cdef long _kk
cdef long _TR
kinfo = None
cdef int* p_kinfo


def init(long N, long k, long TR):
    #  fx size   oo.N+nfo1
    #  called like _kfar.init(oo.N, oo.k, oo.TR)
    #  (oo.N + 1) + 2    smpx x is this size
    global __N, _Np1, _k, _TR, _kk, kinfo, p_kinfo
    _TR = TR
    __N  =  N    #  this is oo.N
    _Np1= N+1
    _k  = k
    _kk = k*k
    kinfo = _N.zeros(3, dtype=_N.int32)
    cdef int[::1] v_kinfo = kinfo
    p_kinfo     = &v_kinfo[0]
    kinfo[0] = k
    kinfo[1] = k



########################   FFBS
#def armdl_FFBS_1itrMP(y, Rv, F, q2, N, k, fx00, fV00):   #  approximation
@cython.boundscheck(False)
@cython.wraparound(False)
def armdl_FFBS_1itrMP(double[:, ::1] gau_obs, double[:, ::1] gau_var, double[:, :, ::1] F, double[:, :, ::1] iF, double[::1] q2, long[::1] Ns, long[::1] ks, double[:, :, ::1] fx, double[:, :, :, ::1] fV, fV_chol_L, double[:, :, :, ::1] ifV, double[:, :, ::1] px, double[:, :, :, ::1] pV, smpx, double[:, :, ::1] K):   #  approximation
    global __N, _Np1, _k, _kk, _TR, p_kinfo, kinfo
    #ttt1 = _tm.time()
    cdef double [:, :, :, ::1] mv_fV_chol_L = fV_chol_L
    cdef double *p_fV_chol_L  = &mv_fV_chol_L[0, 0, 0, 0]
    cdef double* p_gau_obs  = &gau_obs[0, 0]
    cdef double* p_gau_var  = &gau_var[0, 0]
    cdef double* p_F        = &F[0, 0, 0]
    cdef double* p_iF        = &iF[0, 0, 0]
    cdef double* p_q2        = &q2[0]
    #cdef double* p_fx        = &fx[0, 0, 0, 0]
    cdef double* p_fx        = &fx[0, 0, 0]
    cdef double* p_fV        = &fV[0, 0, 0, 0]
    cdef double* p_ifV        = &ifV[0, 0, 0, 0]
    #cdef double* p_px        = &fx[0, 0, 0, 0]
    cdef double* p_px        = &px[0, 0, 0]
    cdef double* p_pV        = &pV[0, 0, 0, 0]
    cdef double* p_K        = &K[0, 0, 0]
    cdef double[:, :, ::1] smpx_mv = smpx
    cdef double* p_smpx            = &smpx_mv[0, 0, 0]  # TR, Np1, k

    sx_nz_vars  = _N.empty((_TR, _Np1))
    sx_norms = _N.random.randn(_TR, _Np1)

    cdef double[:, ::1] sx_nz_vars_mv = sx_nz_vars
    cdef double[:, ::1] sx_norms_mv = sx_norms
    cdef double* p_sx_nz_vars = &sx_nz_vars_mv[0, 0]
    cdef double* p_sx_norms   = &sx_norms_mv[0, 0]

    #  fx   TR x N x k
    #  fV   TR x N x k x k

    cdef long tr, i, k2, ik, jk
    k2 = _k*_k

    dotvals = _N.zeros(_TR)
    cdef double[::1] v_dotvals = dotvals
    #cdef double dotval
    cdef double* p_dotval = &v_dotvals[0]
    ucmvnrms = _N.random.randn(_TR, _k)
    cdef double[:, ::1] v_ucmvnrms = ucmvnrms
    cdef double* p_ucmvnrms = &v_ucmvnrms[0, 0]

    #print(C.shape)    #  Cholesky of last covariance TR x k x k    
    #print(ucmvnrms.shape)   #  TR x k
    #print(smXN.shape)       #  TR x k


    #ttt2 = _tm.time()

    eachL     = _N.empty((_k, _k), order='F')   #  handed to LAPACK dpotrf
    eachLdiag     = _N.empty((_k, _k), order='F')   #  handed to LAPACK dpotrf
    Li        = _N.empty((_k, _k))   #  handed to LAPACK dpotrf
    b         = _N.zeros(_k)

    cdef double[::1, :] v_eachL = eachL
    cdef double[::1, :] v_eachLdiag = eachLdiag
    cdef double[:, ::1] v_Li = Li
    cdef double[::1] v_b     = b

    with nogil:
        for tr in range(_TR):
            FFdv_new(&p_gau_obs[tr*_Np1], &p_gau_var[tr*_Np1], &p_F[tr*_k*_k], p_q2[tr], &p_fx[tr*_Np1*_k], &p_fV[tr*_Np1*_k*_k], &p_px[tr*_Np1*_k], &p_pV[tr*_Np1*_k*_k], &p_K[tr*_Np1*_k])
            # ##########  BS

        invCovMats_cdef(_TR, _Np1, _k, p_fV, p_fV_chol_L, p_ifV, p_kinfo, v_eachL, v_eachLdiag, v_Li, v_b)
    # #  TR x k x k   mult   TR x k   we want TR x k
    # #  basically for each trial, 


    # smXN       = _N.einsum("njk,nk->nj", fV_chol_L[:, __N], ucmvnrms) + fx[:, _Np1]
    # smpx[:, __N] = smXN   #  not as a memview

    # print("0000000000000000000000000000")
    # print(smpx[:, __N])
    # ssss = _N.empty((_TR, _k))

        for tr in range(_TR):
            #  doing this for the last
            for ik in range(_k):   #  nth trial, jth time bin
                p_dotval[tr] = 0
                for jk in range(_k):
                    p_dotval[tr] += p_fV_chol_L[_Np1*k2*tr+__N*k2+ik*_k+jk] * p_ucmvnrms[tr*_k+jk] 
                p_dotval[tr] += p_fx[tr*_k*_Np1 + _Np1*_k + ik]
                p_smpx[tr*_Np1*_k + __N*_k + ik] = p_dotval[tr]

            BSvec(&p_iF[tr*_k*_k], &p_ifV[tr*_Np1*_k*_k], p_q2[tr], &p_fx[tr*_Np1*_k], &p_fV[tr*_Np1*_k*_k], &p_smpx[tr*_Np1*_k], &p_sx_nz_vars[tr*_Np1], &p_sx_norms[tr*_Np1])

            #ssss[tr, ik] = dotval
    #     #smpx[:, _Np1+1] = smXN   #  not as a memview
    #     #p_smpx[n, :, __N] = smXN   #  not as a memview
    # #     #ttt6 = _tm.time()
    #print(smpx[:, __N])

        # for tr in range(_TR):
        #     #  doing this for the last
        #     for ik in range(_k):   #  nth trial, jth time bin
        #         dotval = 0
        #         for jk in range(_k):
        #             dotval += p_fV_chol_L[_Np1*k2*tr+__N*k2+ik*_k+jk] * p_ucmvnrms[tr*_k+jk] 
        #         dotval += p_fx[tr*_k*_Np1 + _Np1*_k + ik]
        #         p_smpx[tr*_Np1*_k + __N*_k + ik] = dotval
        # #     #smpx[:, _Np1+1] = smXN   #  not as a memview
        # #     #p_smpx[n, :, __N] = smXN   #  not as a memview
        # # #     #ttt6 = _tm.time()

        #for tr in range(_TR):















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
cdef void FFdv_new(double* p_gau_obs, double* p_gau_var, double* p_F, double q2, double* p_fx, double* p_fV, double* p_px, double* p_pV, double* p_K) nogil:   #  approximate KF    #  k==1,dynamic variance
    global __N, _k, _kk, _Np1
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
cdef void BSvec(double* p_iF, double *p_ifV, double q2, double* p_fx, double* p_fV, double* p_smpx, double* p_sx_nz_vars, double* p_sx_norms) nogil:
    #  Backward sampling.
    #
    #  1)  find covs - only requires values calculated from filtering step
    #  2)  Only p,p-th element in cov mat is != 0.
    #  3)  genearte 0-mean normals from variances computed in 2)
    #  4)  calculate means, add to 0-mean norms (backwds samp) to p-th component
    # 
    global __N, _k, _Np1
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

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
#cdef void invCovMats_cdef(int TR, int N, int k, double[:, :, :, ::1] vCovs, double[:, :, :, ::1] vLout, double[:, :, :, ::1] vCovsOut, int[::1] kinfo, double[::1, :] v_eachL, double[::1, :] v_eachLdiag, double[:, ::1] v_Li, double[::1] v_b) nogil:
cdef void invCovMats_cdef(int TR, int N, int k, double *p_Covs, double* p_Lout, double* p_invCov, int* p_kinfo, double[::1, :] v_eachL, double[::1, :] v_eachLdiag, double[:, ::1] v_Li, double[::1] v_b) nogil:
    """
    vCovs     - the fV array
    vLout     - provide this
    vCovsOut  - provide this
    """
#def chol_trials_n(int TR, int N, int k, A, int[::1] kinfo):
#def chol_trials_n(int TR, int N, A, int[::1] kinfo):
    #cdef double[:, :, :, ::1] vA = A
    # cdef double* p_Covs = &vCovs[0, 0, 0, 0]
    # cdef double* p_Lout = &vLout[0, 0, 0, 0]
    # cdef double* p_invCov = &vCovsOut[0, 0, 0, 0]

    #eachL     = _N.empty((k, k), order='F')   #  handed to LAPACK dpotrf
    #eachLdiag     = _N.empty((k, k), order='F')   #  handed to LAPACK dpotrf

    #Li        = _N.empty((k, k))   #  handed to LAPACK dpotrf

    #b         = _N.zeros(k)
    #cdef double[::1] v_b = b
    cdef double* p_b = &v_b[0]
    #cdef double[::1, :] v_eachL = eachL
    #cdef double[::1, :] v_eachLdiag = eachLdiag
    #cdef double[:, ::1] v_Li = Li

    #each     = _N.empty((k, k))
    #cdef double[:, ::1] v_each = each

    cdef double * p_eachL = &v_eachL[0, 0]
    cdef double * p_eachLdiag = &v_eachLdiag[0, 0]
    cdef double * p_Li = &v_Li[0, 0]
    cdef char uplo = 'L';
    cdef double dotval
    ##  lda should just be n  (matrix not embedded)
    #cdef int* p_kinfo = &kinfo[0]
    cdef int tr, n, m, icell, ir, ic, ii

    cdef int k2 = k*k
    cdef int k2Ntrn

    for tr in range(TR):
        for n in range(N):
            k2Ntrn = (N*tr+n)*k2
            for ir in range(k):
                for ic in range(ir, k):
                    p_eachL[ir*k+ ic] = p_eachL[ic*k+ ir] = p_Covs[k2Ntrn + ir*k + ic]
            #dpotrf(char *uplo, int *n, d *a, int *lda, int *info) nogil
            _cll.dpotrf(&uplo, &p_kinfo[0], p_eachL, &p_kinfo[1], &p_kinfo[2])

            ##  copy back Cholesky L matrix
            for ir in range(k):
                for ic in range(0, ir):
                    p_Lout[k2Ntrn+ir*k+ ic] = p_eachL[ic*k + ir]
                p_Lout[k2Ntrn+ir*k+ ir] = p_eachL[ir*k + ir]
                #p_eachLdiag[ir*k + ir] = 1./p_eachL[ir*k + ir]
                p_eachLdiag[ir*k + ir] = 1./p_eachL[ir*k + ir]


            #  calculate inverse of Cholesky L
            #  Build inverse L  one column at a time
            for ic in range(k): 
                #for ir in range(k):
                #    p_b[ir] = 0
                p_b[ic] = 1
                if ic > 0:
                    p_b[ic-1] = 0
                else:
                    p_b[k-1] = 0

                #  Li[0, ic]   p_eachL[0, 0]
                p_Li[ic] = p_b[0] * p_eachLdiag[0]   #  we already know that Li[r, c] = 0 if r >= c
                for m in range(1, k):    ##  m-th row of ic-th column of Li
                    dotval = 0
                    #_N.dot(L[m, 0:m], Li[0:m, ic])

                    #  Li == 0 if ii < ic or ii < ir, this term is 0.
                    for ii in range(ic, m):
                        dotval += p_eachL[ii*k + m] * p_Li[ii*k+ic]
                    p_Li[m*k+ic] = (p_b[m] - dotval) * p_eachLdiag[m*k+ m]

            ##  now calculate inverse of covariance
            for ic in range(k):
                for ir in range(ic, k):   #  ir >= ic
                    dotval = 0
                    #_N.dot(Li.T[ir, :], Li[:, ic]) = 
                    #  _N.dot(Li[:, ir], Li[:, ic])
                    #  Li[ii, ir]   and Li[ii, ic]
                    #  Li == 0 if ii < ic or ii < ir, this term is 0.
                    #  ir >= ic.  ic is the lower bound, so use it instead of 0
                    #for ii in range(k):
                    for ii in range(ic, k):
                        dotval += p_Li[ii*k+ ir]*p_Li[ii*k+ic]
                    p_invCov[k2Ntrn + ir*k + ic] = p_invCov[k2Ntrn + ic*k + ir] = dotval

