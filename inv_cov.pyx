cimport cython
cimport scipy.linalg.cython_lapack as _cll
import numpy as _N
cimport numpy as _N
from libc.stdlib cimport malloc, free


def chol(double[:, ::1] vA, int[::1] kinfo):
    cdef double* p_A = &vA[0, 0]
    cdef char uplo = 'L';
    ##  lda should just be n  (matrix not embedded)
    cdef int* p_kinfo = &kinfo[0]

    with nogil:
        #dpotrf(char *uplo, int *n, d *a, int *lda, int *info) nogil
        _cll.dpotrf(&uplo, &p_kinfo[0], p_A, &p_kinfo[1], &p_kinfo[2])

#def chol_trials_n(int TR, int N, int k, double[:, :, :, ::1] vA, int[::1] kinfo):
# def chol_trials_n(int TR, int N, int k, A, int[::1] kinfo):
#     A_Forder = _N.array(A, order="F")
#     cdef double[::1, :, :, :] vA_Forder = A_Forder
#     cdef double* p_A_Forder = &vA_Forder[0, 0, 0, 0]
#     cdef double* p_A = &vA[0, 0, 0, 0]
#     cdef char uplo = 'U';
#     ##  lda should just be n  (matrix not embedded)
#     cdef int* p_kinfo = &kinfo[0]
#     cdef int tr, n, icell, ir, ic

#     # https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html
#     #  matrix is TR x N x k x k
#     #  C ordering
#     #  (0, 0, 0, 0) and (0, 0, 0, 1)   are 1 unit apart
#     #  (0, 0, 0, 0) and (0, 0, 1, 0)   are k units apart
#     #  (0, 0, 1, 0) and (0, 0, 1, 0)   are k unit apart
#     #  (0, 0, 0, 0) and (1, 0, 0, 0)   are k*k*N units apart
#     #  (0, 0, 0, 0) and (1, 0, 0, 1)   are k*k*N+1 units apart
#     #  (0, 0, 0, 0) and (1, 0, 1, 0)   are k*k*N+k units apart
#     #  F ordering
#     #  (0, 0, 0, 0) and (1, 0, 0, 0)   are 1 unit apart
#     #  (0, 0, 0, 0) and (0, 1, 0, 0)   are TR units part
#     #  (0, 0, 0, 0) and (1, 1, 0, 0)   are TR + 1 units part
#     #  (0, 0, 0, 0) and (TR-1, 0, 0, 0)   are TR - 1 units part
#     #  (0, 0, 0, 0) and (TR-1, 1, 0, 0)   are TR + TR - 1 units part
#     #  (0, 0, 0, 0) and (tr, 1, 0, 0)   are TR + tr units part
#     #  (0, 0, 0, 0) and (tr, n, 0, 0)   are n*TR + tr units part
#     #  (0, 0, 0, 0) and (tr, n, i1, 0)   are tr + n*TR + i1*N*TR units part
#     #  (0, 0, 0, 0) and (tr, n, i1, i2)   are tr + n*TR + i1*N*TR units part
#     #  (0, 0, 0, 0) and (tr, n, i1, i2)   are tr + n*TR + i1*N*TR + i2*k*N*TR units part


#     #  (tr, n, i1, i2)     tr + n*TR + i1*N*TR + i2*k*N*TR
#     #  (tr, n, i1+1, i2)   tr + n*TR + (i1+1)*N*TR + i2*k*N*TR
#     # for tr in range(TR):
#     #     for n in range(N):
#     #         #  k*k*n      k*k*n*tr   
#     #         #icell = k*k*n*(1+tr)
#     #         icell = tr + n*TR
#     #         #dpotrf(char *uplo, int *n, d *a, int *lda, int *info) nogil

#     #         _cll.dpotrf(&uplo, &p_kinfo[0], &(p_A_Forder[icell]), &p_kinfo[1], &p_kinfo[2])
#     #         # for ic in range(k):
#     #         #     for ir in range(0, ic):
#     #         #         p_A_Forder[icell + ir*N*TR + ic*k*N*TR] = 0

#     # for tr in range(TR):
#     #     for n in range(N):
#     #         print("---------")
#     #         print(A_Forder[tr, n])

#     with nogil:
#         for tr in range(TR):
#             for n in range(N):
#                 #  k*k*n      k*k*n*tr   
#                 icell = k*k*n*(1+tr)
#                 #dpotrf(char *uplo, int *n, d *a, int *lda, int *info) nogil
#                 _cll.dpotrf(&uplo, &p_kinfo[0], &(p_A[icell]), &p_kinfo[1], &p_kinfo[2])
#                 # for ic in range(k):
#                 #     for ir in range(0, ic):
#                 #         p_A[icell + ir*k + ic] = 0

                
#     for tr in range(TR):
#         for n in range(N):
#             print("---------")
#             print(A[tr, n])


def chol_trials_n(int TR, int N, int k, double[:, :, :, ::1] vA, double[:, :, :, ::1] vOut, int[::1] kinfo):
#def chol_trials_n(int TR, int N, int k, A, int[::1] kinfo):
#def chol_trials_n(int TR, int N, A, int[::1] kinfo):
    #cdef double[:, :, :, ::1] vA = A
    cdef double* p_A = &vA[0, 0, 0, 0]
    cdef double* p_Out = &vOut[0, 0, 0, 0]

    each     = _N.empty((k, k), order='F')
    cdef double[::1, :] v_each = each
    #each     = _N.empty((k, k))
    #cdef double[:, ::1] v_each = each


    cdef double * p_each = &v_each[0, 0]
    cdef char uplo = 'L';
    ##  lda should just be n  (matrix not embedded)
    cdef int* p_kinfo = &kinfo[0]
    cdef int tr, n, icell, ir, ic

    cdef int k2 = k*k
    cdef int k2Ntrn

    with nogil:
        for tr in range(TR):
            for n in range(N):
                k2Ntrn = (N*tr+n)*k2
                for ir in range(k):
                    for ic in range(ir, k):
                        p_each[ir*k+ ic] = p_each[ic*k+ ir] = p_A[k2Ntrn + ir*k + ic]
                #dpotrf(char *uplo, int *n, d *a, int *lda, int *info) nogil
                _cll.dpotrf(&uplo, &p_kinfo[0], p_each, &p_kinfo[1], &p_kinfo[2])
                for ir in range(k):
                    for ic in range(0, ir+1):
                        p_Out[k2Ntrn+ir*k+ ic] = p_each[ic*k + ir]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def invCovMats(int TR, int N, int k, double[:, :, :, ::1] vCovs, double[:, :, :, ::1] vLout, double[:, :, :, ::1] vCovsOut, int[::1] kinfo):
    """
    vCovs     - the fV array
    vLout     - provide this
    vCovsOut  - provide this
    """
#def chol_trials_n(int TR, int N, int k, A, int[::1] kinfo):
#def chol_trials_n(int TR, int N, A, int[::1] kinfo):
    #cdef double[:, :, :, ::1] vA = A
    cdef double* p_Covs = &vCovs[0, 0, 0, 0]
    cdef double* p_Lout = &vLout[0, 0, 0, 0]
    cdef double* p_invCov = &vCovsOut[0, 0, 0, 0]

    eachL     = _N.empty((k, k), order='F')   #  handed to LAPACK dpotrf
    eachLdiag     = _N.empty((k, k), order='F')   #  handed to LAPACK dpotrf

    Li        = _N.empty((k, k))   #  handed to LAPACK dpotrf

    b         = _N.zeros(k)
    cdef double[::1] v_b = b
    cdef double* p_b = &v_b[0]
    cdef double[::1, :] v_eachL = eachL
    cdef double[::1, :] v_eachLdiag = eachLdiag
    cdef double[:, ::1] v_Li = Li

    #each     = _N.empty((k, k))
    #cdef double[:, ::1] v_each = each

    cdef double * p_eachL = &v_eachL[0, 0]
    cdef double * p_eachLdiag = &v_eachLdiag[0, 0]
    cdef double * p_Li = &v_Li[0, 0]
    cdef char uplo = 'L';
    cdef double dotval
    ##  lda should just be n  (matrix not embedded)
    cdef int* p_kinfo = &kinfo[0]
    cdef int tr, n, m, icell, ir, ic, ii

    cdef int k2 = k*k
    cdef int k2Ntrn

    with nogil:
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


# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def invCovMats_cdef(int TR, int N, int k, double[:, :, :, ::1] vCovs, double[:, :, :, ::1] vLout, double[:, :, :, ::1] vCovsOut, int[::1] kinfo, double[::1, :] v_eachL, double[::1, :] v_eachLdiag, double[:, ::1] v_Li, double[::1] v_b):
#     """
#     vCovs     - the fV array
#     vLout     - provide this
#     vCovsOut  - provide this
#     """
# #def chol_trials_n(int TR, int N, int k, A, int[::1] kinfo):
# #def chol_trials_n(int TR, int N, A, int[::1] kinfo):
#     #cdef double[:, :, :, ::1] vA = A
#     cdef double* p_Covs = &vCovs[0, 0, 0, 0]
#     cdef double* p_Lout = &vLout[0, 0, 0, 0]
#     cdef double* p_invCov = &vCovsOut[0, 0, 0, 0]

#     #eachL     = _N.empty((k, k), order='F')   #  handed to LAPACK dpotrf
#     #eachLdiag     = _N.empty((k, k), order='F')   #  handed to LAPACK dpotrf

#     #Li        = _N.empty((k, k))   #  handed to LAPACK dpotrf

#     #b         = _N.zeros(k)
#     #cdef double[::1] v_b = b
#     cdef double* p_b = &v_b[0]
#     #cdef double[::1, :] v_eachL = eachL
#     #cdef double[::1, :] v_eachLdiag = eachLdiag
#     #cdef double[:, ::1] v_Li = Li

#     #each     = _N.empty((k, k))
#     #cdef double[:, ::1] v_each = each

#     cdef double * p_eachL = &v_eachL[0, 0]
#     cdef double * p_eachLdiag = &v_eachLdiag[0, 0]
#     cdef double * p_Li = &v_Li[0, 0]
#     cdef char uplo = 'L';
#     cdef double dotval
#     ##  lda should just be n  (matrix not embedded)
#     cdef int* p_kinfo = &kinfo[0]
#     cdef int tr, n, m, icell, ir, ic, ii

#     cdef int k2 = k*k
#     cdef int k2Ntrn

#     with nogil:
#         for tr in range(TR):
#             for n in range(N):
#                 k2Ntrn = (N*tr+n)*k2
#                 for ir in range(k):
#                     for ic in range(ir, k):
#                         p_eachL[ir*k+ ic] = p_eachL[ic*k+ ir] = p_Covs[k2Ntrn + ir*k + ic]
#                 #dpotrf(char *uplo, int *n, d *a, int *lda, int *info) nogil
#                 _cll.dpotrf(&uplo, &p_kinfo[0], p_eachL, &p_kinfo[1], &p_kinfo[2])

#                 ##  copy back Cholesky L matrix
#                 for ir in range(k):
#                     for ic in range(0, ir):
#                         p_Lout[k2Ntrn+ir*k+ ic] = p_eachL[ic*k + ir]
#                     p_Lout[k2Ntrn+ir*k+ ir] = p_eachL[ir*k + ir]
#                     #p_eachLdiag[ir*k + ir] = 1./p_eachL[ir*k + ir]
#                     p_eachLdiag[ir*k + ir] = 1./p_eachL[ir*k + ir]


#                 #  calculate inverse of Cholesky L
#                 #  Build inverse L  one column at a time
#                 for ic in range(k): 
#                     #for ir in range(k):
#                     #    p_b[ir] = 0
#                     p_b[ic] = 1
#                     if ic > 0:
#                         p_b[ic-1] = 0
#                     else:
#                         p_b[k-1] = 0

#                     #  Li[0, ic]   p_eachL[0, 0]
#                     p_Li[ic] = p_b[0] * p_eachLdiag[0]   #  we already know that Li[r, c] = 0 if r >= c
#                     for m in range(1, k):    ##  m-th row of ic-th column of Li
#                         dotval = 0
#                         #_N.dot(L[m, 0:m], Li[0:m, ic])

#                         #  Li == 0 if ii < ic or ii < ir, this term is 0.
#                         for ii in range(ic, m):
#                             dotval += p_eachL[ii*k + m] * p_Li[ii*k+ic]
#                         p_Li[m*k+ic] = (p_b[m] - dotval) * p_eachLdiag[m*k+ m]

#                 ##  now calculate inverse of covariance
#                 for ic in range(k):
#                     for ir in range(ic, k):   #  ir >= ic
#                         dotval = 0
#                         #_N.dot(Li.T[ir, :], Li[:, ic]) = 
#                         #  _N.dot(Li[:, ir], Li[:, ic])
#                         #  Li[ii, ir]   and Li[ii, ic]
#                         #  Li == 0 if ii < ic or ii < ir, this term is 0.
#                         #  ir >= ic.  ic is the lower bound, so use it instead of 0
#                         #for ii in range(k):
#                         for ii in range(ic, k):
#                             dotval += p_Li[ii*k+ ir]*p_Li[ii*k+ic]
#                         p_invCov[k2Ntrn + ir*k + ic] = p_invCov[k2Ntrn + ic*k + ir] = dotval


def chol_ichol_trials_n(int TR, int N, int k, double[:, :, :, ::1] vCovs, double[:, :, :, ::1] vLout, double[:, :, :, ::1] vLiout, int[::1] kinfo):
    """
    vCovs     - the fV array
    vLout     - provide this
    vCovsOut  - provide this
    """
#def chol_trials_n(int TR, int N, int k, A, int[::1] kinfo):
#def chol_trials_n(int TR, int N, A, int[::1] kinfo):
    #cdef double[:, :, :, ::1] vA = A
    cdef double* p_Covs = &vCovs[0, 0, 0, 0]
    cdef double* p_Lout = &vLout[0, 0, 0, 0]
    cdef double* p_Liout = &vLiout[0, 0, 0, 0]

    eachL     = _N.empty((k, k), order='F')   #  handed to LAPACK dpotrf
    Li        = _N.empty((k, k))   

    b         = _N.zeros(k)
    cdef double[::1] v_b = b
    cdef double* p_b = &v_b[0]
    cdef double[::1, :] v_eachL = eachL
    cdef double[:, ::1] v_Li = Li

    #each     = _N.empty((k, k))
    #cdef double[:, ::1] v_each = each


    cdef double * p_eachL = &v_eachL[0, 0]
    cdef double * p_Li = &v_Li[0, 0]
    cdef char uplo = 'L';
    cdef double dotval
    ##  lda should just be n  (matrix not embedded)
    cdef int* p_kinfo = &kinfo[0]
    cdef int tr, n, m, icell, ir, ic, ii

    cdef int k2 = k*k
    cdef int k2Ntrn

    #with nogil:
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
                for ic in range(0, ir+1):
                    p_Lout[k2Ntrn+ir*k+ ic] = p_eachL[ic*k + ir]

            #  calculate inverse of Cholesky L
            #  Build inverse L  one column at a time
            for ic in range(k):    
                for ir in range(k):
                    p_b[ir] = 0
                p_b[ic] = 1
                # if ic > 0:
                #     p_b[ic-1] = 0

                #  Li[0, ic]   p_eachL[0, 0]
                p_Li[ic] = p_b[0] / p_eachL[0]   #  we already know that Li[r, c] = 0 if r >= c
                #v_Li[0, ic] = p_b[0] / v_eachL[0, 0]   #  we already know that Li[r, c] = 0 if r >= c
                for m in range(1, k):    ##  m-th row of ic-th column of Li
                    dotval = 0
                    #_N.dot(L[m, 0:m], Li[0:m, ic])
                    for ii in range(m):
                        dotval += p_eachL[ii*k + m] * p_Li[ii*k+ic]
                        #dotval += v_eachL[m, ii] * v_Li[ii, ic]
                    p_Li[m*k+ic] = (p_b[m] - dotval) / p_eachL[m*k+ m]
                    #v_Li[m, ic] = (p_b[m] - dotval) / v_eachL[m, m]

            ##  copy back Cholesky L matrix
            for ir in range(k):
                for ic in range(0, ir+1):
                    p_Liout[k2Ntrn+ir*k+ ic] = p_Li[ir*k + ic]

