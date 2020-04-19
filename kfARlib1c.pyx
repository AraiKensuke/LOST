import numpy as _N
import time as _tm

import warnings
warnings.filterwarnings("error")

cimport cython
from libc.math cimport sqrt
from libc.stdio cimport printf


#######  ADDING MULTITRIAL SUPPORT
"""
p        AR order
Ftrgt    Ftrgt[0]  noise amp.  Ftrgt[1:]  AR(p) coeffs
f        freq.
f0       bandpass
f1
zr       amp. at band stop
"""
# cdef extern from "math.h":
#     double exp(double)
#     double sqrt(double)
#     double log(double)
#     double abs(double)

def armdl_FFBS_1itr_singletrial(long Nobs,
                                double[::1] gau_obs, double[::1] gau_var,
                                double F0, double q2,
                                double[::1] fx, double[::1] fV,
                                double[::1] px, double[::1] pV,
                                double[::1] smpx, double[::1] K):
    cdef double dQ      =    q2
    cdef double GQGT    =    dQ

    cdef double* p_y                  = &gau_obs[0]
    cdef double* p_Rv                 = &gau_var[0]
    cdef double* p_fx                 = &fx[0]    
    cdef double* p_fV                 = &fV[0]
    cdef double* p_px                 = &px[0]    
    cdef double* p_pV                 = &pV[0]
    cdef double* p_K                  = &K[0]
    cdef double* p_smpx               = &smpx[0]    
    
    #  do this until p_V has settled into stable values
    cdef int m, n
    cdef double mat

    nrands = _N.random.randn(Nobs)
    cdef double[::1] mv_nrands        = nrands
    cdef double* p_nrands             = &mv_nrands[0]
    with nogil:
        for n from 1 <= n < Nobs:
            p_px[n] = F0 * p_fx[n - 1]
            p_pV[n] = F0 * p_fV[n - 1] * F0 + GQGT

            mat  = 1 / (p_pV[n] + p_Rv[n])
            p_K[n] = p_pV[n]*mat
            p_fx[n]    = p_px[n] + p_K[n]*(p_y[n] - p_px[n])
            p_fV[n] = (1 - p_K[n])* p_pV[n]

        BS1(Nobs, F0, q2, p_fx, p_fV, p_px, p_pV, p_nrands, p_smpx)

cdef void BS1(long Nobs, 
              double F0, double q2,
              double* p_fx, double* p_fV,
              double* p_px, double* p_pV,
              double* p_nrands, double *p_smX) nogil:
    cdef double GQGT    = q2

    p_smX[Nobs-1] = p_fx[Nobs-1] + sqrt(p_fV[Nobs-1]) * p_nrands[Nobs-1]
    cdef double F02  = F0*F0
    cdef double p1, p2, p3, Sig, M

    ##########  SMOOTHING step
    cdef double iGQGT= 1./GQGT
    cdef double FTiGQGTF = iGQGT*F02
    cdef int n
    
    #for n in xrange(_d.N - 1, -1, -1):
    for n from Nobs-2 >= n > -1:
        Sig  = 1/(FTiGQGTF + 1/(p_fV[n]))
        p1   = p_fV[n]* F0
        p2   = 1/(F02*p_fV[n] +GQGT)
        p3   = p_smX[n+1] - F0*p_fx[n]
        M    = p_fx[n] + p1* p2* p3
        p_smX[n]= M + sqrt(Sig) * p_nrands[n]

