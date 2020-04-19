import numpy as _N
cimport numpy as _N
cimport cython
import time as _tm

"""
c functions
"""
cdef extern from "math.h":
    double sqrt(double)


###  Most expensive operation here is the SVD
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def BSvec(double[:, ::1] iF, long N, long k, double q2, double[:, :, ::1] fx, double[:, :, ::1] fV, double[:, ::1] smXN):
    #  Backward sampling.
    #
    #  1)  find covs - only requires values calculated from filtering step
    #  2)  Only p,p-th element in cov mat is != 0.
    #  3)  genearte 0-mean normals from variances computed in 2)
    #  4)  calculate means, add to 0-mean norms (backwds samp) to p-th component
    # 

    cdef int n, i, j, ii, jj, nKK, nK, ik, n_m1_KK, i_m1_K, iik, kmk, km1, kp1, np1k
    cdef double trm1, trm2, trm3, c, Fs

    kmk = (k-1)*k
    km1 = k-1
    kp1 = k+1
    kk  = k*k

    smX_ram   = _N.empty((N+1, k))   #  where to store our samples

    smX_ram[N] = smXN[0]
    cdef double[:, ::1] smX_rammv = smX_ram   #  memory view
    cdef double* p_smX_ram = &smX_rammv[0, 0]
    cdef double[:, :, ::1] fxmv = fx   #  memory view
    cdef double* p_fx = &fxmv[0, 0, 0]

    ifV    = _N.linalg.inv(fV)

    cdef double[:, :, ::1] ifVmv = ifV
    cdef double* p_ifV         = &ifVmv[0, 0, 0]
    ####   ANALYTICAL.  
    nz_vars    = _N.empty(N+1)# "nik,nkj->nij", INAF, fV
    cdef double[::1]      nz_vars_mv  = nz_vars
    cdef double*  p_nz_vars  = &nz_vars_mv[0]
    cdef double*        p_iF = &iF[0, 0]
    cdef double iF_p1_2     = iF[k-1, 0]*iF[k-1, 0]

    norms = _N.random.randn(N+1)
    cdef double[::1] normsmv = norms
    cdef double* p_norms = &normsmv[0]

    for j in xrange(N+1):
        p_nz_vars[j] = sqrt((q2*iF_p1_2)/(1+q2*p_ifV[j*kk + kmk + km1]*iF_p1_2))*p_norms[j]

        #print "noise %d" % j
        #print (q2*iF_p1_2)/(1+q2*p_ifV[j*kk + kmk + km1]*iF_p1_2)



        
    ##  ptrs for ifV, smX_ram, iF, fx
    ###  analytical method.  only update 1 





    # """
    # TESTING
    # """
    # Ik      = _N.identity(k)
    # IkN   =  _N.tile(Ik, (N+1, 1, 1))
    # smX   = _N.empty((N+1, k))   #  where to store our samples
    # cdef double[:, ::1] smXmv = smX   #  memory view
    # smX[N] = smXN[:]

    # fFT    = _N.dot(fV, F.T)  # dot([N+1 x k x k], [k, k])
    # #  sum F_{il} V_{lm} F_{mj}
    # FfFT   = _N.einsum("il,nlj->nij", F, fFT)
    # t2 = _tm.time()
    # iv     = _N.linalg.inv(FfFT + _N.tile(GQGT, (N+1,1,1)))
    # A      = _N.einsum("nik,nkj->nij", fFT, iv)
    # INAF   = IkN*1.000001 - _N.dot(A, F)
    # PtN    = _N.einsum("nik,nkj->nij", INAF, fV)  #  covarainces
    # ##  NOW multivariate normal
    # #mvn1   = _N.random.multivariate_normal(_N.zeros(k), Ik, size=(N+1))
    # mvn1   = _N.random.randn(N+1, k)  #  slightly faster
    # t3 = _tm.time()
    # S,V,D  = _N.linalg.svd(PtN)  #  PtN is almost 0 everywhere
    # Vs     = _N.sqrt(V)
    # VsRn2  =  Vs*mvn1
    # zrmn   = _N.einsum("njk,nk->nj", S, VsRn2)

    # cdef double[:, ::1] zrmnmv = zrmn

    # #  out of order calculation.  one of the terms can be calculated
    # INAFfx = _N.einsum("nj,nj->n", INAF[:, k-1], fx[:, :, 0])
    # cdef double[::1] INAFfxmv = INAFfx
    # last   = _N.zeros(k)
    # last[k-1] = 1

    # #  temp storage
    # Asx = _N.empty(k)
    # cdef double[::1] Asxmv = Asx
    """
    TESTING
    """
    """
    if k == 9:
        for n from N > n >= 0:
            nKK = n*k*k
            nK  = n*k
            np1k = (n+1)*k

            c = 1 + q2*p_ifV[nKK + kmk + km1]*iF_p1_2

            Fs = 0
            trm2 = 0
            trm3 = 0

            #for ik in xrange(km1):  #  shift
            #  ik == 0, 1, 2, 3, 4, 5, 6, 7
            #    p_smX_ram[nK + ik] = p_smX_ram[np1k + ik+1]
            #    trm2 += p_smX_ram[np1k + ik+1]*p_ifV[nKK + kmk + ik]
            #    Fs += p_iF[kmk + ik]*p_smX_ram[np1k+ ik]
            #    trm3 += p_fx[nK + ik]*p_ifV[nKK + kmk + ik]


            #  ik == 0
            p_smX_ram[nK] = p_smX_ram[np1k +1]
            trm2 += p_smX_ram[np1k +1]*p_ifV[nKK + kmk]
            Fs += p_iF[kmk]*p_smX_ram[np1k]
            trm3 += p_fx[nK]*p_ifV[nKK + kmk]
            #  ik == 1
            p_smX_ram[nK+1] = p_smX_ram[np1k +2]
            trm2 += p_smX_ram[np1k +2]*p_ifV[nKK + kmk+1]
            Fs += p_iF[kmk+1]*p_smX_ram[np1k+1]
            trm3 += p_fx[nK+1]*p_ifV[nKK + kmk+1]
            #  ik == 2
            p_smX_ram[nK+2] = p_smX_ram[np1k +3]
            trm2 += p_smX_ram[np1k +3]*p_ifV[nKK + kmk+2]
            Fs += p_iF[kmk+2]*p_smX_ram[np1k+2]
            trm3 += p_fx[nK+2]*p_ifV[nKK + kmk+2]
            #  ik == 3
            p_smX_ram[nK+3] = p_smX_ram[np1k +4]
            trm2 += p_smX_ram[np1k +4]*p_ifV[nKK + kmk+3]
            Fs += p_iF[kmk+3]*p_smX_ram[np1k+3]
            trm3 += p_fx[nK+3]*p_ifV[nKK + kmk+3]
            #  ik == 4
            p_smX_ram[nK+4] = p_smX_ram[np1k +5]
            trm2 += p_smX_ram[np1k +5]*p_ifV[nKK + kmk+4]
            Fs += p_iF[kmk+4]*p_smX_ram[np1k+4]
            trm3 += p_fx[nK+4]*p_ifV[nKK + kmk+4]
            #  ik == 5
            p_smX_ram[nK+5] = p_smX_ram[np1k +6]
            trm2 += p_smX_ram[np1k +6]*p_ifV[nKK + kmk+5]
            Fs += p_iF[kmk+5]*p_smX_ram[np1k+5]
            trm3 += p_fx[nK+5]*p_ifV[nKK + kmk+5]
            #  ik == 6
            p_smX_ram[nK+6] = p_smX_ram[np1k +7]
            trm2 += p_smX_ram[np1k +7]*p_ifV[nKK + kmk+6]
            Fs += p_iF[kmk+6]*p_smX_ram[np1k+6]
            trm3 += p_fx[nK+6]*p_ifV[nKK + kmk+6]
            #  ik == 7
            p_smX_ram[nK+7] = p_smX_ram[np1k +8]
            trm2 += p_smX_ram[np1k +8]*p_ifV[nKK + kmk+7]
            Fs += p_iF[kmk+7]*p_smX_ram[np1k+7]
            trm3 += p_fx[nK+7]*p_ifV[nKK + kmk+7]


            # last
            Fs += p_iF[kmk + km1]*p_smX_ram[np1k+ km1]
            trm3 += p_fx[nK + km1]*p_ifV[nKK + kmk + km1]

            trm1 = Fs*p_ifV[nKK + kmk+ km1]

            p_smX_ram[nK + km1]= Fs - q2*iF_p1_2*(trm1 + trm2 - trm3)/c + p_nz_vars[n]
    """
    for n from N > n >= 0:
        nKK = n*k*k
        nK  = n*k
        np1k = (n+1)*k

        c = 1 + q2*p_ifV[nKK + kmk + km1]*iF_p1_2

        Fs = 0
        trm2 = 0
        trm3 = 0

        for ik in xrange(km1):  #  shift
            p_smX_ram[nK + ik] = p_smX_ram[np1k + ik+1]
            trm2 += p_smX_ram[np1k + ik+1]*p_ifV[nKK + kmk + ik]
            Fs += p_iF[kmk + ik]*p_smX_ram[np1k+ ik]
            trm3 += p_fx[nK + ik]*p_ifV[nKK + kmk + ik]
        Fs += p_iF[kmk + km1]*p_smX_ram[np1k+ km1]
        trm3 += p_fx[nK + km1]*p_ifV[nKK + kmk + km1]

        trm1 = Fs*p_ifV[nKK + kmk+ km1]

        p_smX_ram[nK + km1]= Fs - q2*iF_p1_2*(trm1 + trm2 - trm3)/c + p_nz_vars[n]


    return smX_ram


