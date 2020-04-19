import numpy as _N
cimport numpy as _N
cimport cython
import time as _tm

"""
c functions
"""
cdef extern from "math.h":
    double sqrt(double)

#Ik = _N.identity(9)
#IkN= _N.tile(Ik, (750, 1, 1))


###  Most expensive operation here is the SVD
@cython.boundscheck(False)
@cython.wraparound(False)
def BSvec(F, N, _N.intp_t k, GQGT, fx, fV, smXN):
    t1 = _tm.time()
    Ik      = _N.identity(k)
    IkN   =  _N.tile(Ik, (N+1, 1, 1))
    smX   = _N.empty((N+1, k))   #  where to store our samples
    cdef double[:, ::1] smXmv = smX   #  memory view
    smX[N] = smXN[:]

    fFT    = _N.dot(fV, F.T)  # dot([N+1 x k x k], [k, k])
    #  sum F_{il} V_{lm} F_{mj}
    FfFT   = _N.einsum("il,nlj->nij", F, fFT)
    t2 = _tm.time()
    iv     = _N.linalg.inv(FfFT + _N.tile(GQGT, (N+1,1,1)))
    A      = _N.einsum("nik,nkj->nij", fFT, iv)
    INAF   = IkN*1.000001 - _N.dot(A, F)
    PtN    = _N.einsum("nik,nkj->nij", INAF, fV)  #  covarainces
    ##  NOW multivariate normal
    #mvn1   = _N.random.multivariate_normal(_N.zeros(k), Ik, size=(N+1))
    mvn1   = _N.random.randn(N+1, k)  #  slightly faster
    t3 = _tm.time()
    S,V,D  = _N.linalg.svd(PtN)  #  PtN is almost 0 everywhere
    Vs     = _N.sqrt(V)
    VsRn2  =  Vs*mvn1
    zrmn   = _N.einsum("njk,nk->nj", S, VsRn2)

    #print "noises"
    #print PtN

    cdef double[:, ::1] zrmnmv = zrmn

    #  out of order calculation.  one of the terms can be calculated
    INAFfx = _N.einsum("nj,nj->n", INAF[:, k-1], fx[:, :, 0])
    cdef double[::1] INAFfxmv = INAFfx
    last   = _N.zeros(k)
    last[k-1] = 1

    #  temp storage
    Asx = _N.empty(k)
    cdef double[::1] Asxmv = Asx

    cdef _N.intp_t t, i, n
    t4 = _tm.time()

    for n in xrange(N - 1, -1, -1):
        #smX[n] = zrmn[n, :, 0] + INAFfx[n]*last + _N.dot(A[n], smX[n+1])
        _N.dot(A[n], smX[n+1], out=Asx)
        smXmv[n, k-1] = zrmnmv[n, k-1] + Asxmv[k-1] + INAFfxmv[n]
        for i from 0 <= i < k-1:
            smXmv[n, i] = zrmnmv[n, i] + Asxmv[i]

    t5 = _tm.time()


    #print "%(t2t1).3f   %(t3t2).3f   %(t4t3).3f   %(t5t4).3f" % {"t2t1" : (t2-t1), "t3t2" : (t3-t2), "t4t3" : (t4-t3), "t5t4" : (t5-t4)}
    return smX



###  Most expensive operation here is the SVD
@cython.boundscheck(False)
@cython.wraparound(False)
def BSvecChol(F, N, _N.intp_t k, GQGT, fx, fV, smXN):
    Ik      = _N.identity(k)
    IkN   =  _N.tile(Ik, (N+1, 1, 1))
    smX   = _N.empty((N+1, k))   #  where to store our samples
    cdef double[:, ::1] smXmv = smX   #  memory view
    smX[N] = smXN[:]

    fFT     = _N.empty((N+1, k, k))    
    _N.dot(fV, F.T, out=fFT)  # dot([N+1 x k x k], [k, k])
    FfFTr     = _N.empty((k, k, N+1))
    _N.dot(F, fFT.T, out=FfFTr)
    iv     = _N.linalg.inv(FfFTr.T + _N.tile(GQGT, (N+1,1,1)))
    A      = _N.empty((N+1, k, k))# "nik,nkj->nij", fFT, iv
    for j in xrange(N+1):
       _N.dot(fFT[j], iv[j], out=A[j])

    INAF   = IkN*1.000001 - _N.dot(A, F)
    ####  EINSUM slow when mat x mat
    PtN    = _N.empty((N+1, k, k))# "nik,nkj->nij", INAF, fV
    for j in xrange(N+1):
       _N.dot(INAF[j], fV[j], out=PtN[j])
    #print PtN

    ##  NOW multivariate normal
    #t1 = _tm.time()
    mvn1   = _N.random.randn(N+1, k)  #  slightly faster

    try:
        C       = _N.linalg.cholesky(PtN)    ###  REPLACE svd with Cholesky
        zrmn   = _N.einsum("njk,nk->nj", C, mvn1)
    except _N.linalg.linalg.LinAlgError:
        S,V,D  = _N.linalg.svd(PtN)  #  PtN is almost 0 everywhere
        Vs     = _N.sqrt(V)
        VsRn2  =  Vs*mvn1
        zrmn   = _N.einsum("njk,nk->nj", S, VsRn2)
    cdef double[:, ::1] zrmnmv = zrmn
    #t2 = _tm.time()
    #print (t2-t1)

    #  out of order calculation.  one of the terms can be calculated
    INAFfx = _N.einsum("nj,nj->n", INAF[:, k-1], fx[:, :, 0])
    cdef double[::1] INAFfxmv = INAFfx
    last   = _N.zeros(k)
    last[k-1] = 1

    #  temp storage
    Asx = _N.empty(k)
    cdef double[::1] Asxmv = Asx

    cdef _N.intp_t t, i, n

    #for n in xrange(N - 1, -1, -1):
    for n from N > n >= 0:
        _N.dot(A[n], smX[n+1], out=Asx)
        smXmv[n, k-1] = zrmnmv[n, k-1] + Asxmv[k-1] + INAFfxmv[n]
        for i from 0 <= i < k-1:
            smXmv[n, i] = zrmnmv[n, i] + Asxmv[i]

    return smX


###  Most expensive operation here is the SVD
@cython.boundscheck(False)
@cython.wraparound(False)
def BSvecSVD(F, N, _N.intp_t k, GQGT, fx, fV, smXN):
    #print "SVD"
    #global Ik, IkN
    Ik      = _N.identity(k)
    IkN   =  _N.tile(Ik, (N+1, 1, 1))
    smX   = _N.empty((N+1, k))   #  where to store our samples
    cdef double[:, ::1] smXmv = smX   #  memory view
    smX[N] = smXN[:]

    fFT     = _N.empty((N+1, k, k))    
    _N.dot(fV, F.T, out=fFT)  # dot([N+1 x k x k], [k, k])
    FfFTr     = _N.empty((k, k, N+1))
    _N.dot(F, fFT.T, out=FfFTr)  #  FfFTr[:, :, n]  is symmetric
    iv     = _N.linalg.inv(FfFTr.T + _N.tile(GQGT, (N+1,1,1)))

    A      = _N.empty((N+1, k, k))# "nik,nkj->nij", fFT, iv
    for j in xrange(N+1):
       _N.dot(fFT[j], iv[j], out=A[j])

    INAF   = IkN - _N.dot(A, F)
    PtN = _N.einsum("nj,nj->n", INAF[:, k-1], fV[:,k-1])

    #print PtN

    mvn1   = _N.random.randn(N+1)  #  
    zrmn     = _N.sqrt(PtN)*mvn1

    #  out of order calculation.  one of the terms can be calculated
    INAFfx = _N.einsum("nj,nj->n", INAF[:, k-1], fx[:, :, 0])  #  INAF last row only
    cdef double[::1] INAFfxmv = INAFfx
    last   = _N.zeros(k)
    last[k-1] = 1

    #  temp storage
    Asx = _N.empty(k)
    cdef double[::1] Asxmv = Asx

    cdef _N.intp_t t, i, n

    for n in xrange(N - 1, -1, -1):
        _N.dot(A[n], smX[n+1], out=smX[n])
        smXmv[n, k-1] += zrmn[n] + INAFfxmv[n]


    return smX



###  Most expensive operation here is the SVD
@cython.boundscheck(False)
@cython.wraparound(False)
def BSvecSVD_new(F, N, _N.intp_t k, GQGT, fx, fV, smXN):
    #print "SVD"
    #global Ik, IkN
    Ik      = _N.identity(k)
    IkN   =  _N.tile(Ik, (N+1, 1, 1))
    smX   = _N.empty((N+1, k))   #  where to store our samples
    cdef double[:, ::1] smXmv = smX   #  memory view
    smX[N] = smXN[:]

    fFT     = _N.empty((N+1, k, k))    
    _N.dot(fV, F.T, out=fFT)  # dot([N+1 x k x k], [k, k])
    FfFTr     = _N.empty((k, k, N+1))
    _N.dot(F, fFT.T, out=FfFTr)  #  FfFTr[:, :, n]  is symmetric

    ##  was doing B^{-1}, but we only want result of operation on B^{-1}.
    B      = FfFTr.T + _N.tile(GQGT, (N+1,1,1))
    A = _N.transpose(_N.linalg.solve(B, _N.transpose(fFT, axes=(0, 2, 1))), axes=(0, 2, 1))

    INAF   = IkN - _N.dot(A, F)
    PtN = _N.einsum("nj,nj->n", INAF[:, k-1], fV[:,k-1])

    #print PtN

    mvn1   = _N.random.randn(N+1)  #  
    zrmn     = _N.sqrt(PtN)*mvn1

    #  out of order calculation.  one of the terms can be calculated
    INAFfx = _N.einsum("nj,nj->n", INAF[:, k-1], fx[:, :, 0])  #  INAF last row only
    cdef double[::1] INAFfxmv = INAFfx
    last   = _N.zeros(k)
    last[k-1] = 1

    #  temp storage
    Asx = _N.empty(k)
    cdef double[::1] Asxmv = Asx

    cdef _N.intp_t t, i, n

    for n in xrange(N - 1, -1, -1):
        _N.dot(A[n], smX[n+1], out=smX[n])
        smXmv[n, k-1] += zrmn[n] + INAFfxmv[n]


return smX
