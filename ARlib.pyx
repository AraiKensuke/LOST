import numpy as _N

#  atan for 1 pair of (x, y)
def azim(x, y):
    if x == 0 and y > 0:
        atan =  0.5*_N.pi
    elif x == 0 and y < 0:
        atan = 1.5*_N.pi
    elif y >= 0 and x > 0:#Q1
        atan = _N.arctan(y / x)
    elif y > 0 and x <= 0:#Q2
        atan = _N.arctan(y / x) + _N.pi
    elif y <= 0 and x < 0:#Q3
        atan = _N.arctan(y / x) + _N.pi
    elif y < 0 and x >= 0:
        atan = _N.arctan(y / x) + 2*_N.pi
    else:
        atan = -100
    return atan

def roots2ARcoeff(rs):
    if type(rs) == list:
        k = len(rs)
        rs= _N.array(rs)
    else:
        k = max(rs.shape)    #  1-d or n-d array
        rs= rs.reshape(1, k)
    A  = _N.zeros((k, k),  dtype=rs.dtype)
    B  = _N.empty((k, 1),  dtype=rs.dtype)
    F  = _N.empty((k, 1),  dtype=rs.dtype)

    for c in xrange(k-1):     #  columns
        A[:, c] = rs**(k-c-1)
    A[:, k-1] = 1
    B[:, 0] = rs**k

    try:
        F = _N.linalg.solve(A, B)
    except _N.linalg.linalg.LinAlgError:
        print A, B
        F = _N.zeros((k, 1))
    return F.T.real

def generateValidAR(k):
    """
    r    = 2
    nImg = 2   #  we want oscill.
    nReal= 0

    while r < k-2:
        if k - r > 1:
            if _N.random.rand() < 0.5:
                nImg += 2
                r += 2
            else:
                nReal += 1
                r += 1
        else:
            nReal += 1
            r += 1
    """

    r    = 0
    nImg = 0   #  we want oscill.
    nReal= 0

    while r < k:
        if k - r > 1:
            if _N.random.rand() < 0.5:
                nImg += 2
                r += 2
            else:
                nReal += 1
                r += 1
        else:
            nReal += 1
            r += 1

#    amps = 0.6*_N.random.rand(nImg/2 + nReal) + 0.4
    amps = _N.random.rand(nImg/2 + nReal)

    dtyp = _N.float
    if nImg > 0:
        dtyp = _N.complex
        phzs = _N.random.rand(nImg/2) * 2 * _N.pi

    rs = _N.empty(k,       dtype=dtyp)
    A  = _N.empty((k, k),  dtype=dtyp)
    B  = _N.empty((k, 1),  dtype=dtyp)
    F  = _N.empty((k, 1),  dtype=dtyp)

    for n in xrange(nImg/2):
        ph        = phzs[n]
        rs[2*n]   = amps[n] * (_N.cos(ph) + 1j * _N.sin(ph))
        rs[2*n+1] = rs[2*n].real - 1j*rs[2*n].imag 
    for n in xrange(nReal):
        rs[nImg + n]   = amps[nImg/2 + n]

    for c in xrange(k-1):
        A[:, c] = rs**(k-c-1)
    A[:, k-1] = 1
    B[:, 0] = rs**k
    
    F = _N.linalg.solve(A, B)
    return F.real


def cmplxRoots_old(arC):
    N   = len(arC)   # polynomial degree       a_1 B + a_2 B^2
    A   = _N.zeros((N, N))
    bBdd = True
    iBdd = 1

    A[0, :] = arC
    _N.fill_diagonal(A[1:, 0:], 1)

    vals, vecs = _N.linalg.eig(A)
    vroots = _N.empty(N)

    #  stable when roots (eigenvalues) are within unit circle
    for roots in xrange(N):
        zR = 1 / vals[roots]
        vroots[roots] = (zR * zR.conj()).real
        if vroots[roots] < 1:
            bBdd = False
            iBdd = 0

    return bBdd, iBdd, vroots, vals

def ARevals(arC):
    #  The eigenvalues of the AR coefficient matrix
    #  The eigenvalues are reciprocals of the roots of characteristic 
    #  polynomial (using Mike West definition, Wikipedia has slightly
    #  different definition).

    N   = len(arC)   # polynomial degree       a_1 B + a_2 B^2
    A   = _N.zeros((N, N))
    bBdd = True
    iBdd = 1


    A[0, :] = arC
    _N.fill_diagonal(A[1:, 0:], 1)

    vals, vecs = _N.linalg.eig(A)
    mags     = _N.empty(N)

    #  stable when roots (eigenvalues) are within unit circle
    for ev in xrange(N):
        mags[ev] = _N.sqrt((vals[ev] * vals[ev].conj()).real)
        if mags[ev] > 1:    #  Opposite of Mike West
            bBdd = False
            iBdd = 0

    return bBdd, iBdd, mags, vals

def ARroots(arC):
    #  The roots of the characteristic polynomial
    #  CAUTION:   Wikipedia definition of characteristic polynomial
    #  differs from used by Mike West
    #  Wikipedia   CP = z^p - F_i z^(p-i)   
    #  Mike West   CP = 1   - F_i u^i
    #  They are inverses of each other:   z = 1/u  (and then mult by u-p)
    #  In Mike West, the mag. of roots >= 1 for stationarity

    N   = len(arC)   # polynomial degree       a_1 B + a_2 B^2
    arr = _N.empty(N+1)
#    arr[0:N] = arC[::-1]   #  For Wikipedia
#    arr[N]   = -1
    arr[1:N+1] = -1*arC
    arr[0]   = 1

    bBdd = True
    iBdd = 1
    roots = _N.polynomial.polynomial.polyroots(arr)
    mag     = _N.empty(N)
    #  stable when roots (eigenvalues) are within unit circle
    for r in xrange(N):
        mag[r] = _N.sqrt((roots[r] * roots[r].conj()).real)
        if mag[r] < 1:
            bBdd = False
            iBdd = 0
    return bBdd, iBdd, mag, roots

def good_prior_cov_for_AR(k):
    """
    #  for high k, relatively lower mag, higher neighbor correlation appears 
    to give reasonable AR coefficients
    """
    Mag=0.15 + 0.75/k
    cor=0.8*(1 - _N.exp(-0.3*k))
    CM = dcyCovMat(k, _N.linspace(Mag, Mag/k, k), cor)
    return CM

def diffuse_cov_for_AR(k):
    """
    #  for high k, relatively lower mag, higher neighbor correlation appears 
    to give reasonable AR coefficients
    """
    Mag=0.5
    cor=0.03
    CM = dcyCovMat(k, _N.linspace(Mag, Mag, k), cor)
    CM[k-1, k-1] = 0.2
    return CM

def dcyCovMat(N, diagMags, corr):
    """
    serial numbers with a decaying correlation
    N         NxN matrix
    diagMag   magnitude of the diagnals
    corr      cc of neighboring elements
    """
    covM = _N.empty((N, N))
    _N.fill_diagonal(covM, diagMags)
    for l in xrange(1, N):
        for n in xrange(N - l):
            covM[n, n+l] = (0.5*(diagMags[n]+diagMags[n+l]))*corr**l
            covM[n+l, n] = covM[n, n+l]

    return covM

def MetrSampF(N, k, smpx, q2, pr_uF, pr_cvF, trCv, burn=100, Fini=None):
    """
    sample F from full conditional given sample of latent and parameters
    N        size of latent state data
    smpx     sampled latent state
    pr_uF    prior mean for F
    pr_cvF   prior cov for F
    q2       transition noise
    From paper Wolfgang Polasek, Song Jin   "From Data to Knowledge"
    minacc   minimum number of new points accepted.  to make sure sampler
             is well-mixed.  burn alone may not be an acceptable way to
             ensure this

    prior dist of F
    likelihood
    """
    y   = smpx[1:N, 0].reshape(N-1, 1)    #  a column vector
    xnm = smpx[0:N-1, :]                  #  a column vector

    ipr_cvF= _N.linalg.inv(pr_cvF)        #  inverse prior cov.

    ### conditional posterior moments
    iCov   = ipr_cvF + _N.dot(xnm.T, xnm)/q2   # inv conditional posterior cov.
    Cov    = _N.linalg.inv(iCov)       # conditional posterior cov.
    M  = _N.dot(Cov, _N.dot(ipr_cvF, pr_uF) + (1/q2)*_N.dot(xnm.T, y)) # conditional posterior mean

    #   initial value of F
    if Fini == None:
        bBdd= False
        while not bBdd:
            F    = _N.random.multivariate_normal(pr_uF[:, 0], pr_cvF)  # sample from prior
            F    = F.reshape(k, 1)
            bBdd, iBdd, vr, evals = ARevals(F[:, 0])
    else:
        F  = _N.empty((k, 1))
        F[:, 0] = Fini

    FM  = F - M

    #  The conditional posterior is characterized by M and Cov
    aO  = -0.5*_N.dot(FM.T, _N.dot(iCov, FM))    #  arguments to exp

    rands = _N.random.rand(burn)

    acc =0
    n = 1
    while n < burn:
        Fn  = _N.random.multivariate_normal(F[:, 0], trCv)  #  sample from cond. post.
        Fn    = Fn.reshape(k, 1)
        FnM  = Fn - M
        aC  = -0.5*_N.dot(FnM.T, _N.dot(iCov, FnM))
        bBdd, iBdd, vr, evals = ARevals(Fn[:, 0])

        # if bBdd:   # quick hack
        #     r   = aC - aO
        #     if _N.log(rands[n]) < min(r, 0):
        #         F = Fn
        #         aO = aC
        # n += 1

        if bBdd and (evals.dtype==_N.complex):   # quick hack
            bLoFreq = False
            for i in xrange(k):
                azm = azim(evals[i].real, evals[i].imag)
                if (azm < 0.4):# and (azm < _N.pi):
                    bLoFreq = True
#                if (azm > 0.4) and (azm < _N.pi):
#                    bHiFreq = True

#            if bLoFreq and (not bHiFreq):
            if bLoFreq:
                r   = aC - aO
                if _N.log(rands[n]) < min(r, 0):
                    F = Fn
                    aO = aC
        n += 1

    return F[:,0]


def MetrSampF_egv(N, k, smpx, q2, sae, burn=100, Fini=None):
    """
    sample F from full conditional given sample of latent and parameters
    N        size of latent state data
    smpx     sampled latent state
    sae      sampAReig object
    q2       transition noise
    From paper Wolfgang Polasek, Song Jin   "From Data to Knowledge"
    minacc   minimum number of new points accepted.  to make sure sampler
             is well-mixed.  burn alone may not be an acceptable way to
             ensure this
    """
    y   = smpx[1:N, 0].reshape(N-1, 1)    #  a column vector
    xnm = smpx[0:N-1, :]                  #  a column vector

    ### conditional posterior moments
    iCov   = _N.dot(xnm.T, xnm)/q2     # inv conditional posterior cov.
    Cov    = _N.linalg.inv(iCov) 
    M      = _N.dot(Cov, _N.dot(xnm.T, y))/q2 # conditional posterior mean
    # print M
    # print iCov

    #   initial value of F
    if Fini == None:
#        F = generateValidAR(k)   #  returns a column vector
        F = sae.draw()   #  returns a column vector
    else:
        Fini = Fini.reshape((k, 1))
        F  = Fini

    FM  = F - M

    #  The Fn's being generated are not uniform in AR space
    #  This non-uniformity acts as a prior?
    aO  = -0.5*_N.dot(FM.T, _N.dot(iCov, FM))    #  arguments to exp


    rands = _N.random.rand(burn)

    for n in xrange(burn):
#        Fn  = generateValidAR(k)
        Fn  = sae.draw()
        FnM  = Fn - M

        aC  = -0.5*_N.dot(FnM.T, _N.dot(iCov, FnM))
        r   = _N.exp(aO - aC)    #  or compare aO - aC with 0
        if rands[n] < min(r, 1):
            F = Fn
            aO = aC

#     lrands = _N.log(_N.random.rand(burn))

#     for n in xrange(burn):
#         Fn  = generateValidAR(k)
#         FnM  = Fn - M

#         aC  = -0.5*_N.dot(FnM.T, _N.dot(iCov, FnM))
# #        r   = _N.exp(aO - aC)    #  or compare aO - aC with 0
#         lr   = aC - aO    #  or compare aO - aC with 0
# #        print "---    %(aC).3e   %(aO).3e    %(diff).3e" % {"aC" : aC, "aO" : aO, "diff" : (aO-aC)}
#         if lrands[n] < min(lr, 0):
#             F = Fn
#             aO = aC


    return F[:,0]

