import utilities as _U
import numpy as _N
import scipy.stats as _ss

def ampAngRep(z, sp="\n", f_order=False):
    #  if f_order=True, give me rank of each element (in ascending order)
    #  if f=3, 9, 1, 200 Hz  
    #  return me (2, 0, 1, 3)  _ss.rankdata
    prt = ""
    fs  = []
    amps= []
    
    if (type(z) == list) or (type(z) == _N.ndarray):
        L = len(z)

        for l in xrange(L):
            cStr  = "R"
            zv = z[l]
            r  = _N.abs(zv)
            ang= _N.angle(zv)
            if ang < 0:
                ang = 2*_N.pi + ang

            if (type(zv) == _N.complex) or (type(zv) == _N.complex64) or (type(zv) == _N.complex128) or (type(zv) == complex):
                cStr = "C"
                if zv.imag == 0:
                    cStr = "R"

            if ((cStr == "C") and (ang < _N.pi)) and (ang != 0):
                prt += "C  [%(r) .2f,  %(fa).3f]%(sp)s" % {"r" : r, "fa" : (ang / _N.pi), "t" : cStr, "sp" : sp}
                fs.append(ang / _N.pi)
                amps.append(r)
            elif cStr == "R":
                prt += "R  [%(r) .2f, ..]%(sp)s" % {"r" : zv.real, "t" : cStr, "sp" : sp}
    else:
        cStr  = "R"
        r  = _N.abs(z)
        ang= _N.angle(z)
        if ang < 0:
            ang = 2*_N.pi + ang
        if (type(zv) == _N.complex) or (type(zv) == _N.complex64) or (type(zv) == _N.complex128):
            cStr = "R"

        prt += "%(t)s  [%(r) .2f  %(fa).3f]%(sp)s" % {"r" : r, "fa" : (ang / _N.pi), "t" : cStr, "sp" : sp}
    if f_order == False:
        return prt[:-1]
    else:
        rks   = _N.array(_ss.rankdata(fs, method="ordinal") - 1, dtype=_N.int).tolist()   #  rank each element.  A=[3, 0, 1]  rks=[3, 1, 2]
        indOrFs= [rks.index(x) for x in range(len(fs))]
        return prt[:-1], indOrFs, _N.array(fs), _N.array(amps)


"""
def initF(nR, nCS, nCN, cohrtSlow=False):
    #  random AR coeff set.  nR real roots and nCPr cmplx pairs
    nCPr = nCS + nCN

    if nCPr == 0:
        iRs = _N.empty(nR + 2*nCPr, dtype=_N.float64)    # inverse roots
    else:
        iRs = _N.empty(nR + 2*nCPr, dtype=_N.complex)    # inverse roots

    iRs[0:nR] = -_N.random.rand(nR)

    dth   = .1 / nCS

    for n in xrange(nCPr):
        if n < nCS:
            th = _N.pi*dth * (n+1)
        else:
            th = _N.pi*(0.4*_N.random.rand() + 0.4)
        if cohrtSlow and n < nCS:
            r  = 0.9
        else:
            r  = 0.4 + 0.4 * _N.random.rand()
        iRs[nR + 2*n]     = r*(_N.cos(th) + _N.sin(th)*1j)
        iRs[nR + 2*n + 1] = r*(_N.cos(th) - _N.sin(th)*1j)

    return iRs
"""

def initF(nR, nCS, nCN, ifs=None):
    #  random AR coeff set.  nR real roots and nCPr cmplx pairs
    nCPr = nCS + nCN

    if nCPr == 0:
        iRs = _N.empty(nR + 2*nCPr, dtype=_N.float64)    # inverse roots
    else:
        iRs = _N.empty(nR + 2*nCPr, dtype=_N.complex)    # inverse roots

    #iRs[0:nR] = -0.6 - 0.4*_N.random.rand(nR)
    #iRs[0:nR] = 0.1*_N.random.rand(nR)
    iRs[0:nR] = -0.4   #  negative value here helps AR(3) when using modulus prior
    #iRs[0:nR] = 0.4   #  was good for AR(p > 3) when using modulus prior

    if (ifs == None):
        ifs  = []
        for n in xrange(nCS):
            ifs.append(0.03 * (n + 1) * _N.pi)

    for n in xrange(nCPr):
        if n < nCS:
            print ifs[n]
            th = ifs[n]
            r  = 0.97
        else:  #  Looking at how noise roots distribute themselves
               #  they seem to spread out fairly evenly in spectrum
            th = _N.pi*(0.15 + 0.8 / (nCN - (n - nCS)))   #  above 80Hz, weak
            r  = (1 + _N.random.rand()) / (2 * _N.sqrt(nCN))   #  r ~ (1 / (2*nCN))
            #r  = 1.5 / (2 * _N.sqrt(nCN))   #  r ~ (1 / (2*nCN))
            #r  = 0.5+0.3*_N.random.rand()
            #r   = 0.8
        iRs[nR + 2*n]     = r*(_N.cos(th) + _N.sin(th)*1j)
        iRs[nR + 2*n + 1] = r*(_N.cos(th) - _N.sin(th)*1j)
    print iRs

    return iRs

def initFL(nR, nCS, nCN, ifs=None):
    #  random AR coeff set.  nR real roots and nCPr cmplx pairs
    nCPr = nCS + nCN

    if nCPr == 0:
        iRs = _N.empty(nR + 2*nCPr, dtype=_N.float64)    # inverse roots
    else:
        iRs = _N.empty(nR + 2*nCPr, dtype=_N.complex)    # inverse roots

    #iRs[0:nR] = -0.6 - 0.4*_N.random.rand(nR)
    #iRs[0:nR] = 0.1*_N.random.rand(nR)
    iRs[0:nR] = 0.0

    if (ifs == None):
        ifs  = []
        for n in xrange(nCS):
            ifs.append(0.03 * (n + 1) * _N.pi)

    for n in xrange(nCPr):
        if n < nCS:
            th = ifs[n]
            r  = 0.95
        else:  #  Looking at how noise roots distribute themselves
               #  they seem to spread out fairly evenly in spectrum
            th = _N.pi*(0.6 + 0.3 / (nCN - (n - nCS)))   #  above 80Hz, weak
            #r  = (1 + _N.random.rand()) / (2 * _N.sqrt(nCN))   #  r ~ (1 / (2*nCN))
            r  = 1.5 / (2 * _N.sqrt(nCN))   #  r ~ (1 / (2*nCN))
            #r  = 0.3 + 0.3*_N.random.rand()
        iRs[nR + 2*n]     = r*(_N.cos(th) + _N.sin(th)*1j)
        iRs[nR + 2*n + 1] = r*(_N.cos(th) - _N.sin(th)*1j)

    return iRs

def FfromLims(nR, Cs, Cn, AR2lims):
    #  random AR coeff set.  nR real roots and nCPr cmplx pairs
    nCPr = Cs + Cn
    if nCPr == 0:
        iRs = _N.empty(nR + 2*nCPr, dtype=_N.float64)    # inverse roots
    else:
        iRs = _N.empty(nR + 2*nCPr, dtype=_N.complex)    # inverse roots

    iRs[0:nR] = -_N.random.rand(nR)
    #kk = 0.002
    #  50Hz = 0.1    k x Hz    k is 0.002
    for n in xrange(nCPr):
        lo = AR2lims[n, 0]
        hi = AR2lims[n, 1]
        if n < Cs:  #  noize
            r  = 0.8
        else:
            r  = 0.5
        th = lo + (hi - lo)*_N.random.rand()

        iRs[nR + 2*n]     = r*(_N.cos(th) + _N.sin(th)*1j)
        iRs[nR + 2*n + 1] = r*(_N.cos(th) - _N.sin(th)*1j)

    return iRs

def dcmpcff(alfa):
    p   = len(alfa)

    R   = p
    C   = 0

    dtyp    = _N.float64
    if alfa.dtype == _N.complex128:
        dtyp = _N.complex128
    A       = _N.zeros((p, p), dtype=dtyp)
    AA      = _N.zeros((p, p), dtype=dtyp)
    D       = _N.ones((p, 1),  dtype=dtyp)

    if dtyp == _N.complex128:
        R    = 0
        i    = p - 1
        while i > -1:
            if alfa[i].imag != 0:
                C += 1
                i -= 2
            else:
                R += 1
                i -= 1

    bbc = _N.empty(p, dtype=_N.complex128)
    for m in xrange(p):
        AA[m, m] = 1
        for j in xrange(p):
            if j != m:
                AA[m, m] *= 1 - alfa[j]/alfa[m]
        bbc[m] = 1 / AA[m, m]

    b   = None
    c   = None

    if R > 0:
        b      = _N.empty(R)
        b[:]   = bbc[0:R].real
    if C > 0:
        c   = _N.empty(2*C, dtype=_N.complex128)
        c[:]   = bbc[R:R+2*C]

    return b, c

def betterProposal(J, Ji, U):
    """
    Move the proposal nearer to 
    """
    a          = 0.25*Ji[1, 1]
    b          = -1.5*Ji[0, 1]
    c          = 2*Ji[0, 0] + U[0, 0]*Ji[0, 1] + U[1, 0]*Ji[1, 1]
    d          = -2 * (U[0, 0]*Ji[0, 0] + U[1, 0]*Ji[0, 1])
    if a != 0:
        roots      = _U.cubicRoots(a, b, c, d)
    else:
        roots      = [(-c + _N.sqrt(c*c - 4*b*d)) / (2*b), \
                      (-c - _N.sqrt(c*c - 4*b*d)) / (2*b)]

    rM1  = None
    for r in roots:
        if (r.imag == 0) and (r >= -2) and (r <= 2):
            rM1 = r
    rM2  = -0.25*rM1*rM1

    return rM1, rM2

def buildLims(Cn, freq_lims, nzLimL=90.):
    Ns   = len(freq_lims)   #  # of signal components
    radians = _N.zeros((Ns + Cn, 2))
    twpi = 2*_N.pi

    for ns in xrange(Ns):
        wlL = 1000. / freq_lims[ns][0]     #  wavelength of low frequency
        wlH = 1000. / freq_lims[ns][1]
        p1a =  twpi/wlH    # 2pi / lamba
        p1b =  twpi/wlL

        radians[ns, 0] = p1a; radians[ns, 1] = p1b
    for ns in xrange(Cn):
        wlL = 1000. / nzLimL     #  "noise" from 90Hz
        wlH = 1000. / 500.
        p1a =  twpi/wlH
        p1b =  twpi/wlL

        radians[Ns+ns, 0] = p1a; radians[Ns+ns, 1] = p1b

    return radians
