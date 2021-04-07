from LOST.LOSTdirs import resFN
import numpy as _N
from LOST.ARcfSmplFuncs import ampAngRep, dcmpcff#betterProposal
import matplotlib.pyplot as _plt
from LOST.filter import gauKer
import scipy.stats as _ss
import pickle
import numpy.polynomial.polynomial as _Npp

def createFlucOsc(f0, f0VAR, N, dt0, TR, Bf=[0.99], Ba=[0.99], amp=1, amp_nz=0, stdf=None, stda=None, sig=0.1, smoothKer=0, dSF=5, dSA=5):
    """
    AR as a generative model creates oscillations where amplitude is 
    sometimes small - to the point of calling it an oscillation is not quite 
    dSA, dSF    modulations created like (AR / (dSX x stddev)).
    """
    out = _N.empty((TR, N))

    if stdf == None:
        x, y = createDataAR(100000, Bf, sig, sig)
        stdf  = _N.std(x)   #  choice of 4 std devs to keep phase monotonically increasing
    if stda == None:
        x, y = createDataAR(100000, Ba, sig, sig)
        stda  = _N.std(x)   #  choice of 4 std devs to keep phase monotonically increasing

    trm = 500
    dts  = _N.empty(N-1)

    t    = _N.empty(N)
    t[0] = 0

    if smoothKer > 0:
        gk = gauKer(smoothKer)
    for tr in range(TR):
        ph0 = _N.random.rand() * 2*_N.pi
        bGood = False
        while not bGood:
            f, dum = createDataAR(N+trm, Bf, sig, sig, trim=trm)

            fn   = 1 + (f / (dSF*stdf))  #  going above 4 stdf very rare.  |xn| is almost < 1

            bGood = True
            lt0s = _N.where(fn < 0)
            print("fn < 0 at %d places" % len(lt0s[0]))

            #if _N.min(fn) > 0:
            #    bGood = True

        for i in range(1, N):
            dt = dt0 * fn[i]
            t[i] = t[i-1] + dt
            dts[i-1] = dt

        
        y = _N.sin(2*_N.pi*(f0 + f0VAR[tr])*t + ph0)

        bGood = False
        while not bGood:
            a, dum = createDataAR(N+trm, Ba, sig, sig, trim=trm)
            an   = a / (dSA*stda)   #  in limit of large N, std(xn) = 1

            AM   = 1 + an  #  if we get fluctuations 2 stds bigger, 
            bGood = True
            lt0s = _N.where(AM < 0)
            print("AM < 0 at %d places" % len(lt0s[0]))
            #if _N.min(AM) > 0:
            #    bGood = True

        if smoothKer > 0:
            out[tr] = _N.convolve(y*AM*(amp+_N.random.randn()*amp_nz),  gk, mode="same")
        else:
            out[tr] = y*AM*amp

    return out

def createOscPacket(f0, f0VAR, N, dt0, TR, tM0, tS0, tJ=0, Bf=[0.99], amp=1, stdf=None, stda=None, sig=0.1, smoothKer=0):
    """
    Generate a wave packet.  Amplitude envelope is trapezoidal
    """
    out = _N.empty((TR, N))

    if stdf == None:
        x, y = createDataAR(100000, Bf, sig, sig)
        stdf  = _N.std(x)   #  choice of 4 std devs to keep phase monotonically increasing

    trm = 100
    dts  = _N.empty(N-1)

    t    = _N.empty(N)
    t[0] = 0

    gkC = gauKer(int(tS0*0.2))  #  slight envelope
    if smoothKer > 0:
        gk = gauKer(smoothKer)

    AM = _N.empty(N)

    for tr in range(TR):
        ph0 = _N.random.rand() * 2*_N.pi

        tM = tM0 + int(tJ*_N.random.randn())
        tS = tS0 + int(tJ*_N.random.randn())
        bGood = False
        while not bGood:
            f, dum = createDataAR(N+trm, Bf, sig, sig, trim=trm)

            fn   = 1 + (f / (3*stdf))  #  going above 4 stdf very rare.  |xn| is almost < 1

            if _N.min(fn) > 0:
                bGood = True

        for i in range(1, N):
            dt = dt0 * fn[i]
            t[i] = t[i-1] + dt
            dts[i-1] = dt

        
        y = _N.sin(2*_N.pi*(f0 + f0VAR[tr])*t + ph0)

        t0 = tM - tS if (tM - tS > 0) else 0
        t1 = tM + tS if (tM + tS < N) else N
        AM[0:t0]     = 0.15 + 0.03*_N.random.randn(t0)
        AM[t0:t1] = 1 + 0.2*_N.random.randn(t1-t0)
        AM[t1:]      = 0.15 + 0.03*_N.random.randn(N-t1)
        AMC = _N.convolve(AM, gkC, mode="same")

        if smoothKer > 0:
            out[tr] = _N.convolve(y*AMC*amp, gk, mode="same")
        else:
            out[tr] = y*AMC*amp

    return out

def createModEnvelopes(TR, N, t0, t1, jitterTiming, jitterLength, gkW=0, weak=0.1):
    """
    Generate a wave packet.  Amplitude envelope is trapezoidal
    """
    out = _N.empty((TR, N))

    if gkW > 0:
        gk = gauKer(gkW)

    AM = _N.empty(N)

    for tr in range(TR):
        w    = int(((t1-t0) + jitterLength * _N.random.randn())/2.)
        m    = int(0.5*(t1+t0) + jitterTiming * _N.random.randn())

        if m-w > 0:
            AM[m-w:m+w] = 1
            AM[0:m-w] = weak
        else:
            AM[0:m+w] = 1

        AM[m+w:]    = weak
        if gkW > 0:
            out[tr] = _N.convolve(AM, gk, mode="same")
        else:
            out[tr] = AM
        out[tr] /= _N.max(out[tr])

    return out

def createUpDn(TR, N, minDurL, lmdL, minDurH, lmdH, lowV=-1, upM=1, pUp=0.5):
    """
    upM will take time period and multiply if an upstate
    """
    updn = _N.empty((TR, N))
    szL   = 10*N*TR/lmdL
    szH   = 10*N*TR/lmdH
    rvL = _ss.expon.rvs(scale=lmdL, size=szL)
    rvH = _ss.expon.rvs(scale=lmdH, size=szH)
    fltrdL     = rvL[rvL > minDurL]
    fltrdH     = rvH[rvH > minDurH]
    ir = 0   #  random exponential
    vals  = _N.array([0, lowV])
    for tr in range(TR):
        iS = 0 if (_N.random.rand() < 0.5) else 1
        t = 0

        while t < N:
            m = 1
            if iS == 0:
                m = upM
                updn[tr, t:t+int(fltrdH[ir]*m)] = vals[iS]
                t += int(fltrdH[ir]*m)
            else:
                updn[tr, t:t+int(fltrdL[ir]*m)] = vals[iS]
                t += int(fltrdL[ir]*m)
            iS = 1 - iS
            ir += 1
    return updn
"""
def createDataAR(N, B, err, obsnz, trim=0):
    #  a[1]^2 + 4a[0]
    #  B[0] = -0.45
    #  B[1] =  0.9
    err = _N.sqrt(err)
    obsnz  = _N.sqrt(obsnz)
    p = len(B)

    x    = _N.empty(N)
    y    = _N.empty(N)

    #  initial few
    for i in range(p+1):
        x[i] = err*_N.random.randn()
        y[i] = obsnz*_N.random.randn()

    for i in range(p+1, N):
        x[i] = _N.dot(B, x[i-1:i-p-1:-1]) + err*_N.random.randn()
        #  y = Hx + w   where w is a zero-mean Gaussian with cov. matrix R.
        #  In our case, H is 1 and R is 1x1
        y[i] = x[i] + obsnz*_N.random.randn()

    return x[trim:N], y[trim:N]
"""
def createDataAR(N, B, err, obsnz, trim=0):
    #  a[1]^2 + 4a[0]
    #  B[0] = -0.45
    #  B[1] =  0.9
    err = _N.sqrt(err)
    obsnz  = _N.sqrt(obsnz)
    p = len(B)
    BB = _N.array(B[::-1])   #  B backwards

    x    = _N.empty(N)
    y    = _N.empty(N)

    #  initial few
    for i in range(p+1):
        x[i] = err*_N.random.randn()
        y[i] = obsnz*_N.random.randn()

    nzs = err*_N.random.randn(N)
    for i in range(p+1, N):
        #x[i] = _N.dot(B, x[i-1:i-p-1:-1]) + nzs[i]
        x[i] = _N.dot(BB, x[i-p:i]) + nzs[i]
        #  y = Hx + w   where w is a zero-mean Gaussian with cov. matrix R.
        #  In our case, H is 1 and R is 1x1
    onzs = obsnz*_N.random.randn(N)
    y[p+1:N] = x[p+1:N] + onzs[p+1:N]

    return x[trim:N], y[trim:N]

def createDataPP(N, B, beta, u, stNz, p=1, trim=0, x=None, absrefr=0):
    #  a[1]^2 + 4a[0]
    #  B[0] = -0.45
    #  B[1] =  0.9
    if type(u) != _N.ndarray:
        u = _N.ones(N) * u
    if x is None:
        k = len(B)
        stNz = _N.sqrt(stNz)

        rands = _N.random.randn(N)
        x    = _N.empty(N)
        for i in range(k+1):
            x[i] = stNz*rands[i]
        for i in range(k+1, N):
            #  for k = 2, x[i] = B[0]*x[i-2], B[1]*x[i - 1]
            #  B[0]   is the weight of oldest time point
            #  B[k-1] is weight of most recent time point
            #        x[i] = _N.dot(B, x[i-k:i]) + err*_N.random.randn()
            x[i] = _N.dot(B, x[i-1:i-k-1:-1]) + stNz*rands[i]
    else:
        k = 0

    spks = _N.zeros(N)
    prbs = _N.zeros(N)
    fs   = _N.zeros(N)

    #  initial few

    beta0 = beta[0]
    lspk  = -2*absrefr
    for i in range(k, N):
        e = _N.exp(u[i] + beta0* x[i]) * dt
        prbs[i]  = (p*e) / (1 + e)
        spks[i] = _N.random.binomial(1, prbs[i])
        if spks[i] == 1:
            if i - lspk <= absrefr:
                spks[i] = 0
            else:
                lspk = i

    fs[:] = prbs / dt

    return x[trim:N], spks[trim:N], prbs[trim:N], fs[trim:N]

def createDataPPl2(TR, N, dt, B, wgts, u_psth, innovar=0.01, lambda2=None, nRhythms=1, p=1, x=None, offset=None, cs=1, etme=None):
    """
    B:  AR coefficeints
    wgts:  weight for each component   (will be summed)
    """
    beta = _N.array([1., 0.])
    #  a[1]^2 + 4a[0]
    #  B[0] = -0.45
    #  B[1] =  0.9
    if type(u_psth) != _N.ndarray:
        u_psth = _N.ones(N) * u_psth
    if etme is None:
        etme = _N.ones(N)

    buf  = 500
    sqrtq2 = _N.sqrt(innovar)
    #if (stNz is not None) and (_N.sum(stNz) == 0):
    #xc   = _N.zeros((nRhythms, N+buf))   #  components
    x = None#_N.zeros(N+buf)
    #else:
    if x is None:
        xc   = _N.empty((nRhythms, N+buf))   #  components
        for nr in range(nRhythms):
            k = len(B[nr])
            #sstNz = _N.sqrt(stNz[nr])

            rands = _N.random.randn(N+buf)

            #  B[k-1]*xc[i-k] + B[k-2]*xc[i-k+1] + ... B[0] * xc[i-1]
            for i in range(k+1):
                xc[nr, i] = sqrtq2*rands[i]
            for i in range(k, N+buf):
                #  for k = 2, x[i] = B[0]*x[i-2], B[1]*x[i - 1]
                #  B[0]   is the weight of oldest time point
                #  B[k-1] is weight of most recent time point
                #        x[i] = _N.dot(B, x[i-k:i]) + err*_N.random.randn()
                #xc[nr, i] = _N.dot(B[nr], xc[nr, i-1:i-k-1:-1]) + sstNz*rands[i]
                xc[nr, i] = _N.dot(B[nr], xc[nr, i-k:i][::-1]) + sqrtq2*rands[i]
            xc[nr] *= wgts[nr]
    else:
        buf  = 0
        xc   = x

    if nRhythms > 1:
        x = _N.sum(xc, axis=0)     #  collapse
    else:
        x = xc.reshape(N+buf, )
        
    spks = _N.zeros(N)
    prbs = _N.zeros(N)
    latst = _N.zeros(N)
    prbsWithHist = _N.zeros(N)
    prbsNOsc = _N.zeros(N)   #  no osc.
    fs   = _N.zeros(N)

    #  initial few

    lh = 0
    if lambda2 is not None:
        lh    = len(lambda2)
    else:
        lh = 1
        lambda2 = _N.ones(1)
    #lh    = 300   #  at most 2000
    ht    = -int(_N.random.rand() * 50)

    for i in range(N):
        e = _N.exp(u_psth[i] + offset[i] + cs * etme[i] * x[i+buf]) * dt
        #print(u_psth[i] + offset[i] + cs * etme[i] * x[i+buf])
        prbs[i] = (p*e) / (1 + e)
        e = _N.exp(u_psth[i]) * dt
        prbsNOsc[i]  = (p*e) / (1 + e)

        lmd = 1 if i - ht >= lh else lambda2[i - ht]
        
        prbsWithHist[i] = prbs[i] * lmd
        prbsWithHist[i] = prbsWithHist[i] if prbsWithHist[i] < 1 else 1
        
        prbsNOsc[i] *= lmd
        spks[i] = _N.random.binomial(1, prbsWithHist[i])
        if spks[i] == 1:
            ht = i+1    #  lambda2[0] is history 1 bin after spike
        latst[i] = x[i+buf]

    fs[:] = prbsWithHist / dt
    return xc[:, buf:], spks, latst, prbsWithHist, fs, prbsNOsc

"""
def createDataPPl2(TR, N, dt, B, u, stNz, lambda2, nRhythms=1, p=1, x=None, offset=None, cs=1, etme=None):
    beta = _N.array([1., 0.])
    #  a[1]^2 + 4a[0]
    #  B[0] = -0.45
    #  B[1] =  0.9
    if type(u) != _N.ndarray:
        u = _N.ones(N) * u
    if etme is None:
        etme = _N.ones(N)

    buf  = 500
    if _N.sum(stNz) == 0:
        xc   = _N.zeros((nRhythms, N+buf))   #  components
        x = _N.zeros(N+buf)
    else:
        if x == None:
            xc   = _N.empty((nRhythms, N+buf))   #  components
            for nr in range(nRhythms):
                k = len(B[nr])
                sstNz = _N.sqrt(stNz[nr])

                rands = _N.random.randn(N+buf)

                for i in range(k+1):
                    xc[nr, i] = sstNz*rands[i]
                for i in range(k+1, N+buf):
                    #  for k = 2, x[i] = B[0]*x[i-2], B[1]*x[i - 1]
                    #  B[0]   is the weight of oldest time point
                    #  B[k-1] is weight of most recent time point
                    #        x[i] = _N.dot(B, x[i-k:i]) + err*_N.random.randn()
                    xc[nr, i] = _N.dot(B[nr], xc[nr, i-1:i-k-1:-1]) + sstNz*rands[i]
        else:
            buf  = 0
            xc   = x

        if nRhythms > 1:
            x = _N.sum(xc, axis=0)     #  collapse
        else:
            x = xc.reshape(N+buf, )
        
    spks = _N.zeros(N)
    prbs = _N.zeros(N)
    prbsNOsc = _N.zeros(N)   #  no osc.
    fs   = _N.zeros(N)

    #  initial few

    lh    = len(lambda2)
    #lh    = 300   #  at most 2000
    hst  = []    #  spikes whose history is still felt

    for i in range(N):
        e = _N.exp(u[i] + cs * etme[i] * x[i+buf]) * dt
        prbs[i]  = (p*e) / (1 + e)
        e = _N.exp(u[i]) * dt
        prbsNOsc[i]  = (p*e) / (1 + e)

        L  = len(hst)
        lmbd = 1

        for j in range(L - 1, -1, -1):
            ht = hst[j]
            #  if i == 10, ht == 9, lh == 1
            #  10 - 9 -1 == 0  < 1.   Still efective
            #  11 - 9 -1 == 1         No longer effective
            if i - ht - 1 < lh:
                lmbd *= lambda2[i - ht - 1]
            else:
                hst.pop(j)
        prbs[i] *= lmbd
        prbsNOsc[i] *= lmbd
        spks[i] = _N.random.binomial(1, prbs[i])
        if spks[i] == 1:
            hst.append(i)

    fs[:] = prbs / dt
    return xc[:, buf:], spks, prbs, fs, prbsNOsc
"""

def createDataPPl2Simp(TR, N, dt, B, u, stNz, lambda2, nRhythms=1, p=1, x=None, offset=None, cs=1):
    beta = _N.array([1., 0.])
    #  a[1]^2 + 4a[0]
    #  B[0] = -0.45
    #  B[1] =  0.9
    if type(u) != _N.ndarray:
        u = _N.ones(N) * u

    buf  = 100
    if _N.sum(stNz) == 0:
        xc   = _N.zeros((nRhythms, N+buf))   #  components
        x = _N.zeros(N+buf)
    else:
        if x == None:
            xc   = _N.empty((nRhythms, N+buf))   #  components
            for nr in range(nRhythms):
                k = len(B[nr])
                sstNz = _N.sqrt(stNz[nr])

                rands = _N.random.randn(N+buf)

                for i in range(k+1):
                    xc[nr, i] = sstNz*rands[i]
                for i in range(k+1, N+buf):
                    #  for k = 2, x[i] = B[0]*x[i-2], B[1]*x[i - 1]
                    #  B[0]   is the weight of oldest time point
                    #  B[k-1] is weight of most recent time point
                    #        x[i] = _N.dot(B, x[i-k:i]) + err*_N.random.randn()
                    xc[nr, i] = _N.dot(B[nr], xc[nr, i-1:i-k-1:-1]) + sstNz*rands[i]
        else:
            buf  = 0
            xc   = x

        if nRhythms > 1:
            x = _N.sum(xc, axis=0)     #  collapse
        else:
            x = xc.reshape(N+buf, )
        
    spks = _N.zeros(N)
    prbs = _N.zeros(N)
    fs   = _N.zeros(N)

    #  initial few

    lh    = len(lambda2)

    #lh    = 300   #  at most 2000
    hst  = []    #  spikes whose history is still felt

    ls     = -int(_N.random.rand()*20)
    for i in range(N):
        e = _N.exp(u[i] + cs * x[i+buf]) * dt
        prbs[i]  = (p*e) / (1 + e)
        prbsWithHist[i] = prbs[i]*lambda2[i-ls-1]
        try:
            spks[i] = _N.random.binomial(1, prbs[i]*lambda2[i-ls-1])
        except IndexError:
            print("i  %(i)d   ls  %(ls)d     i-ls-1  %(ils1)d" % {"i" : i, "ls" : ls, "ils1" : (i-ls-1)})
        if spks[i] == 1:
            ls = i

    fs[:] = prbs / dt
    return xc[:, buf:], spks, prbs, prbsWithHist, fs

#def plottableSpkTms(dN, ymin, ymax):
def plottableSpkTms(dN, y):
    #  for each spike time,
    ts = []
    N  = len(dN)
    for n in range(N):
        if dN[n] == 1:
            ts.append(n)

    x_ticks = []
    y_ticks = []
    for t in ts:
        x_ticks.append(t)
        y_ticks.append(y)
#        x_ticks.append([t, t])
#        y_ticks.append([ymin, ymax])
    return x_ticks, y_ticks

def saveset(name, noparam=False):
    #  u, B, singleFreqAR, dt, stNz, x, dN, prbs
    xprbsdN = _N.empty((N + 1, 3))
    xprbsdN[:, 0] = x[:]
    xprbsdN[:, 1] = prbs[:]
    xprbsdN[:, 2] = dN[:]

    _N.savetxt(resFN("xprbsdN.dat", dir=name, create=True), xprbsdN, fmt="%.5e")

    if not noparam:
        fp = open(resFN("params.py", dir=name, create=True), "w")
        fp.write("u=%.3f\n" % u)
        fp.write("beta=%s\n" % arrstr(beta))
        fp.write("ARcoeff=_N.array(%s)\n" % str(ARcoeff))
        fp.write("alfa=_N.array(%s)\n" % str(alfa))
        fp.write("#  ampAngRep=%s\n" % ampAngRep(alfa))
        fp.write("dt=%.2e\n" % dt)
        fp.write("stNz=%.3e\n" % stNz)
        fp.write("absrefr=%d\n" % absrefr)
        fp.close()

def savesetMT(TR, spkdat, gtdat, model, lambda2, psth, outdir, AR_components=None):
    #  u, B, singleFreqAR, dt, stNz, x, dN, prbs
    fmt = ""
    fmtgt = ""
    datfn = "spk_dat"
    if model != "bernoulli":
        #fmt += "% .2e "
        fmt += "%d "
        fmt *= TR

    _N.savetxt("%(od)s/%(dfn)s.dat" % {"od" : outdir, "dfn" : datfn}, spkdat, fmt=("% d" * TR))
    pcklme = {}
    pcklme["gtdatWOhist"] = gtdat[:, ::2]
    pcklme["gtdatWhist"]  = gtdat[:, 1::2]
    pcklme["spkdat"]      = spkdat
    pcklme["psth"]        = psth
    pcklme["lambda2"]     = lambda2
    if AR_components is not None:
        pcklme["AR_components"]     = AR_components
    dmp = open("%(od)s/generative.pkl" % {"od" : outdir}, "wb")
    pickle.dump(pcklme, dmp, -1)
    dmp.close()


def savesetMTnosc(TR, alldat, name):   #  also save PSTH
    #  u, B, singleFreqAR, dt, stNz, x, dN, prbs
    bfn = "cnt_data"
    bfn = "noscp"
    fmt = "% .2e "
    fmt *= TR

    _N.savetxt(resFN("%s.dat" % bfn, dir=name, create=True), alldat, fmt=fmt)

def savesetARpn(name):
    #  u, B, singleFreqAR, dt, stNz, x, dN, prbs
    xxlxhy = _N.empty((N + 1, 4))
    xxlxhy[:, 0] = x[:]
    xxlxhy[:, 1] = loAR[:]
    xxlxhy[:, 2] = hiAR[:]
    xxlxhy[:, 3] = y[:]

    _N.savetxt(datFN("xxlxhy.dat", dir=name, create=True), xxlxhy, fmt="%.5e")

    fp = open(resFN("params.py", dir=name, create=True), "w")
    fp.write("arLO=%s\n" % str(arLO))   #  want to keep these as list
    fp.write("arHI=%s\n" % str(arHI))
    fp.write("dt=%.2e\n" % dt)
    fp.write("stNzL=%.3e\n" % stNzL)
    fp.write("stNzH=%.3e\n" % stNzH)
    fp.write("obsNz=%.3e\n" % obsNz)
    fp.write("H=%s\n" % arrstr(H))
    fp.close()

def arrstr(_arr):
    dim1 = 0
    if type(_arr) == list:
        dim1 = 1
        arr = _N.array(_arr)
    else:
        arr = _arr
    if len(arr.shape) == 1:   #  assume it is a row vector
        dim1 = 1
        cols = arr.shape[0]
        arrR = arr.reshape((1, cols))
    else:
        arrR = arr

    if dim1 == 0:
        strg = "_N.array(["
    else:
        strg = "_N.array("

    c = 0
    for r in range(arrR.shape[0] - 1):
        strg += "["
        for c in range(arrR.shape[1] - 1):
            strg += ("% .6e" % arrR[r, c]) + ", "
        strg += ("% .6e" % arrR[r, c]) + "], \n"

    #  for last row (or first, if only 1 row)
    r = arrR.shape[0] - 1
    strg += "["

    c = 0
    for c in range(arrR.shape[1] - 1):
        strg += ("% .6e" % arrR[r, c]) + ", "
    strg += ("% .6e" % arrR[r, arrR.shape[1] - 1]) + "]"
    if dim1 == 0:
        strg += "])"
    else:
        strg += ")"

    return strg

def quickPSTH(alldat, TR, COLS, plot=False, fn=None, dt=0.001):
    #  dt for bin size of spikes
    spks = []
    N    = alldat.shape[0]
    
    for tr in range(TR):
        if COLS == 4:
            spks.extend(_N.where(alldat[:, COLS-2+tr*COLS] == 1)[0])
        elif COLS == 3:
            spks.extend(_N.where(alldat[:, COLS-1+tr*COLS] == 1)[0])

    mult = 0.01 * TR
    psthBIN = N / 100
    if plot and (fn != None):
        fig = _plt.figure(figsize=(9, 4))
        #_plt.hist(spks, bins=_N.linspace(0, N, 100), color="black")
        hist, bins = _N.histogram(spks, bins=_N.linspace(0, N, 100))
        _plt.plot(bins[0:-1], hist / (psthBIN*dt*TR))
        
        _plt.xlim(0, N)
        #_plt.xticks(_N.arange(0, 1001, 200), _N.arange(0, 1.01, 0.2), fontsize=26)

        #_plt.yticks(_N.arange(40, 161, 40), _N.arange(0, 161/mult, 40/mult), fontsize=26)
        #  N counts in bin 1 bin of 10msxTR  
        _plt.xlabel("seconds", fontsize=30)
        _plt.ylabel("Hz", fontsize=30)
        fig.subplots_adjust(bottom=0.2, top=0.85, left=0.15, right=0.9)
        _plt.savefig(fn)
        _plt.close()

    return spks

def disjointSubset(_superSet, subSetA):
    #  Give me subSetB that is a disjoint subset of superSet and subSetA
    if type(_superSet) is _N.ndarray:
        superSet = _superSet.tolist()
    else:
        superSet = _superSet
    for i in range(len(subSetA)):
        try:
            superSet.pop(superSet.index(subSetA[i]))
        except ValueError:
            print("Warning 2nd set is not a subset of the superset")
            pass
    return list(superSet)

def ISIs(dat, trials=None, t0=0, t1=None):
    isis = []
    if trials is None:
        TR = dat.shape[0]
        trials = _N.arange(TR)
    if t1 is None:
        t1 = dat.shape[1]

    for tr in trials:
        spkts = _N.where(dat[tr, t0:t1] == 1)[0]
        isi   = _N.diff(spkts)
        isis.extend(isi)

    return _N.array(isis)

def findMode(dmp, setname, burn, NMC, startIt=None, NB=20, NeighB=1, dir=None):
    smp_u  = dmp["u"]
    smp_aS = dmp["aS"]
    fs     = dmp["fs"]
    amps   = dmp["amps"]
    amps   = dmp["amps"]
    startIt = burn if startIt == None else startIt
    aus = _N.mean(smp_u[:, startIt:], axis=1)
    aSs = _N.mean(smp_aS[startIt:], axis=0)

    L   = burn + NMC - startIt

    hist, bins = _N.histogram(fs[startIt:, 0], _N.linspace(_N.min(fs[startIt:, 0]), _N.max(fs[startIt:, 0]), NB))
    indMfs =  _N.where(hist == _N.max(hist))[0][0]
    indMfsL =  max(indMfs - NeighB, 0)
    indMfsH =  min(indMfs + NeighB+1, NB-1)
    loF, hiF = bins[indMfsL], bins[indMfsH]

    hist, bins = _N.histogram(amps[startIt:, 0], _N.linspace(_N.min(amps[startIt:, 0]), _N.max(amps[startIt:, 0]), NB))
    indMamps  =  _N.where(hist == _N.max(hist))[0][0]
    indMampsL =  max(indMamps - NeighB, 0)
    indMampsH =  min(indMamps + NeighB+1, NB)
    loA, hiA = bins[indMampsL], bins[indMampsH]

    fig = _plt.figure(figsize=(8, 8))
    fig.add_subplot(2, 1, 1)
    _plt.hist(fs[startIt:, 0], bins=_N.linspace(_N.min(fs[startIt:, 0]), _N.max(fs[startIt:, 0]), NB), color="black")
    _plt.axvline(x=loF, color="red")
    _plt.axvline(x=hiF, color="red")
    fig.add_subplot(2, 1, 2)
    _plt.hist(amps[startIt:, 0], bins=_N.linspace(_N.min(amps[startIt:, 0]), _N.max(amps[startIt:, 0]), NB), color="black")
    _plt.axvline(x=loA, color="red")
    _plt.axvline(x=hiA, color="red")
    if dir is None:
        _plt.savefig(resFN("chosenFsAmps", dir=setname))
    else:
        _plt.savefig(resFN("%s/chosenFsAmps" % dir, dir=setname))
    _plt.close()

    indsFs = _N.where((fs[startIt:, 0] >= loF) & (fs[startIt:, 0] <= hiF))
    indsAs = _N.where((amps[startIt:, 0] >= loA) & (amps[startIt:, 0] <= hiA))

    asfsInds = _N.intersect1d(indsAs[0], indsFs[0]) + startIt
    q = _N.mean(smp_q2[0, startIt:])


    #alfas = _N.mean(allalfas[asfsInds], axis=0)
    pcklme = [aus, q, allalfas[asfsInds], aSs]

    if dir is None:
        dmp = open(resFN("bestParams.pkl", dir=setname), "wb")
    else:
        dmp = open(resFN("%s/bestParams.pkl" % dir, dir=setname), "wb")
    pickle.dump(pcklme, dmp, -1)
    dmp.close()


def downsamplespkdat(dat, isipctl, max_evry=3):
    """
    look at ISI distribution, and if the shortest ISI is above 1
    if we make bin too large, will lead to inaccurate history effect
    can appear as strong "bursting" effect
    """
    TR = dat.shape[0]

    isis_list = []
    for tr in range(TR):
        spkts = _N.where(dat[tr] == 1)[0]
        isis_list.extend(_N.diff(spkts))

    isis = _N.array(isis_list)

    maxISI = _N.max(isis)    

    cnts, bins = _N.histogram(isis, bins=_N.linspace(0, maxISI+1, maxISI+2))

    y = _N.cumsum(cnts) / _N.sum(cnts)
    evry = _N.where((y[0:-1] < isipctl) & (y[1:] >= isipctl))[0][0] + 1
    evry = evry if evry <= max_evry else max_evry

    #  downsample data

    L = dat.shape[1]

    newL = L // evry
    dnew = _N.array(dat[:, 0:newL * evry])

    dnew_rs = _N.sum(dnew.reshape((TR, newL, evry)), axis=2)
    
    x, y = _N.where(dnew_rs > 1)

    dnew_rs[x, y] = 1

    return evry, dnew_rs


def ARcoeff(modulus_fs, Fs):
    alfa = alfarep(modulus_fs, Fs)
    ARcoeff          = (-1*_Npp.polyfromroots(alfa)[::-1][1:]).real
    return ARcoeff

def alfarep(modulus_fs, Fs):
    """
    modulus can also be negative if th == 0
    """
    dt = 1./Fs
    l_alfa = []
    for mod_f in modulus_fs:
        hz = mod_f[1]
        r = mod_f[0]

        if hz != 0:
            th = 2*hz*dt
        
            l_alfa.append(r*(_N.cos(_N.pi*th) + 1j*_N.sin(_N.pi*th)))
            l_alfa.append(r*(_N.cos(_N.pi*th) - 1j*_N.sin(_N.pi*th)))
        else:
            l_alfa.append(r)

    alfa  = _N.array(l_alfa)

    return alfa
