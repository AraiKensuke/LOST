import sys
import os
import numpy as _N
import matplotlib.pyplot as _plt
import scipy.stats as _ss
import re as _re

#  for 2 time series that may share a slow common trend, this calculates 
#  the CC in smaller blocks so that correlation arising from trend is
#  minimize
def ccW12(y, N, blks):
    pcs = []
    for i in xrange(0, N - blks, blks/2):
        pc, pv = _ss.pearsonr(y[i:i + blks, 0], y[i:i + blks, 1])
        pcs.append(pc)

    return pcs

def universalNDarraySave(datarray, fn):
    #  always save data as little endian.  On big endian machines, 
    #  use endian=little to view file in gnuplot
    if sys.byteorder == "big":
        datarray.byteswap(True)  #  make little endian
    datarray.tofile(fn)
    if sys.byteorder == "big":
        datarray.byteswap(True)  #  make big endian

def savetxtWCom(fname, X, fmt="%.18e", delimiter=" ", com=""):
    _N.savetxt(fname, X, fmt=fmt, delimiter=delimiter)

    #We read the existing text from file in READ mode
    src=open(fname, "r")
    oline=src.readlines()
    #Here, we prepend the string we want to on first line
    oline.insert(0,"%s\n" % com)
    src.close()
     
    #We again open the file in WRITE mode 
    src=open(fname,"w")
    src.writelines(oline)
    src.close()
    """
    fp = open("temp", "w")
    if (com != "") and (not com[len(com)-1] == '\n'):
        com = com + "\n"
    fp.write(com)
    fp.close()
    os.system("mv -- \"" + fname + "\" temp2")
    os.system("cat temp temp2 > \"" + fname + "\"")
    os.system("rm temp temp2")
    """

def RoesslerTS(T, dt = 0.01, a=0.1, b=0.1, c=14):
    kanwa = 10000
    rows  = int(T/dt)
    x = _N.zeros(rows)
    y = _N.zeros(rows)
    z = _N.zeros(rows)
    _x = _N.zeros(kanwa)
    _y = _N.zeros(kanwa)
    _z = _N.zeros(kanwa)

    _x[0] = _N.random.rand()
    _y[0] = _N.random.rand()
    _z[0] = _N.random.rand()
    
    for i in range(1, kanwa):
        _x[i] = _x[i - 1] + (-_y[i - 1] - _z[i - 1])*dt
        _y[i] = _y[i - 1] + (_x[i - 1] + a*_y[i - 1])*dt
        _z[i] = _z[i - 1] + (b + _z[i - 1] * (_x[i - 1] - c))*dt

    x[0] = _x[kanwa - 1]
    y[0] = _y[kanwa - 1]
    z[0] = _z[kanwa - 1]

    for i in range(1, rows):
        x[i] = x[i - 1] + (-y[i - 1] - z[i - 1])*dt
        y[i] = y[i - 1] + (x[i - 1] + a*y[i - 1])*dt
        z[i] = z[i - 1] + (b + z[i - 1] * (x[i - 1] - c))*dt

    return x, y, z

def pickNIndicesOutOfM(N, M, fromThisArray=None):
    """
    N random indices from an array of length M.  if N < 0, don't do anything
    """
    rinds = range(0, M)   #  random indices
    comp  = []
    if (M > N) and (N > 0):
        cl = M   # current length
        for n in range(0, M - N):
            kill = int(_N.random.rand() * cl)
            comp.append(rinds[kill])
            rinds.pop(kill)

            cl -= 1

    if fromThisArray != None:
        if len(fromThisArray) < M:
            print("Warning, fromThisArray too short.  Need at least M elements")
        rindsA = []
        compA  = []

        for i in range(len(rinds)):
            rindsA.append(fromThisArray[rinds[i]])
        for i in range(len(comp)):
            compA.append(fromThisArray[comp[i]])
        return rindsA, compA

    return rinds, comp

def theseInds(arr, inds):
    """
    give me list made of members of arr as specified by list of indices inds
    """
    if len(arr) <= max(inds):  #  10   max is 10 - 1
        return None
    if min(inds) < 0:
        return None
    ret = []
    for i in inds:
        ret.append(arr[i])
    return ret

def shuffle(arr):
    L = len(arr)
    t = arr[:]
    shuffled = []
    for l in xrange(L):
        elem = t.pop(int(L*_N.random.rand()))
        shuffled.append(elem)
        L -= 1
    return shuffled

def shuffleThsInds(arr, inds):
    """
    instead of shuffling all elements in list, shuffle only between the indices given by inds
    """
    mtbl = []
    for l in xrange(len(inds)):
        mtbl.append(arr[inds[l]])  #  mutable elements

    shmtbl = shuffle(mtbl)   
    shuffled = arr[:]

    tot = 0
    for l in xrange(len(inds)):
        shuffled[inds[l]] = shmtbl[l]  #  mutable
    return shuffled

def g_randomGrpsOfM(g, M):
    """
    split M elements into g groups
    if doesn't split equally, ignore last few elements of large group
    ex.  [a, b, c, d, e, f, g, h]  --> g_randomGrpsOfM(2, 8)
         [[a, c, e, f], [b, d, g, h]]
    """
    pg = M / g    #  number per group
    aM = pg * g   #  ajusted M so each grp has same number

    lgs= aM
    lgg= range(0, M)
    retArr = []
    while lgs > pg:
        smg, lgg = pickNIndicesOutOfM(pg, lgs, fromThisArray=lgg)
        retArr.append(smg)
        lgs -= pg
    retArr.append(lgg)
    return retArr

#  random array of +/- 1    
def ranPlusMinus(N):
    a = _N.ones(N, dtype=_N.int)
    for n in range(N):
        if _N.random.rand() > 0.5:
            a[n] = -1

    return a

def OUPn(TEND, dt, tau, x0=0, xm=0, rect=False):
    k = 1. / tau

    D = k*k    #  Choose this to make autocorr integral always same value
    SD = _N.sqrt(D)
    return OUP(TEND, dt, SD, k, x0=x0, xm=xm, rect=rect)

def OUP(TEND, dt, sig, k, rmin=0.1, x0=0, xm=0, rect=False):
    atts = _N.arange(0, TEND, dt)
    SDT = _N.sqrt(dt)
    
    L    = len(atts)
    x    = _N.zeros(L)
    x[0] = x0

    noise=_N.random.randn(L)
    for i in range(L - 1):  # <dW^2> = dt, so to sig x dW == sig x SDT x N(0, 1)
        dx = -k * (x[i] - xm) * dt + sig * SDT * noise[i]
        x[i + 1] += x[i] + dx

    del noise

    if rect:
        for i in range(L):
            if x[i] < rmin:
                x[i] = rmin
    return x


#   spk times in 1/dt
def poiSpksFromRate(r, dt, t0=0):
    spks = []
    L    = len(r)
    rands= _N.random.rand(L)

    for i in range(L):
        if rands[i] < r[i]*dt:
            spks.append(i + t0)
    del rands
    return spks

#  Tmax, t0 in ms
def gammaSpksFromRate(r, k, Tmax=-1, t0=0, refractory=2, returnISIs=False):
    """
    r    in Hz
    """
    if ((type(r) == float) or (type(r) == int)) and (Tmax > 0):
        r = _N.ones(2*Tmax) * r
        #  make r a bit longer so we don't go overflow 1000./r[t] l8r on
    else:
        print("wrong, if give me constant rate, tell me how long spk sequence")
    #  k - 1 is Poisson

    f     = _N.mean(r)            
    fms   = f * 0.001             # freq (spks / ms)
    avgisi_ms = 1. / fms

    L     = int(Tmax / avgisi_ms) + 5  #  do + 5 just in case L is 0
    L *= 10
    if L < 10:
        L *= 10  #  lower expected spikes, more variablity
    isis  = _N.random.gamma(k, 1./k, size=L)
    isiL  = len(isis)

#    for i in xrange(isiL):
#        if isiL < refractory

    spks = []
    t = int(isis[0] * (1000./r[0]))
    s = 0
    spks.append(t + t0)
    s += 1
    rISIs = []

    #  r[t] in Hz  1/r[t] in seconds.  want it in ms.
    while t < Tmax:
#        print "s: " + str(s) + "   L: " + str(L)
        isi  = int(isis[s] * (1000./r[t]))
        if isi < refractory:
            isi = int(refractory + _N.abs(2*refractory*_N.random.randn()))
        rISIs.append(isi)
        t += isi
        spks.append(t + t0)
        s += 1
    spks.pop()   # remove last one as it is past Tmax

    if returnISIs:
        return spks, rISIs
    return spks

#  Tmax in ms             
def gammaSpksFromRate2(r, k, Tmax=-1, t0=0, refractory=2, returnISIs=False, smpf=1000):
    """
    r    in Hz
    """
    smpf  = float(smpf)
    if smpf != 1000:
        Tmax = (smpf / 1000.) * Tmax
        #  make r a bit longer so we don't go overflow 1000./r[t] l8r on
    #  k - 1 is Poisson

    dt    = 1./smpf
    f     = _N.mean(r)            
    fms   = f * dt             # freq (spks / ms)
    avgisi_ms = 1. / fms

    L     = int(Tmax / avgisi_ms) + 5  #  do + 5 just in case L is 0
    L *= 10
    if L < 10:
        L *= 10  #  lower expected spikes, more variablity
    isis  = _N.random.gamma(k, 1./k, size=L)   #  this is an absolute
    isiL  = len(isis)

    spks = []
    t = int(isis[0] * (smpf/r))   #  in units of dt
    refractory = refractory * (smpf / 1000.)
    s = 0
    spks.append(t + t0)
    s += 1
    rISIs = []

    #  r[t] in Hz  1/r[t] in seconds.  want it in ms.
    while t < Tmax:
#        print "s: " + str(s) + "   L: " + str(L)
        isi  = int(isis[s] * (smpf/r))
        if isi < refractory:
            isi = int(refractory + _N.abs(2*refractory*_N.random.randn()))
        rISIs.append(isi)
        t += isi
        spks.append(t + t0)
        s += 1
    spks.pop()   # remove last one as it is past Tmax

    if returnISIs:
        return spks, rISIs
    return spks
             

    
#  _plt.xcorr, _plt.acorr return x-axis in units of samples
#  Here, we give it in units of time
def plotcorrWTimeAxis(x, maxlags, dt, skip=1):
    tL = -maxlags * dt * skip
    tR =  -tL
    
    ac = _plt.acorr(x[::skip], maxlags=maxlags, usevlines=False)
    _plt.close()
#  data is in ac[1]
    tps =  _N.arange(tL, tR + dt, skip*dt)   # time points
    _plt.plot(tps, ac[1])
    _plt.grid()
    _plt.ylim(-1, 1)
    _plt.axhline(y=0, ls="--")

    return tps, ac[1]

#  what is prob. of given empirial value given a histogram of
#  nullhypothesis data values (prob. that empv is 
def pvalGivenNullHypDist(nulldatas, empv, binw=1., minpval=0.0, maxpval=1.):
    #  in case where nulldatas is counts, binw=1. makes perfect sense
    #  in other cases, for example nulldatas = body mass of McDonalds
    min = _N.min(nulldatas)
    max = _N.max(nulldatas)

    bins= _N.arange(min - binw, max + binw, binw)
    out = _plt.hist(nulldatas, bins=bins, normed=True)

    dx  = out[1][1] - out[1][0]
    pts = len(out[1])
    A   = 0

    if empv < out[1][0]:
        if maxpval < 1.:   #  A is 0, pval is 1
            return maxpval
        else:
            return 1.

    for i in range(pts - 1):
        if (empv >= out[1][i]) and (empv <= out[1][i + 1]):
            break
        A += out[0][i]

    A *= dx
        
    if 1 - A < minpval:   #  prob. of measuring larger than
        return minpval
    return 1 - A
    

def js(pvals):
    jss = _N.zeros(len(pvals))
    for i in range(len(pvals)):
        jss[i] = _N.log((1 - pvals[i]) / pvals[i])
    return jss

def allcomboschoose2(total):
    """
    all combos of choosing 2 elements from total choices
    ex. allcomboschoose2(4)  --> [0,1],[0,2],[0,3],[1,2],[1,3],[2,3]
    """
    combos = []
    for i in range(total):
        for j in range(i + 1, total):
            combos.append([i, j])
    return combos

def factorial(N):
    ret = 1
    for n in range(1, N + 1):
        ret *= n
    return ret

def fromBinDat(dat, ISIs=False, SpkTs=False):
    if len(dat.shape) == 1:
        dat = dat.reshape((len(dat), 1))
    N  = dat.shape[0]
    TR = dat.shape[1]

    isis  = []
    spkts = []
    for tr in xrange(TR):
        ts = _N.where(dat[:, tr] == 1)[0]
        spkts.extend(ts)
        isis.extend(ts[1:] - ts[0:-1])
    if ISIs and (not SpkTs):
        return isis
    elif (not ISIs) and SpkTs:
        return spkts
    else:
        return isis, spkts

def toISI(spksByTrialOrNeuron):
    """
    return me an isis as an array of arrays, either by neuron
    or by trial.  
    spksByTrialOrNeuron should be an array of arrays, either
    by neuron or by trial
    """
    tOrN = len(spksByTrialOrNeuron)

    isis = []

    for n in range(tOrN):
        isis.append([])
        
    for n in range(tOrN):
        for ispt in range(len(spksByTrialOrNeuron[n]) - 1):
            isis[n].append(spksByTrialOrNeuron[n][ispt + 1] - 
                           spksByTrialOrNeuron[n][ispt])

    return isis

def toSpkT(isisByTrialOrNeuron):
    """
    reverse of toISI.
    """
    tOrN = len(isisByTrialOrNeuron)

    spkTs = []

    for n in range(tOrN):
        spkTs.append([])
        spkTs[n].append(0)
        for isii in range(len(isisByTrialOrNeuron[n])):
            spkTs[n].append(spkTs[n][-1] + isisByTrialOrNeuron[n][isii])

    return spkTs

def discreteAndLinzNormWF(waveDat, binsz=3, divs=8, dynrange=3):
    """
    discretize WF, in binsz=3
    waveDat is WF to be discretized
    dynrange  how many SDs to consider in discretization?  anything outside will get lower or upper extreme value
    divs      how many subdivisions of dynrange?
    """
    L       = len(waveDat)
    Lb      = len(waveDat[0:L:binsz])

    mn      = _N.mean(waveDat[0:L:binsz])
    sd      = _N.std(waveDat[0:L:binsz])

    rv      = _ss.norm(0, 1)
    wvd     = _N.zeros(Lb)
    wvd[:]  = (waveDat[0:L:binsz] - mn) / sd
    disc    = _N.zeros(Lb, dtype=int)
    
    for l in xrange(Lb):
        #  standardized wave form value
        sy  = wvd[l]
        
        if sy < -dynrange:
            sy = -dynrange
        elif sy > dynrange:
            sy = dynrange

        # cum val
        cv = rv.cdf(sy)
        disc[l] = int(cv * divs)

    return disc

def loadtxt2Darr(fn, dtype=_N.float32):
    """
    _N.loadtxt returns all sorts of weird shapes of array as output.
    I'm usually expecting a 2D array
    """
    #  returns numy array
    #  if size is 1, shape is len 0
    arr = _N.loadtxt(fn, dtype=_N.int16)    
    
    if arr.size == 1:
        arr = arr.reshape(1, 1)
    elif len(arr.shape) == 1:
        arr = _N.reshape(arr, (len(arr), 1))
    return arr

def rtheta(x):
    N = len(x)
    rp= 0
    ip= 0
    
    tot   = _N.sum(x)
    ths   = _N.zeros(N)

    #  ReiTheta = sum_j  cos(th_j) + i sin(th_j)
    rt    = 0
    for n in xrange(N):
        rt +=    x[n]
        tht = 2*_N.pi*rt / tot

        rp += _N.cos(tht)
        ip += _N.sin(tht)

        
    R    = _N.sqrt(rp**2 + ip**2) / N
    return R
    
    
def rmnan(arr, col=0, row=0, axis=0):
    #  basically work with situation where each row is individual data 
    #  col, row is row or column in which to look for NaN data.
    if axis == 1:
        arr = arr.T
        col = row
    reshaped = False
    if len(arr.shape) == 1:
        reshaped = True
        arr = arr.reshape(arr.shape[0], 1)
        
    L   = arr.shape[0]
    okrows = []
    for l in xrange(L):
        if not _N.isnan(arr[l, col]):
            okrows.append(arr[l, :])
    retArr = _N.array(okrows)

    if reshaped:
        retArr = retArr.reshape(len(okrows), )
    if axis == 1:
        retArr = retArr.T
    return retArr

    
def strArr(arr):
    #  clean printing of array
    if len(arr.shape) == 1:
        rs = 1
        cs = arr.shape[0]
        arr = arr.reshape((1, cs))
    else:
        rs = arr.shape[0]
        cs = arr.shape[1]
    str= ""
    for r in xrange(rs):
        for c in xrange(cs - 1):
            str += "% .3f " % arr[r, c]
        str += "% .3f" % arr[r, cs - 1]
        if r != rs - 1:
            str += "\n"
    return str


def behvRelevant(dat, ddat, vd=500, os=0):
    L   = len(dat)

    strt     = 0

    l        = 0
    dgtz     = _N.zeros(L - 1)
    iZ       = 0

    minmax   = []
    minmax.append([os, dat[0], 0, False])
    for l in xrange(L - 1):
        if ddat[l] != 0:
            dgtz[l] = -1
            if ddat[l] > 0:
                dgtz[l] = 1
            stop = l
            if iZ > 0:
                if ((ddat[strt - 1] > 0) and (ddat[stop] < 0)):
                    # a flat MAXIMUM  (otherwise pt. of inflection)
                    minmax.append([(strt + stop)/ 2 + os, dat[strt], 1, False])
                elif ((ddat[strt - 1] < 0) and (ddat[stop] > 0)):
                    # a flat MINIMUM  (otherwise pt. of inflection)
                    minmax.append([(strt + stop)/ 2 + os, dat[strt], 0, False])
            elif (l > 1) and (dgtz[l - 1] != dgtz[l]):
                if ddat[l - 1] > 0:
                    minmax.append([l + os, dat[l], 1, False])
                elif ddat[l - 1] < 0:
                    minmax.append([l + os, dat[l], 0, False])
                else:
                    minmax.append([l + os, dat[l], -1, False])
            iZ = 0
        if (ddat[l] == 0):   #  flat sections of the graph
            if iZ == 0:
                strt = l
            dgtz[l] = 0
            iZ += 1
    #  R end point
    if dat[L - 4] > dat[L - 1]:
        minmax.append([L - 1 + os, dat[L - 1], 0, False])
    else:
        minmax.append([L - 1 + os, dat[L - 1], 1, False])

    #  copy minmax to cminmax (so easy access to those whose ignore==False)
    cminmax = minmax[:]
    for m in xrange(len(cminmax) - 1, -1, -1):
        if cminmax[m][3]:
            cminmax.pop(m)
    #  first, delete extrema that are close to each other in y-value
    for m in xrange(1, len(cminmax) - 1):
        if (abs(minmax[m - 1][1] - minmax[m][1]) < vd):
            minmax[m][3] = True

    #  copy minmax to cminmax (so easy access to those whose ignore==False)
    dminmax = cminmax[:]
    for m in xrange(len(dminmax) - 1, 0, -1):
        if dminmax[m][3]:
            dminmax.pop(m)

    #  first, delete neighboring extrema that are of same type
    for m in xrange(len(dminmax) - 2, 0, -1):
        if dminmax[m][2] == dminmax[m + 1][2]:
            if dminmax[m][2] == 1:
                if dminmax[m][1] > dminmax[m + 1][1]:
                    dminmax.pop(m+1)
                else:
                    dminmax.pop(m)
            elif dminmax[m][2] == 0:
                if dminmax[m][1] < dminmax[m + 1][1]:
                    dminmax.pop(m+1)
                else:
                    dminmax.pop(m)

    #  compare last point to 
    L = len(dminmax)
    if _N.abs((dminmax[L - 2][1] - dminmax[L - 1][1])) < vd:
        dminmax.pop(L - 2)

    jts = []    #  just times
    for m in xrange(len(dminmax)):
        jts.append(dminmax[m][0])
    return dminmax, jts

def locExtremum(dat, ts=50, ys=30, os=0):
    L = len(dat)
    if L <= ts:
        ys = ys * (float(L) / ts)
        ts = L -1

    i1 = ts/10
    i2 = ts/10*9
    exts = []
    typs = []
    for t in xrange(0, L - ts, ts/3):
        imin = dat[t:t+ts].argmin()
        imax = dat[t:t+ts].argmax()
        ymax = dat[imax + t]
        ymin = dat[imin + t]
        yL   = dat[t]
        yR   = dat[t + ts]
        AMP  = ymax - ymin
        bgy  = AMP * 0.2
        if AMP > ys:
            if (imin > i1) and (imin < i2) and (yL - ymin) > bgy and (yR - ymin) > bgy:
                try:
                    exts.index(t + imin + os)
                except ValueError:
                    exts.append(t + imin + os)
                    typs.append(0)   #  minima
            if (imax > i1) and (imax < i2) and (ymax - yL) > bgy and (ymax - yR) > bgy:
                try:
                    exts.index(t + imax + os)
                except ValueError:
                    exts.append(t + imax + os)
                    typs.append(1)   #  maxima
    for t in xrange(L - ts - 1, L - ts - 2, -ts/3):
        imin = dat[t:t+ts].argmin()
        imax = dat[t:t+ts].argmax()
        ymax = dat[imax + t]
        ymin = dat[imin + t]
        yL   = dat[t]
        yR   = dat[t + ts]
        AMP  = ymax - ymin
        bgy  = AMP * 0.2
        if AMP > ys:
            if (imin > i1) and (imin < i2) and (yL - ymin) > bgy and (yR - ymin) > bgy:
                try:
                    exts.index(t + imin + os)
                except ValueError:
                    exts.append(t + imin + os)
                    typs.append(0)   #  minima
            if (imax > i1) and (imax < i2) and (ymax - yL) > bgy and (ymax - yR) > bgy:
                try:
                    exts.index(t + imax + os)
                except ValueError:
                    exts.append(t + imax + os)
                    typs.append(1)   #  maxima
    return exts, typs

def base10toN(num,n, minlen=None):
    """Change a  to a base-n number.
    Up to base-36 is supported without special notation."""
    num_rep={10:'a',
         11:'b',
         12:'c',
         13:'d',
         14:'e',
         15:'f',
         16:'g',
         17:'h',
         18:'i',
         19:'j',
         20:'k',
         21:'l',
         22:'m',
         23:'n',
         24:'o',
         25:'p',
         26:'q',
         27:'r',
         28:'s',
         29:'t',
         30:'u',
         31:'v',
         32:'w',
         33:'x',
         34:'y',
         35:'z'}
    new_num_string=''
    current=num
    while current!=0:
        remainder=current%n
        if 36>remainder>9:
            remainder_string=num_rep[remainder]
        elif remainder>=36:
            remainder_string='('+str(remainder)+')'
        else:
            remainder_string=str(remainder)
        new_num_string=remainder_string+new_num_string
        current=current/n
    if minlen != None:
        while len(new_num_string) < minlen:
            new_num_string = "0" + new_num_string

    return new_num_string

def readAsKeyedData(fn, keyIsInt=False):
    """
    data in format 
    dat1   dat2   dat3   ...  #  ID string
    """
    dat = _N.loadtxt(fn)
    fp  = open(fn, "r")
    lns = fp.readlines()
    if len(dat.shape) == 1:
        dat = dat.reshape((dat.shape[0], 1))

    i   = 0
    ids = []
    for ln in lns:
        p = _re.compile("^[\-\d\.\s\w]+#\s*(.*)$")   #  only lines that have data
        m = p.match(ln)
        if m != None:
            id = m.group(1)
            ids.append(id)

    keyedDat = {}
    for i in xrange(len(ids)):
        if keyIsInt:
            keyedDat[int(ids[i])] = dat[i, :]
        else:
            keyedDat[ids[i]] = dat[i, :]   #  string
    return keyedDat

def readmat_commentAsExtraColumn(fn):
    """
    data in format 
    dat1   dat2   dat3   ...  #  ID string
    """
    dat = _N.loadtxt(fn)
    fp  = open(fn, "r")
    lns = fp.readlines()

    if len(dat.shape) == 1:
        dat = dat.reshape((1, dat.shape[0]))
    R, C    = dat.shape
    augM= _N.zeros((dat.shape[0], dat.shape[1] + 1))

    for i in xrange(R):
        augM[i, 0:C] = dat[i, 0:C]
    
    i   = 0
    for ln in lns:
        p = _re.compile("^[\-\d\.\s\w]+#\s*(.*)$")  # only lines that have data
        m = p.match(ln)
        if m != None:
            id = m.group(1)
            augM[i, C] = float(id)
            i += 1
    return augM

def binUsingIndexAsKey(X, ick, bins=5, binIntv=0):
    """
    bins           # of bins
    binIntv == 0   equally sized bins
    binIntv == 1   equally populated bins
    """
    rows, cols = X.shape

    if binIntv == 0:
        #  bins (x-value boundaries for bins)
        xbins = _N.linspace(min(X[:, ick]), max(X[:, ick]), bins + 1)
    else:
        sdatX = _N.sort(X[:, ick])
        inds  = _N.linspace(0, rows - 1, bins + 1).astype(int)  # inclusive
        #  bins (x-value boundaries for bins)
        xbins = sdatX[inds.astype(int)]

    xbins[-1] += 1   #  make it slightly larger (>= lower, < upper bound)
    
    _bdat  = []

    for b in xrange(bins):
        _bdat.append([])

    for i in xrange(rows):
        for b in xrange(bins):
            if (X[i, ick] >= xbins[b]) and (X[i, ick] < xbins[b + 1]):
                _bdat[b].append(X[i, :])

    bdat   = []

    for b in xrange(bins):
        bdat.append(_N.array(_bdat[b]))

    return bdat

#  union     create a non-duplicating union between two lists
#  intersect only pick items in both   A intrsct B   =   B intrsct A
#  subtractintersect                   A sintrsct B !=   B sintrsct A

def subtract(A, B):
    retA = A[:]
    for item in B:
        try:
            i = retA.index(item)
            retA.pop(i)
        except ValueError:
            pass
    return retA

def strArray(a):
    #  nicely formatted string version of array
    #  ["0.200", ".1", "5.0"] --> ["0.2", "0.1", "5"]
    strA = []

    for elm in a:
        s = str(elm)
        L = len(s)
        lastI = len(s) - 1
        trimEnd = False
        try:
            if s.index('.') >= 0:
                trimEnd = True
        except ValueError:
            pass
        
        if trimEnd and (s[lastI] == '0') or (s[lastI] == '.'):
            while (s[lastI] == '0'):
                lastI -= 1
            if s[lastI] == '.':
                lastI -= 1
        strA.append(s[0:lastI+1])
    return strA
        
def norm(dat):
    u = _N.mean(dat)
    s = _N.std(dat)

    return (dat - u)/s            
    
def cleanPlot(ax):
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

def verycleanPlot(ax):
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

def uniqFN(filename, serial=False, iStart=1, returnPrevious=False):
    """
    if we want uniqFN likes file1.jpg, file2.jpg file3.jpg
    we need to have filename="file.jpg"
    """
    if not serial:   #  only time to look for a bare filename
        if not os.access(filename, os.F_OK):
            return filename
    else:
        fn, ext = os.path.splitext(filename)   #    ext includes .
        i   = iStart
        fnI = "%(fn)s-%(i)d" % {"fn" : fn, "i" : i}
        ofnI = None
        while os.access(fnI + ext, os.F_OK):
            i += 1
            ofnI= fnI
            fnI = "%(fn)s-%(i)d" % {"fn" : fn, "i" : i}

        if not returnPrevious:
            return fnI + ext
        else:
            if ofnI is not None:
                return ofnI + ext, fnI + ext
            else:
                return None, fnI + ext

def cubicRoots(a, b, c, d):
    #  ax^3 + bx^2 + cx + d = 0
    sqrt3 = 1.7320508075688772
    u1 = 1
    u2 = 0.5*(-1 + 1j*sqrt3)
    u3 = 0.5*(-1 - 1j*sqrt3)

    d0 = b*b - 3*a*c
    d1 = 2*b*b*b - 9*a*b*c + 27*a*a*d
    D  = 18*a*b*c*d - 4*b*b*b*d + b*b*c*c - 4*a*c*c*c - 27*a*a*d*d

    thrd = 1./3
    if -27*a*a*D < 0:
        C  = (0.5*(d1 + 1j*_N.sqrt(27*a*a*D)))**thrd
    else:
        C  = (0.5*(d1 + _N.sqrt(-27*a*a*D)))**thrd

    if C != 0:
        x1 = -(b + u1*C + d0 / (u1*C)) / (3*a)
        x2 = -(b + u2*C + d0 / (u2*C)) / (3*a)
        x3 = -(b + u3*C + d0 / (u3*C)) / (3*a)
    elif (C == 0) and (d0 == 0):
        x1 = -b / (3*a)
        x2 = -b / (3*a)
        x3 = -b / (3*a)

    if _N.abs(x1.imag / x1.real) < 1e-10:
        x1 = x1.real
    if _N.abs(x2.imag / x2.real) < 1e-10:
        x2 = x2.real
    if _N.abs(x3.imag / x3.real) < 1e-10:
        x3 = x3.real

    return x1, x2, x3

#http://randlet.com/blog/python-significant-figures-format/
def to_precision(x,p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(_N.log10(x))
    tens = 10**(e - p + 1)
    n = _N.floor(x/tens)

    if n < 10**(p - 1):
        e = e -1
        tens = 10**(e - p+1)
        n = _N.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= 10**p:
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)


def arr_str(arr, elementfmt="%.4f", elements_per_line=20, strpad="  "):
    L = arr.shape[0]

    strout = ""
    for l in range(1, L+1):
        strout += ("%s, " % elementfmt) % arr[l-1]
        if l % elements_per_line == 0:
            strout += "\n%s" % strpad
        
    return strout
