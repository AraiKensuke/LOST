from kassdirs import resFN
import scipy.signal as _ssig
import re as _re
exf("filter.py")


#  modulation histogram.  phase @ spike
#setname="080402-2-3-theta"
setname="tim_n8"
p = _re.compile("^\d{6}")   # starts like "exptDate-....."
m = p.match(setname)

bRealDat = False
COLS = 3

if m == None:
    bRealDat = False
    COLS = 3

dat  = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
N, cols = dat.shape

TR   = cols / COLS

M    = 3   #  mult by
datM = _N.zeros((N/M, cols))

missingSpikes = 0
for tr in xrange(TR):
    for n in xrange(N/M):
        iFound = 0
        for i in xrange(M):
            if (dat[M*n + i, COLS*tr + 2] == 1):
                iFound += 1
            if iFound > 0:
                datM[n, COLS*tr + 2]   = 1   #  we lose 1 spk.  not a big deal
            if iFound > 1:
                print "More than 1 spikes in this collapsed bin"
                missingSpikes += iFound - 1

    datM[:, COLS*tr:COLS*tr+2] = dat[::M, COLS*tr:COLS*tr+2]
    if bRealDat:
        datM[:, COLS*tr+3] = dat[::M, COLS*tr+3]

setnameM="%(sn)sM%(M)d" % {"sn" : setname, "M" : M}

if bRealDat:
    fmt = "% .3e % .3e %d %.3f " * TR
    _N.savetxt(resFN("xprbsdN.dat", dir=setnameM, create=True), datM, fmt=fmt)
else:
    fmt = "% .3e %.3e %d " * TR
    _N.savetxt(resFN("xprbsdN.dat", dir=setnameM, create=True), datM, fmt=fmt)
