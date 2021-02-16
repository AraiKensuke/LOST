import numpy.polynomial.polynomial as _Npp
from LOST.kflib import createDataAR
import scipy.stats as _ss
from LOST.ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import LOST.commdefs as _cd

ram = False
if ram:
    import LOST.ARcfSmplNoMCMC_ram as _arcfs
else:
    import LOST.ARcfSmplNoMCMC as _arcfs
from LOST.ARcfSmplNoMCMC import ARcfSmpl, FilteredTimeseries
import numpy as _N

use_prior     = _cd.__COMP_REF__
ARord         = _cd.__NF__

r    = 0.99
th   = 0.05
th1pi= _N.pi*th

alfa  = _N.array([r*(_N.cos(th1pi) + 1j*_N.sin(th1pi)), 
                  r*(_N.cos(th1pi) - 1j*_N.sin(th1pi))])

ARcoeff          = (-1*_Npp.polyfromroots(alfa)[::-1][1:]).real

#  used to generate
gCn      = 0   # noise components
gR       = 1
gCs      = 1
#  guessing AR coefficients of this form
Cn      = 3   # noise components
Cs      = 1
R       = 3

k     = 2*(Cn + Cs) + R
N     = 5000
TR    = 1

sgnl, y = createDataAR(N+k, ARcoeff, 0.1, 0.1)

nzs     = _N.empty((gR + gCn, N+k))
EXMPL   = 1

obsvd   = _N.empty((TR, N+k))
sdSgnl  = _N.std(sgnl)

scl     = _N.empty(gR + gCn)

fSigMax       = 500.    #  fixed parameters
freq_lims     = [[1 / .85, fSigMax]]
sig_ph0L      = -1
sig_ph0H      = -(0.5*0.5)   #  

radians      = buildLims(Cn, freq_lims, nzLimL=1.)
AR2lims      = 2*_N.cos(radians)

F_alfa_rep  = initF(R, Cs, Cn).tolist()   #  init F_alfa_rep
if ram:
    alpR        = _N.array(F_alfa_rep[0:R])
    alpC        = _N.array(F_alfa_rep[R:])
else:
    alpR        = F_alfa_rep[0:R]
    alpC        = F_alfa_rep[R:]
q2          = _N.array([0.01])

smpx        = _N.empty((TR, N+2, k))

#  q2  --  Inverse Gamma prior
a_q2         = 1e-1;          B_q2         = 1e-6

MS          = 1
ITER        = 1000

fs           = _N.empty((ITER, Cn + Cs))
amps         = _N.empty((ITER, Cn + Cs))


#  oo.smpx[:, 1+oo.ignr:, 0:ook], oo.smpx[:, oo.ignr:, 0:ook-1]
if ram:
    _arcfs.init(N, k, 1, R, Cs, Cn, aro=_cd.__NF__)
    smpx_contiguous1        = _N.zeros((TR, N + 1, k))
    smpx_contiguous2        = _N.zeros((TR, N + 2, k-1))

for ex in range(EXMPL):
    obsvd[0] = sgnl

    for n in range(gR):
        AR1 = _N.array([(_N.random.rand() - 0.5)*2])*0.1
        #AR1 = _N.array([0.98])
        nzs[n], y = createDataAR(N+k, AR1, 0.05, 0.05)

        sdNz = _N.std(nzs[n])
        scl[n] = _N.random.rand() * MS

        obsvd[0] += scl[n] * ((sdSgnl / sdNz) * nzs[n])

    for n in range(N):
        smpx[0, n+2] = obsvd[0, n:n+k]
    for m in range(TR):
        smpx[0, 1, 0:k-1]   = smpx[0, 2, 1:]
        smpx[0, 0, 0:k-2]   = smpx[0, 2, 2:]
    if ram:
        _N.copyto(smpx_contiguous1, 
                  smpx[:, 1:])
        _N.copyto(smpx_contiguous2, 
                  smpx[:, 0:, 0:k-1])

    for it in range(ITER):
        if ram:
            _arcfs.ARcfSmpl(N, k, 1, AR2lims, smpx_contiguous1, smpx_contiguous2, q2, R, Cs, Cn, alpR, alpC, sig_ph0L, sig_ph0H)
        else:
            _arcfs.ARcfSmpl(N, k, AR2lims, smpx[:, 1:, 0:k], smpx[:, :, 0:k-1], q2, R, Cs, Cn, alpR, alpC, TR, aro=ARord, sig_ph0L=sig_ph0L, sig_ph0H=sig_ph0H)
        F_alfa_rep[0:R] = alpR
        F_alfa_rep[R:]  = alpC
        #F_alfa_rep = alpR + alpC   #  new constructed
        prt, rank, f, amp = ampAngRep(F_alfa_rep, 0.001, f_order=True)
        print(prt)
        
        for m in range(TR):
            amps[it, :]  = amp
            fs[it, :]    = f

        F0          = (-1*_Npp.polyfromroots(F_alfa_rep)[::-1].real)[1:]

        a2 = a_q2 + 0.5*(TR*N + 2)  #  N + 1 - 1
        BB2 = B_q2
        for m in range(TR):
            #   set x00 
            rsd_stp = smpx[m, 3:,0] - _N.dot(smpx[m, 2:-1], F0).T
            BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
        q2[:] = _ss.invgamma.rvs(a2, scale=BB2)
