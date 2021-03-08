import numpy.polynomial.polynomial as _Npp
from LOST.kflib import createDataAR
import scipy.stats as _ss
from LOST.ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import LOST.commdefs as _cd
import LOST.practice.buildsignal as _blds
import matplotlib.pyplot as _plt

ram = True
import numpy as _N
#_N.random.seed(12)

if ram:
    import LOST.ARcfSmplNoMCMC_ram as _arcfs
else:
    #import LOST.ARcfSmplNoMCMC as _arcfs
    import LOST.ARcfSmplFlatf as _arcfs

def getComponents(uts, wts, allalfas, it0, it1):
    ddN   = N
    
    rts = _N.empty((ITER, TR, ddN+1, R, 1))    #  real components   N = ddN
    zts = _N.empty((ITER, TR, ddN+1, Cn+Cs, 1))    #  imag components 


    for it in range(it0, it1):
        for tr in range(TR):
            b, c = dcmpcff(alfa=allalfas[it])
            for r in range(R):
                rts[it, tr, :, r] = b[r] * uts[it, tr, r, :]
            
            for z in range(Cn+Cs):
                #print "z   %d" % z
                cf1 = 2*c[2*z].real
                gam = allalfas[it, R+2*z]
                cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
                zts[it, tr, 0:ddN+1, z] = cf1*wts[it, tr, z, 1:ddN+2] - cf2*wts[it, tr, z, 0:ddN+1]
    return rts, zts


freq_order    = True
ARord         = _cd.__NF__

#  guessing AR coefficients of this form
Cn      = 0   # noise components
Cs      = 19
R       = 1

k     = 2*(Cn + Cs) + R
N     =  10000
#N     =  3000
TR    = 1

dt = 0.001
#dt = 0.003333
#dat    = "twoAR1l"
#__obsvd = _N.loadtxt("practice/%s.dat" % dat)
#__obsvd = _N.loadtxt("practice/eeg.dat")
#__obsvd = _N.loadtxt("practice/oneAR2.dat")
#__obsvd = _N.loadtxt("practice/singleAR4_1_1.dat")
__obsvd = _N.loadtxt("practice/threeAR1_1_1.dat")
#__obsvd = _N.loadtxt("practice/singleAR2_4_1.dat")
shp = __obsvd.shape
nComps = shp[1]-1

#nComps = 1
#_obsvd = __obsvd#[::3]
_obsvd = __obsvd[:, nComps]#[::3]
_cmpts = __obsvd[:, 0:nComps]#[::3]
#_obsvd = __obsvd[::3, nComps]
#_obsvd = __obsvd[:, nComps]
#_obsvd = __obsvd[:, 0]
obsvd = _N.empty((1, N+100))
#obsvd[0] = _obsvd[5000:5000+N+100]
obsvd[0] = _obsvd[0:0+N+100]
cmpts = _cmpts[0:0+N+100, 0:nComps]#[::3]

fSigMax       = 500.    #  fixed parameters
#fSigMax       = 150.    #  fixed parameters
#freq_lims     = [[1 / .85, fSigMax]]
freq_lims     = [[0.000001, fSigMax]]*Cs

sig_ph0L      = -1
#sig_ph0H      = -(0.0*0.0)#95*0.95)   #  
sig_ph0H      = -(0.3*0.3)   #  
#sig_ph0H      = -(0.9*0.9)   #

radians      = buildLims(Cn, freq_lims, nzLimL=1.)
AR2lims      = 2*_N.cos(radians)


F_alfa_rep  = initF(R, Cs, Cn).tolist()   #  init F_alfa_rep

if ram:
    alpR        = _N.array(F_alfa_rep[0:R], dtype=_N.complex)
    alpC        = _N.array(F_alfa_rep[R:], dtype=_N.complex)
    alpC_tmp    = _N.array(F_alfa_rep[R:], dtype=_N.complex)
else:
    alpR        = F_alfa_rep[0:R]
    alpC        = F_alfa_rep[R:]
    alpC_tmp        = list(F_alfa_rep[R:])
q2          = _N.array([0.01])

smpx        = _N.empty((TR, N+2, k))

#  q2  --  Inverse Gamma prior
a_q2         = 0.5;          B_q2         = 1

MS          = 1
#ITER        = 10
ITER        = 2000

fs           = _N.empty((ITER, Cn + Cs))
rs           = _N.empty((ITER, R))
amps         = _N.empty((ITER, Cn + Cs))
q2s          = _N.empty(ITER)
uts          = _N.empty((ITER, TR, R, N+1, 1))
wts          = _N.empty((ITER, TR, Cn+Cs, N+2, 1))

#  oo.smpx[:, 1+oo.ignr:, 0:ook], oo.smpx[:, oo.ignr:, 0:ook-1]
if ram:
    _arcfs.init(N, k, 1, R, Cs, Cn, aro=_cd.__NF__)
    smpx_contiguous1        = _N.zeros((TR, N + 1, k))
    smpx_contiguous2        = _N.zeros((TR, N + 2, k-1))

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

allalfas     = _N.empty((ITER, k), dtype=_N.complex)

xx           = _N.empty(k)

for it in range(ITER):
    if ram:
        uts[it], wts[it] = _arcfs.ARcfSmpl(N, k, 1, AR2lims, smpx_contiguous1, smpx_contiguous2, q2, R, Cs, Cn, alpR, alpC, sig_ph0L, sig_ph0H)
    else:
        _arcfs.ARcfSmpl(N, k, AR2lims, smpx[:, 1:, 0:k], smpx[:, :, 0:k-1], q2, R, Cs, Cn, alpR, alpC, TR, aro=ARord, sig_ph0L=sig_ph0L, sig_ph0H=sig_ph0H)

    F_alfa_rep[0:R] = alpR
    F_alfa_rep[R:]  = alpC
    allalfas[it] = F_alfa_rep
    #F_alfa_rep = alpR + alpC   #  new constructed
    prt, rank, f, amp = ampAngRep(F_alfa_rep, 0.001, f_order=True)

    #  reorder
    
    if it % 50 == 0:
        print("%d  -----------------" % it)
        print(prt)
    
    if freq_order:
        # coh = _N.where(amp > 0.95)[0]
        # slow= _N.where(f[coh] < f_thr)[0]
        # #  first, rearrange 

        for i in range(Cs+Cn):
            alpC_tmp[2*i] = alpC[rank[i]*2]
            alpC_tmp[2*i+1] = alpC[rank[i]*2+1]
        for i in range(Cs+Cn):
            alpC[2*i] = alpC_tmp[2*i]
            alpC[2*i+1] = alpC_tmp[2*i+1]

        amps[it, :]  = amp[rank]
        fs[it, :]    = 0.5*(f[rank]/dt)
    else:
        amps[it, :]  = amp
        fs[it, :]    = 0.5*(f/dt)

    rs[it]       = alpR

    F0          = (-1*_Npp.polyfromroots(F_alfa_rep)[::-1].real)[1:]
    #F0          = _Narray([1.9761, -0.9801])#(-1*_Npp.polyfromroots(F_alfa_rep)[::-1].real)[1:]

    a2 = a_q2 + 0.5*(TR*N + 2)  #  N + 1 - 1
    BB2 = B_q2

    for m in range(TR):
        #   set x00 
        # y  = obsvd[0, k:]
        # x1 = obsvd[0, k-1:-1]
        # x2 = obsvd[0, k-2:-2]
        # x3 = obsvd[0, k-3:-3]
        # x4 = obsvd[0, k-4:-4]
        # x5 = obsvd[0, k-5:-5]
        # x6 = obsvd[0, k-6:-6]
        # x7 = obsvd[0, k-7:-7]
        # x8 = obsvd[0, k-8:-8]
        # x9 = obsvd[0, k-9:-9]
        # x10 = obsvd[0, k-10:-10]
        # x11 = obsvd[0, k-11:-11]
        # x12 = obsvd[0, k-12:-12]
        # x13 = obsvd[0, k-13:-13]
        # x14 = obsvd[0, k-14:-14]
        # x15 = obsvd[0, k-15:-15]
        # x16 = obsvd[0, k-16:-16]
        # x17 = obsvd[0, k-17:-17]
        # x18 = obsvd[0, k-18:-18]

        #  x2[0] 
        # x18[0] = smpx[0, 2, 0]
        # ...
        # x4[0] = smpx[0, 16, 0]
        # x3[0] = smpx[0, 17, 0]
        # x2[0] = smpx[0, 18, 0]
        # x1[0] = smpx[0, 19, 0]
        ###################################3
        #  x18[0] = smpx[0, 2, 0]    ##  OLDEST
        #  x17[0] = smpx[0, 2, 1]
        #  x16[0] = smpx[0, 2, 2]
        #...
        #  x1[0]  = smpx[0, 2, 17]   ##  NEWEST

        #F0 = _N.array([ 3.71118015, -5.38889359,  3.63732766, -0.96059601])
        # = _N.array([1.97609292, -0.9801])
        #rsd_stp = y - (x1 * F0[0] + x2 * F0[1])
        # rsd_stp = y - (x1 * F0[0] + x2 * F0[1] + x3 * F0[2] + x4 * F0[3] + x5 * F0[4] + x6 * F0[5] + x7 * F0[6] + x8 * F0[7] + x9*F0[8] + x10*F0[9] + x11*F0[10] + x12*F0[11] + x13*F0[12] + x14*F0[13] + x15*F0[14] + x16*F0[15] + x17*F0[16] + x18*F0[17])

        rsd_stp = smpx[m, 3:,k-1] - _N.dot(smpx[m, 2:-1], F0[::-1]).T
        # print(rsd_stp[0:10])
        # print(diffs[0:10])
        # print("if know GT coeffs, innovation variance:  %(1).4f   %(2).4f" % {"1
        #" : (_N.std(diffs)**2), "2" : (_N.std(rsd_stp)**2)})

        BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
    q2[:] = _ss.invgamma.rvs(a2, scale=BB2)


    q2s[it] = q2[0]

#clrs = ["black", "red", "green", "orange", "blue", "grey", "purple", "cyan", "brown"]

stat_str_mn = ""
stat_str_md = ""
if freq_order:
    ordrd_fs = fs
else:
    ordrd_fs = _N.sort(fs, axis=1)
fig = _plt.figure(figsize=(9, 12))
fig.add_subplot(3, 1, 1)
for ic in range(Cs+Cn):
    _plt.plot(ordrd_fs[:, ic])#, color=clrs[ic])
fig.add_subplot(3, 1, 2)

dx = 1./500
#  int0^500 dx = 1
for ic in range(Cs+Cn):
    cnts, bins = _N.histogram(ordrd_fs[:, ic], bins=_N.linspace(0, 500, 501), density=True)
    _plt.plot(0.5*(bins[0:-1] + bins[1:]), cnts + (1./500)*2)#, color=clrs[ic])
    
    stat_str_mn += "%.1f   " % _N.mean(ordrd_fs[:, ic])
    stat_str_md += "%.1f   " % _N.median(ordrd_fs[:, ic])
    _plt.axvline(x=_N.mean(ordrd_fs[:, ic]), ls=":")#, color=clrs[ic])
_plt.xlim(0, 500)
fig.add_subplot(3, 1, 3)
_plt.hist(rs, bins=_N.linspace(-1, 1, 201))

_plt.suptitle("freq_order %(fo)s\n%(mn)s\n%(md)s\n" % {"fo" : str(freq_order), "mn" : stat_str_mn, "md" : stat_str_md})


it0=0
it1=2000
__rts, __zts = getComponents(uts, wts, allalfas, it0, it1)
_rts = __rts[:, 0, 1:, :, 0]
_zts = __zts[:, 0, 1:, :, 0]
zts = _N.mean(_zts[it0:it1], axis=0)
rts = _N.mean(_rts[it0:it1], axis=0)

all_zts = _N.sum(_zts, axis=2)

obsvd_r = obsvd[0, 0:N].reshape((1, N))

ss = _N.std(obsvd_r - all_zts, axis=1)
smax = _N.where(ss == _N.max(ss))[0][0]

fig = _plt.figure(figsize=(9, 9))
fig.add_subplot(3, 1, 1)
_plt.plot(obsvd[0], lw=3, color="black")
_plt.plot(all_zts[smax], color="grey")
fig.add_subplot(3, 1, 2)
_plt.plot(obsvd[0], lw=3, color="black")
_plt.plot(all_zts[smax-1], color="grey")
fig.add_subplot(3, 1, 3)
_plt.plot(obsvd[0], lw=3, color="black")
_plt.plot(all_zts[smax+1], color="grey")


amps = _N.std(all_zts, axis=1)
