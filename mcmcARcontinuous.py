import numpy.polynomial.polynomial as _Npp
from LOST.kflib import createDataAR
import scipy.stats as _ss
from LOST.ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import LOST.commdefs as _cd
import LOST.practice.buildsignal as _blds
import matplotlib.pyplot as _plt
import mne

ram = True
import numpy as _N
#_N.random.seed(12)

if ram:
    import LOST.ARcfSmplNoMCMC_ram as _arcfs
else:
    import LOST.ARcfSmplNoMCMC as _arcfs
    #import LOST.ARcfSmplFlatf as _arcfs

def getComponents(uts, wts, allalfas, it0, it1, skp):
    ddN   = N
    
    rts = _N.empty((ITER//skp, TR, ddN+1, R, 1))    #  real components   N = ddN
    zts = _N.empty((ITER//skp, TR, ddN+1, Cn+Cs, 1))    #  imag components 


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


def show(N, C, obsvd, zts, t0=0, t1=None):
    t1 = t1 if t1 is not None else N
    ts = _N.arange(0, N)
    fig = _plt.figure(figsize=(12, 9))
    fig.add_subplot(1, 1, 1)
    AMP0 = _N.max(obsvd[0, 0:N]) - _N.min(obsvd[0, 0:N])
    y0  = 0
    MIN = _N.min(obsvd[0, 0:N])   
    _plt.plot(ts, obsvd[0, k:N+k], color="black")
    minSpc = AMP0*0.05
    for icmp in range(C):
        MAX = _N.max(zts[:, icmp])
        spc = (MAX - MIN)*1.05
        spc = minSpc if spc < minSpc else spc
        y0  -= spc
        _plt.plot(ts, zts[:, icmp] + y0)
        if icmp < nComps:
            _plt.plot(ts, cmpts[k:N+k, icmp] + y0, color="grey")
        MIN = _N.min(zts[:, icmp])
    _plt.yticks([])
    _plt.xlim(t0, t1)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)

def allcmps(N, k, it, zts, rts, obsvd, t0=0, t1=None):
    t1 = t1 if t1 is not None else N
    ts = _N.arange(0, N)
    
    _plt.plot(_N.sum(zts[it], axis=1) + _N.sum(rts[it], axis=1), color="orange")
    _plt.plot(obsvd[k:N+k], color="black")

freq_order    = True
ARord         = _cd.__NF__

#  guessing AR coefficients of this form
Cn      = 0   # noise components
Cs      = 8
R       = 1

k     = 2*(Cn + Cs) + R
#N     =  10000
#N     =  10000
#N     =  10000
N     =  1000
TR    = 1

#Fs=1000
#dt = 0.001
Fs = 300
dt = 0.003333
#dat    = "twoAR1l"
#__eeg = _N.loadtxt("practice/eeg_w_hi.dat")
__eeg = _N.loadtxt("practice/eeg_rps2.dat")
t0    = 14000
#t0    = 6000
__obsvd = __eeg[t0:t0+N+100, 0]

#__obsvd = __eeg[300:11100, 0]
#__obsvd = _N.loadtxt("practice/%s.dat" % dat)
#__obsvd = _N.loadtxt("practice/oneAR2.dat")
#__obsvd = _N.loadtxt("practice/threeAR1_3.dat")
#__obsvd = _N.loadtxt("practice/twoAR1_1_1.dat")
#__obsvd = _N.loadtxt("practice/twoAR2_1.dat")
#__obsvd = _N.loadtxt("practice/singleAR1_1.dat")
if len(__obsvd.shape) == 1:
    __obsvd = __obsvd.reshape((__obsvd.shape[0], 1))
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

hpf = True
fSigMax       = Fs//2    #  fixed parameters

if hpf:
    info  = mne.create_info(["eeg"], Fs, ch_types="eeg")#, montage="standard_1020")
    obsvd_X = _N.empty((obsvd[0].shape[0], 1))
    obsvd_X[:, 0] = obsvd[0]
    raw_orig   = mne.io.RawArray(obsvd_X.T, info)

    filt_low = 6
    filt_high= fSigMax-1

    raw_orig.filter(filt_low, filt_high)
    dat = raw_orig.get_data()
    obsvd[0] = dat[0]

rmoutlr = False

if rmoutlr:
    sd = _N.std(obsvd[0])
    mn = _N.mean(obsvd[0])
    
    toobig = _N.where((obsvd[0] - mn) > 4*sd)[0]
    toolit = _N.where((obsvd[0] - mn) < -4*sd)[0]
    obsvd[0, toobig] = 4*sd
    obsvd[0, toolit] = -4*sd

#fSigMax       = 150.    #  fixed parameters
#freq_lims     = [[1 / .85, fSigMax]]
freq_lims     = [[0.000001, fSigMax]]*Cs

sig_ph0L      = -1
#sig_ph0H      = -(0.0*0.0)#95*0.95)   #  
sig_ph0H      = -(0.)   #  
#sig_ph0H      = -(0.9*0.9)   #

radians      = buildLims(Cn, freq_lims, nzLimL=1., Fs=Fs)
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
#a_q2         = 100000;          B_q2         = 0.000000001
a_q2         = 1.;          B_q2         = 0.1

MS          = 1
#ITER        = 5000
ITER        = 1000

skp          = 50    #  we can quickly approach > 10 GB if we store all.
fs           = _N.empty((ITER//skp, Cn + Cs))
rs           = _N.empty((ITER//skp, R))
amps         = _N.empty((ITER//skp, Cn + Cs))
q2s          = _N.empty(ITER//skp)
uts          = _N.empty((ITER//skp, TR, R, N+1, 1))
wts          = _N.empty((ITER//skp, TR, Cn+Cs, N+2, 1))

#  oo.smpx[:, 1+oo.ignr:, 0:ook], oo.smpx[:, oo.ignr:, 0:ook-1]
if ram:
    _arcfs.init(N, k, 1, R, Cs, Cn, aro=_cd.__NF__)
    smpx_contiguous1        = _N.zeros((TR, N + 1, k))
    smpx_contiguous2        = _N.zeros((TR, N + 2, k-1))

for n in range(N):
    smpx[0, n+2] = obsvd[0, n:n+k][::-1]
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

evry=50
for it in range(ITER):
    itstore = it // skp
    if it % evry == 0:
        if it > 0:
            print("%d  -----------------" % it)
            print(prt)

    if ram:
        uts[itstore], wts[itstore] = _arcfs.ARcfSmpl(N, k, 1, AR2lims, smpx_contiguous1, smpx_contiguous2, q2, R, Cs, Cn, alpR, alpC, sig_ph0L, sig_ph0H, 0.01*0.01)
    else:
        uts[itstore], wts[itstore] = _arcfs.ARcfSmpl(N, k, AR2lims, smpx[:, 1:, 0:k], smpx[:, :, 0:k-1], q2, R, Cs, Cn, alpR, alpC, TR, aro=ARord, sig_ph0L=sig_ph0L, sig_ph0H=sig_ph0H)

    F_alfa_rep[0:R] = alpR
    F_alfa_rep[R:]  = alpC
    allalfas[itstore] = F_alfa_rep
    #F_alfa_rep = alpR + alpC   #  new constructed
    prt, rank, f, amp = ampAngRep(F_alfa_rep, dt, f_order=True)

    #  reorder

    
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

        amps[itstore, :]  = amp[rank]
        fs[itstore, :]    = 0.5*(f[rank]/dt)
    else:
        amps[itstore, :]  = amp
        fs[itstore, :]    = 0.5*(f/dt)

    rs[itstore]       = alpR

    F0          = (-1*_Npp.polyfromroots(F_alfa_rep)[::-1].real)[1:]
    #F0          = _Narray([1.9761, -0.9801])#(-1*_Npp.polyfromroots(F_alfa_rep)[::-1].real)[1:]

    a2 = a_q2 + 0.5*(TR*N + 2)  #  N + 1 - 1
    BB2 = B_q2

    for m in range(TR):
        #   set x00 
        rsd_stp = smpx[m, 3:, 0] - _N.dot(smpx[m, 2:-1], F0).T
        # print(rsd_stp[0:10])
        # print(diffs[0:10])
        # print("if know GT coeffs, innovation variance:  %(1).4f   %(2).4f" % {"1
        #" : (_N.std(diffs)**2), "2" : (_N.std(rsd_stp)**2)})

        BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
    q2[:] = _ss.invgamma.rvs(a2, scale=BB2)


    q2s[itstore] = q2[0]

#clrs = ["black", "red", "green", "orange", "blue", "grey", "purple", "cyan", "brown"]

# stat_str_mn = ""
# stat_str_md = ""
if freq_order:
    ordrd_fs = fs
else:
    ordrd_fs = _N.sort(fs, axis=1)
fig = _plt.figure(figsize=(9, 12))
_plt.subplot2grid((3, 4), (0, 0), colspan=4)
for ic in range(Cs+Cn):
    _plt.plot(ordrd_fs[:, ic], marker=".")#, color=clrs[ic])
_plt.xlabel("Gibbs Iter (skip %d)" % skp)
_plt.ylabel("freq (Hz)")
_plt.subplot2grid((3, 4), (1, 0), colspan=4)

dx = 0.5/Fs
#  int0^500 dx = 1
for ic in range(Cs+Cn):
    cnts, bins = _N.histogram(ordrd_fs[:, ic], bins=_N.linspace(0, fSigMax, fSigMax+1), density=True)
    _plt.plot(0.5*(bins[0:-1] + bins[1:]), cnts)#, color=clrs[ic])
    
    #stat_str_mn += "%.1f   " % _N.mean(ordrd_fs[:, ic])
    #stat_str_md += "%.1f   " % _N.median(ordrd_fs[:, ic])
    _plt.axvline(x=_N.mean(ordrd_fs[:, ic]), ls=":")#, color=clrs[ic])
_plt.xlim(0, Fs//2)
_plt.xlabel("freq (Hz)")
_plt.ylabel("histogram")
_plt.subplot2grid((3, 4), (2, 0), colspan=3)
_plt.title("imag roots")
for ic in range(Cs+Cn):
    _plt.scatter(fs[20:, ic], amps[20:, ic], s=3)
_plt.xlabel("freq (Hz)")
_plt.ylabel("modulus")
_plt.ylim(0, 1)
_plt.subplot2grid((3, 4), (2, 3), colspan=1)
_plt.title("real roots")
for ir in range(R):
    _plt.scatter(_N.ones(rs.shape[0] - 20)*ir, rs[20:, ir], s=3)
_plt.xlabel("root #")
_plt.ylim(-1, 1)
#_plt.suptitle("freq_order %(fo)s\n%(mn)s\n%(md)s\n" % {"fo" : str(freq_order), "mn" : stat_str_mn, "md" : stat_str_md})
fig.subplots_adjust(wspace=0.4, hspace=0.4, left=0.08, bottom=0.08, top=0.93, right=0.95)


it0=0
it1=ITER
it0 = it0 // skp
it1 = it1 // skp

__rts, __zts = getComponents(uts, wts, allalfas, it0, it1, skp)
_rts = __rts[:, 0, 1:, :, 0]
_zts = __zts[:, 0, 1:, :, 0]
zts = _N.mean(_zts[it0:it1], axis=0)
rts = _N.mean(_rts[it0:it1], axis=0)

#show(N, Cs+Cn, obsvd, zts, t0=1000, t1=2000)




