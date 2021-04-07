import numpy.polynomial.polynomial as _Npp
from LOST.kflib import createDataAR
import scipy.stats as _ss
from LOST.ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import LOST.commdefs as _cd
import matplotlib.pyplot as _plt

ram = True
import numpy as _N

if ram:
    import LOST.ARcfSmplNoMCMC_ram as _arcfs
else:
    import LOST.ARcfSmplNoMCMC as _arcfs
    #import LOST.ARcfSmplFlatf as _arcfs

class mcmcARcontinuous:
    freq_order    = True
    ARord         = _cd.__NF__

    #  guessing AR coefficients of this form
    Cn      = 0   # noise components
    Cs      = 10
    R       = 1
    Fs      = None
    fSigMax = None

    N     =  1000
    TR    = 1
    a_q2         = 1.;          B_q2         = 0.1

    smpx    = None
    uts     = None
    wts     = None
    rts     = None    #  the R components
    zts     = None    #  the C components
    q2s     = None
    fs      = None
    rs      = None
    amps    = None
    skp     = None
    TR      = None
    allalfas = None
    obsvd   = None

    def __init__(self, Fs, C, R):
        oo = self
        oo.R = R
        oo.C = C
        oo.k     = 2*C + R
        oo.Fs= Fs
        oo.dt= 1./Fs

        oo.fSigMax    = Fs//2
        oo.freq_lims     = [[0.000001, oo.fSigMax]]*C


    def getComponents(self):
        """
        it0, it1 are gibbs iterations skipped, so should be like ITER//skp
        """
        oo = self
        skpdITER = oo.wts.shape[0]
        N  = oo.smpx.shape[1] - 2
        _rts = _N.empty((skpdITER, oo.TR, N+1, oo.R, 1))    #  real components   N = ddN
        _zts = _N.empty((skpdITER, oo.TR, N+1, oo.C, 1))    #  imag components 

        for it in range(skpdITER):
            for tr in range(oo.TR):
                b, c = dcmpcff(alfa=oo.allalfas[it*oo.skp])
                for r in range(oo.R):
                    _rts[it, tr, :, r] = b[r] * oo.uts[it, tr, r, :]

                for z in range(oo.C):
                    #print "z   %d" % z
                    cf1 = 2*c[2*z].real
                    gam = oo.allalfas[it*oo.skp, oo.R+2*z]
                    cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
                    _zts[it, tr, 0:N+1, z] = cf1*oo.wts[it, tr, z, 1:N+2] - cf2*oo.wts[it, tr, z, 0:N+1]

        oo.rts = _N.array(_rts[:, 0, 1:, :, 0])
        oo.zts = _N.array(_zts[:, 0, 1:, :, 0])

        zts_stds = _N.std(oo.zts, axis=1)
        srtd     = _N.argsort(zts_stds)     #  ITER x C
        for it in range(skpdITER):
            oo.fs[it] = oo.fs[it, srtd[it, ::-1]]
            oo.amps[it] = oo.amps[it, srtd[it, ::-1]]
            oo.zts[it, :, :] = oo.zts[it, :, srtd[it, ::-1]].T

    def showComponents(self, t0=0, t1=None, it0=0, it1=None, gtcompts=None):
        """
        t0, t1   time
        it0, it1  gibbs iter
        """
        oo = self
        N  = oo.smpx.shape[1] - 2
        skpdITER = oo.wts.shape[0]

        it1 = skpdITER if it1 is None else it1
        mzts = _N.mean(oo.zts[it0:it1], axis=0)

        t1 = t1 if t1 is not None else N
        ts = _N.arange(0, N)
        fig = _plt.figure(figsize=(12, 9))
        fig.add_subplot(1, 1, 1)
        AMP0 = _N.max(oo.obsvd[0, 0:N]) - _N.min(oo.obsvd[0, 0:N])
        y0  = 0
        MIN = _N.min(oo.obsvd[0, 0:N])   
        _plt.plot(ts, oo.obsvd[0, oo.k:N+oo.k], color="black")
        minSpc = AMP0*0.05
        
        corrs = _N.empty(oo.C)
        closest_GTcmpts = _N.ones(oo.C, dtype=_N.int) * -1   # -1 if AR comp not THE closest one to one of the GTcomponents given
        for igt in range(gtcompts.shape[1]):
            for icmp in range(oo.C):            
                corrs[icmp], pv = _ss.pearsonr(gtcompts[oo.k+t0:t1+oo.k, igt], mzts[t0:t1, icmp])
            closest_ARcmp = _N.where(corrs == _N.max(corrs))[0][0]
            closest_GTcmpts[closest_ARcmp] = igt
        print(closest_GTcmpts)
        
        #  find the zts that each gtcomponent most similar to
        for icmp in range(oo.C):
            MAX = _N.max(mzts[t0:t1, icmp])
            spc = (MAX - MIN)*1.05
            spc = minSpc if spc < minSpc else spc
            y0  -= spc
            _plt.plot(ts, mzts[t0:t1, icmp] + y0)
            if gtcompts is not None:   #### IF KNOW GT
                igt = closest_GTcmpts[icmp]
                if igt != -1:
                    _plt.plot(ts, gtcompts[oo.k+t0:t1+oo.k, igt] + y0, color="grey")
        _plt.yticks([])
        _plt.xlim(t0, t1)
        fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)

    def componentsAtGibbsIter(self, it, t0=0, t1=None):
        """
        t0, t1   time
        it0, it1  gibbs iter
        """
        oo = self
        N  = oo.smpx.shape[1] - 2

        t1 = t1 if t1 is not None else N
        ts = _N.arange(0, N)

        _plt.plot(_N.sum(oo.zts[it], axis=1) + _N.sum(oo.rts[it], axis=1), color="orange")
        _plt.plot(oo.obsvd[oo.k:N+oo.k], color="black")



    def gibbsSamp(self, N, ITER, obsvd, peek=50, skp=50):
        """
        peek 
        """
        oo = self
        oo.TR           = 1
        sig_ph0L      = -1
        sig_ph0H      = 0   #  
        oo.obsvd         = obsvd
        oo.skp = skp

        radians      = buildLims(0, oo.freq_lims, nzLimL=1., Fs=oo.Fs)
        AR2lims      = 2*_N.cos(radians)

        F_alfa_rep  = initF(oo.R, oo.C, 0).tolist()   #  init F_alfa_rep

        if ram:
            alpR        = _N.array(F_alfa_rep[0:oo.R], dtype=_N.complex)
            alpC        = _N.array(F_alfa_rep[oo.R:], dtype=_N.complex)
            alpC_tmp    = _N.array(F_alfa_rep[oo.R:], dtype=_N.complex)
        else:
            alpR        = F_alfa_rep[0:oo.R]
            alpC        = F_alfa_rep[oo.R:]
            alpC_tmp        = list(F_alfa_rep[oo.R:])
        q2          = _N.array([0.01])

        oo.smpx        = _N.empty((oo.TR, N+2, oo.k))

        oo.fs           = _N.empty((ITER//skp, oo.C))
        oo.rs           = _N.empty((ITER//skp, oo.R))
        oo.amps         = _N.empty((ITER//skp, oo.C))
        oo.q2s          = _N.empty(ITER//skp)
        oo.uts          = _N.empty((ITER//skp, oo.TR, oo.R, N+1, 1))
        oo.wts          = _N.empty((ITER//skp, oo.TR, oo.C, N+2, 1))

        #  oo.smpx[:, 1+oo.ignr:, 0:ook], oo.smpx[:, oo.ignr:, 0:ook-1]
        if ram:
            _arcfs.init(N, oo.k, 1, oo.R, oo.C, 0, aro=_cd.__NF__)
            smpx_contiguous1        = _N.zeros((oo.TR, N + 1, oo.k))
            smpx_contiguous2        = _N.zeros((oo.TR, N + 2, oo.k-1))

        for n in range(N):
            oo.smpx[0, n+2] = oo.obsvd[0, n:n+oo.k][::-1]
        for m in range(oo.TR):
            oo.smpx[0, 1, 0:oo.k-1]   = oo.smpx[0, 2, 1:]
            oo.smpx[0, 0, 0:oo.k-2]   = oo.smpx[0, 2, 2:]
        if ram:
            _N.copyto(smpx_contiguous1, 
                      oo.smpx[:, 1:])
            _N.copyto(smpx_contiguous2, 
                      oo.smpx[:, 0:, 0:oo.k-1])

        oo.allalfas     = _N.empty((ITER, oo.k), dtype=_N.complex)


        for it in range(ITER):
            itstore = it // skp
            if it % peek == 0:
                if it > 0:
                    print("%d  -----------------" % it)
                    print(prt)

            if ram:
                oo.uts[itstore], oo.wts[itstore] = _arcfs.ARcfSmpl(N+1, oo.k, oo.TR, AR2lims, smpx_contiguous1, smpx_contiguous2, q2, oo.R, 0, oo.C, alpR, alpC, sig_ph0L, sig_ph0H, 0.2*0.2)
            else:
                oo.uts[itstore], oo.wts[itstore] = _arcfs.ARcfSmpl(N, oo.k, AR2lims, oo.smpx[:, 1:, 0:oo.k], oo.smpx[:, :, 0:oo.k-1], q2, oo.R, oo.C, 0, alpR, alpC, oo.TR, aro=ARord, sig_ph0L=sig_ph0L, sig_ph0H=sig_ph0H)

            F_alfa_rep[0:oo.R] = alpR
            F_alfa_rep[oo.R:]  = alpC
            oo.allalfas[it] = F_alfa_rep
            #F_alfa_rep = alpR + alpC   #  new constructed
            prt, rank, f, amp = ampAngRep(F_alfa_rep, oo.dt, f_order=True)

            #  reorder

            if oo.freq_order:
                # coh = _N.where(amp > 0.95)[0]
                # slow= _N.where(f[coh] < f_thr)[0]
                # #  first, rearrange 

                for i in range(oo.C):
                    alpC_tmp[2*i] = alpC[rank[i]*2]
                    alpC_tmp[2*i+1] = alpC[rank[i]*2+1]
                for i in range(oo.C):
                    alpC[2*i] = alpC_tmp[2*i]
                    alpC[2*i+1] = alpC_tmp[2*i+1]

                oo.amps[itstore, :]  = amp[rank]
                oo.fs[itstore, :]    = 0.5*(f[rank]/oo.dt)
            else:
                oo.amps[itstore, :]  = amp
                oo.fs[itstore, :]    = 0.5*(f/oo.dt)

            oo.rs[itstore]       = alpR

            F0          = (-1*_Npp.polyfromroots(F_alfa_rep)[::-1].real)[1:]

            a2 = oo.a_q2 + 0.5*(oo.TR*N + 2)  #  N + 1 - 1
            BB2 = oo.B_q2

            for m in range(oo.TR):
                #   set x00 
                rsd_stp = oo.smpx[m, 3:, 0] - _N.dot(oo.smpx[m, 2:-1], F0).T

                BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
            q2[:] = _ss.invgamma.rvs(a2, scale=BB2)

            oo.q2s[itstore] = q2[0]

        it0=0
        it1=ITER
        it0 = it0 // skp
        it1 = it1 // skp
            
        #zts = _N.mean(_zts[it0:it1], axis=0)
        #rts = _N.mean(_rts[it0:it1], axis=0)


    def summary_of_run(self):
        oo = self
        skpdITER = oo.wts.shape[0]
        showITERS = skpdITER//2

        if oo.freq_order:
            ordrd_fs = oo.fs
        else:
            ordrd_fs = _N.sort(oo.fs, axis=1)
        fig = _plt.figure(figsize=(9, 12))
        _plt.subplot2grid((3, 4), (0, 0), colspan=4)
        for ic in range(oo.C):
            _plt.plot(ordrd_fs[:, ic], marker=".")#, color=clrs[ic])
        _plt.xlabel("Gibbs Iter (skip %d)" % oo.skp)
        _plt.ylabel("freq (Hz)")
        _plt.subplot2grid((3, 4), (1, 0), colspan=4)

        dx = 0.5/oo.Fs
        #  int0^500 dx = 1
        for ic in range(oo.C):
            cnts, bins = _N.histogram(ordrd_fs[:, ic], bins=_N.linspace(0, oo.fSigMax, oo.fSigMax+1), density=True)
            _plt.plot(0.5*(bins[0:-1] + bins[1:]), cnts)#, color=clrs[ic])

            #stat_str_mn += "%.1f   " % _N.mean(ordrd_fs[:, ic])
            #stat_str_md += "%.1f   " % _N.median(ordrd_fs[:, ic])
            _plt.axvline(x=_N.mean(ordrd_fs[:, ic]), ls=":")#, color=clrs[ic])
        _plt.xlim(0, oo.Fs//2)
        _plt.xlabel("freq (Hz)")
        _plt.ylabel("histogram")
        _plt.subplot2grid((3, 4), (2, 0), colspan=4)
        _plt.title("imag roots")
        for ic in range(oo.C):
            _plt.scatter(oo.fs[showITERS:, ic], oo.amps[showITERS:, ic], s=3)
        _plt.xlabel("freq (Hz)")
        _plt.ylabel("modulus")
        _plt.ylim(0, 1)
        _plt.xlim(0, oo.Fs//2)
        # _plt.subplot2grid((3, 4), (2, 3), colspan=1)
        # _plt.title("real roots")
        # for ir in range(oo.R):
        #     _plt.scatter(_N.ones(oo.rs.shape[0] - showITERS)*ir, oo.rs[showITERS:, ir], s=3)
        # _plt.xlabel("root #")
        # _plt.ylim(-1, 1)
        #_plt.suptitle("freq_order %(fo)s\n%(mn)s\n%(md)s\n" % {"fo" : str(freq_order), "mn" : stat_str_mn, "md" : stat_str_md})
        fig.subplots_adjust(wspace=0.4, hspace=0.4, left=0.08, bottom=0.08, top=0.93, right=0.95)
    #show(N, Cs+Cn, obsvd, zts, t0=1000, t1=2000)

