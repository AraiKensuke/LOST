import scipy.stats as _ss
from LOST.ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import LOST.commdefs as _cd
import matplotlib.pyplot as _plt
import numpy as _N
import LOST.buildsignal as _ars

N     = 50000

#  Can AR(1) express a signal where low and high frequency are about same
#  amplitude?  
fig = _plt.figure(figsize=(11, 6))
Tdisp=2000
#########################################
comp1         = [[[10, 0.99]], [0.3]]#, [40, 0.99]], []]
#comp2         = [[[40, 0.99]], [0.97]]
comps         = [comp1]#, comp2]
#wgts          = [0.4, 0.6]
wgts          = [1]
sgnl = _ars.build_signal(N, comps, wgts)
_plt.subplot2grid((2, 4), (0, 0), colspan=3)
_plt.plot(sgnl[Tdisp:2*Tdisp])
_plt.subplot2grid((2, 4), (0, 3))
_plt.psd(sgnl, Fs=1000)
_plt.xlim(0, 100)

#########################################
comp1         = [[[10, 0.998], [40, 0.998]], []]
comps         = [comp1]
wgts          = [1]

sgnl = _ars.build_signal(N, comps, wgts)
_plt.subplot2grid((2, 4), (1, 0), colspan=3)
_plt.plot(sgnl[Tdisp:2*Tdisp])
_plt.subplot2grid((2, 4), (1, 3))
_plt.psd(sgnl, Fs=1000)
_plt.xlim(0, 100)
