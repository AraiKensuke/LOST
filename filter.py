import numpy as _N
import scipy.signal as _ssig
from scipy.signal import convolve, remez, freqz, freqs, filter_design, lfilter
from scipy.signal.filter_design import butter, zpk2tf, bessel, cheby1, cheb1ord
from filtfilt import filtfilt
import matplotlib.pyplot as _plt

def base_q4atan(x, y):
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

def bpFiltSpec(fpL, fpH, fsL, fsH, nyqf):
    nyqf = float(nyqf)
    ord, wn = filter_design.buttord((fpL/nyqf, fpH/nyqf), (fsL/nyqf, fsH/nyqf), 1, 12, analog=0)
    b, a = butter(ord, wn, btype="bandpass", analog=0, output='ba')
    w, h = freqz(b, a)
    fig = _plt.figure()
    _plt.plot(w * (nyqf/_N.pi), _N.abs(h))

def lpFiltSpec(fp, fs, nyqf):
    nyqf = float(nyqf)
    ord, wn = filter_design.buttord(fp/nyqf, fs/nyqf, 1, 12, analog=0)
    b, a = butter(ord, wn, btype="lowpass", analog=0, output='ba')
    w, h = freqz(b, a)
    fig = _plt.figure()
    _plt.plot(w * (nyqf/_N.pi), _N.abs(h))

def bpFilt(fpL, fpH, fsL, fsH, nyqf, y):
    nyqf = float(nyqf)
    ord, wn = filter_design.buttord((fpL/nyqf, fpH/nyqf), (fsL/nyqf, fsH/nyqf), 1, 12, analog=0)
    b, a = butter(ord, wn, btype="bandpass", analog=0, output='ba')
    fy  =    filtfilt(b, a, y)
    return fy
        
def lpFilt(fp, fs, nyqf, y, disp=False):
    nyqf = float(nyqf)
    ord, wn = filter_design.buttord(fp/nyqf, fs/nyqf, 1, 12, analog=0)
    b, a = butter(ord, wn, btype="lowpass", analog=0, output='ba')
    fy  =    filtfilt(b, a, y)
    return fy

def hpFilt(fp, fs, nyqf, y, disp=False):
    nyqf = float(nyqf)
    ord, wn = filter_design.buttord(fp/nyqf, fs/nyqf, 1, 12, analog=0)
    b, a = butter(ord, wn, btype="highpass", analog=0, output='ba')
    fy  =    filtfilt(b, a, y)
    return fy

def gauKer(w):
    wf = _N.empty(8*w+1)

    for i in xrange(-4*w, 4*w+1):
        wf[i+4*w] = _N.exp(-0.5*(i*i)/(w*w))

    return wf

