import numpy as _N


Fs = 300
T  = 30 # seconds
ts = _N.linspace(0, T, Fs*T, endpoint=False)

f1  = 30
y1 = _N.sin(2*_N.pi*f1*ts)

f2  = 60
y2 = _N.sin(2*_N.pi*f2*ts)


y = y1 + y2
y += 0.1*_N.random.randn(Fs*T)

_N.savetxt("30_60_at_300.dat", y, fmt="%.4e", )

