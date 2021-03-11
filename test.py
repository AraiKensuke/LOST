import numpy as _N
import matplotlib.pyplot as _plt

lprs = _N.linspace(-50000, -100, 1000)

fig = _plt.figure(figsize=(4, 8))
fig.add_subplot(1, 2, 1)

prs = _N.exp(lprs)

cdf = _N.zeros(prs.shape[0]+1)

cdf[1:] = _N.cumsum(prs)

cdf /= cdf[-1]

_plt.plot(cdf)

fig.add_subplot(1, 2, 2)
lmax = _N.where(lprs == _N.max(lprs))[0][0]

lprs_mmin = lprs - lprs[lmax]

prsBoosted = _N.exp(lprs_mmin)

cdf = _N.zeros(prs.shape[0]+1)

cdf[1:] = _N.cumsum(prsBoosted)

cdf /= cdf[-1]

_plt.plot(cdf)
