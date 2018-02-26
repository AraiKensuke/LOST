import statsmodels.api as _sm

LHbin   = 8  # bin sizes for long history
nLHBins = 16  #  (nLHBins+1) x LHbin  is total history

"""
#  history coeffs:  most recent history terms first
gam = _N.empty(nLHBins)  
gam[:] = [0.5, 0.1, -0.3, -0.05, 0.15, 0, -0.09, 0.005, 0.03, 0, -0.02, 0, 0.003]
gam[:] *= 2.2
beta= _N.empty(LHbin)   #  beta
beta[:] = [-3, -2.5, -2.1, -1.8, -1.5, -1.2, -0.9, -0.3, 0.1, 0.3]
beta[:] *= 2

N       = 60000
st      = _N.zeros(N, dtype=_N.int)
rds     = _N.random.rand(LHbin*(nLHBins+1))
st[_N.where(rds < 0.1)[0]] = 1

dt      = 0.001

probs   = _N.empty(N)

C       = _N.log(30.)


for t in xrange(LHbin*(nLHBins+1), N):
    #  0:9
    hist = st[t-LHbin*(nLHBins+1):t][::-1]  #  hist doesn't include current spk time

    sthcts  = hist[0:LHbin]   #  
    lthcts  = _N.sum(hist[LHbin:LHbin*(nLHBins+1)].reshape(nLHBins, LHbin), axis=1)

    lmd     = _N.exp(_N.dot(beta, sthcts) + _N.dot(gam, lthcts) + C)
    probs[t] = lmd*dt
    if _N.random.rand() < lmd*dt:
        st[t] = 1

    # lthcts   spike counts in bins sized LHbin.  most recent first
"""
dat = _N.loadtxt("../Results/fig1Bus1w/xprbsdN.dat")
st = dat[:, 2::3]

N  = dat.shape[0]
TR = dat.shape[1] / 3
#TR = 90  #  200



#  The design matrix
#  # of params LHBin + nLHBins + 1

Ldf = N - LHbin*(nLHBins+1)
X  = _N.empty((TR, Ldf, LHbin + nLHBins + 1))
X[:, :, 0] = 1  #  offset
y  = _N.empty((TR, Ldf))

for tr in xrange(TR):
    for t in xrange(LHbin*(nLHBins+1), N):
        #  0:9
        hist = st[t-LHbin*(nLHBins+1):t, tr][::-1]

        sthcts  = hist[0:LHbin]   #  
        lthcts  = _N.sum(hist[LHbin:LHbin*(nLHBins+1)].reshape(nLHBins, LHbin), axis=1)
        X[tr, t-LHbin*(nLHBins+1), 1:LHbin+1] = sthcts
        X[tr, t-LHbin*(nLHBins+1), LHbin+1:]  = lthcts
        #y[tr, t-LHbin*(nLHBins+1)]            = st[t, tr]
        y[tr, t-LHbin*(nLHBins+1)]            = st[t, tr]


yr  = y.reshape(TR*Ldf)
Xr  = X.reshape(TR*Ldf, LHbin + nLHBins + 1)
est = _sm.GLM(yr, Xr, family=_sm.families.Poisson()).fit()
offs  = est.params[0]
shrtH = est.params[1:LHbin+1]
oscH  = est.params[LHbin+1:]

cfi = est.conf_int()
oscCI = cfi[LHbin+1:]



_plt.fill_between(range(nLHBins), _N.exp(oscCI[:, 0]), _N.exp(oscCI[:, 1]), color="blue", alpha=0.2)
_plt.plot(range(nLHBins), _N.exp(oscH), lw=2, color="black")
_plt.xlim(0, nLHBins-1)
_plt.savefig("../Results/fig1Bus1w/GLMoschist")
