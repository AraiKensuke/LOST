import numpy as _N
import os as _os

class KFARDat:
    N     = None;    #  # of data for each trial
    Ns    = None;    #  # of data for each trial
    TR    = None;    #  # of trials
    #  KF params
    p_x   = None;    p_V   = None;    p_Vi  = None
    f_x   = None;    f_V   = None
    s_x   = None;    s_V   = None
    J     = None
    Ik    = None

    #  params for AR state
    k     = None;
    ks    = None;
    F     = None
    Fs    = None
    Q     = None;
    G     = None

    #  generative and generated data
    x     = None;

    #  original spks
    dN    = None

    def __init__(self, *args, **kwargs):
        oo = self
        tr1 = kwargs.has_key("onetrial") and kwargs["onetrial"]

        oo.TR      = args[0]
        oo.N       = args[1]
        oo.k       = args[2]   #  if only using as single
        oo.Ns      = oo.N if tr1 else _N.ones(oo.TR, dtype=_N.int)*oo.N
        oo.ks      = oo.k if tr1 else _N.ones(oo.TR, dtype=_N.int)*oo.k

        oo.initKF(oo.TR, oo.N, oo.k, onetrial=tr1)

        _N.fill_diagonal(oo.F[1:, 0:oo.k-1], 1)
        oo.G       = _N.zeros((oo.k, 1))
        oo.G[0, 0] = 1
        oo.Q       = 0.1 if tr1 else _N.empty(oo.TR)

    def initKF(self, TR, N, k, onetrial=False):   #  init KF arrays
        tr1      = onetrial
        Np1      = N + 1
        oo = self
        oo.F     = _N.zeros((k, k))
        oo.Fs    = _N.zeros((k, k)) if tr1 else _N.zeros((TR, k, k))
        if tr1:
            _N.fill_diagonal(oo.Fs[1:, 0:oo.k-1], 1)
        else:
            for tr in xrange(TR):
                _N.fill_diagonal(oo.Fs[tr, 1:, 0:oo.k-1], 1)
        oo.Ik    = _N.identity(k)
        oo.IkN   = _N.tile(oo.Ik, (N+1, 1, 1))

        #  need TR
        #  pr_x[:, 0]  empty, not used
        oo.p_x   = _N.empty((Np1, k, 1))  if tr1 else _N.empty((TR, Np1, k, 1)) 
        oo.p_x[:, 0, 0] = 0
        oo.p_V   = _N.empty((Np1, k, k))  if tr1 else _N.empty((TR, Np1, k, k)) 
        oo.p_Vi  = _N.empty((Np1, k, k))  if tr1 else _N.empty((TR, Np1, k, k)) 
        oo.f_x   = _N.empty((Np1, k, 1))  if tr1 else _N.empty((TR, Np1, k, 1)) 
        oo.f_V   = _N.empty((Np1, k, k))  if tr1 else _N.empty((TR, Np1, k, k)) 
        oo.s_x   = _N.empty((Np1, k, 1))  if tr1 else _N.empty((TR, Np1, k, 1)) 
        oo.s_V   = _N.empty((Np1, k, k))  if tr1 else _N.empty((TR, Np1, k, k)) 
        #  J[0]  empty, not used
        oo.J     = _N.empty((Np1, k, 1))  if tr1 else _N.empty((TR, Np1, k, 1)) 


class KFARGauObsDat(KFARDat):
    #  Gaussian observation noise model
    y     = None
    H     = None
    Il    = None
    R     = None    #  noise
    Rv    = None    #  time-dependent noise
    K     = None

    def __init__(self, *args, **kwargs):
        tr1 = kwargs.has_key("onetrial") and kwargs["onetrial"]
        oo = self
        KFARDat.__init__(self, *args, **kwargs)
        Np1 = oo.N + 1
        oo.H       = _N.zeros((1, oo.k))          #  row vector
        oo.H[0, 0] = 1
        if tr1:
            oo.K       = _N.empty((Np1, oo.k, 1))
            oo.y       = _N.empty(Np1)
            oo.dN      = _N.empty(Np1)
            oo.Rv      = _N.empty(Np1)
        else:
            oo.K       = _N.empty((oo.TR, Np1, oo.k, 1))
            oo.y       = _N.empty((oo.TR, Np1))
            oo.dN      = _N.empty((oo.TR, Np1))
            oo.Rv      = _N.empty((oo.TR, Np1))

    def copyData(self, *args, **kwargs):    #  generative data
        tr1 = kwargs.has_key("onetrial") and kwargs["onetrial"]
        oo = self
        y     = args[0]
        if tr1:
            oo.dN[:] = y
        else:
            if oo.TR == 1:
                oo.dN[0, :] = y
            else:
                oo.dN[:, :] = y

    def copyParams(self, *args, **kwargs):   #  AR params that will get updated by EM
        tr1 = kwargs.has_key("onetrial") and kwargs["onetrial"]
        oo = self
        F0              = args[0]
        Q               = args[1]
        oo.F[0, :]    = F0[:]
        if tr1:
            oo.Fs[0, :]    = F0[:]
            oo.Q           = Q    #  scalar
        else:
            for tr in xrange(oo.TR):
                oo.Fs[tr, 0, :]    = F0[:]
            oo.Q[:]       = Q    #  scalar
