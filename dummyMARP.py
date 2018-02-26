"""
"""
class dummyMARP:
    burn    = None
    NMC     = None
    TR      = None
    fs      = None
    amps    = None
    allalfas= None
    smp_q2  = None
    mnStds  = None
    B       = None
    u       = None

    def __init__(self, burn, NMC, TR, fs, amps, allalfas, q2, mnStds, B, u):
        self.burn     = burn
        self.NMC      = NMC
        self.TR       = TR
        self.fs       = fs
        self.amps     = amps
        self.allalfas = allalfas
        self.smp_q2   = q2
        self.smp_q2   = q2
        self.mnStds   = mnStds
        self.B        = B
        self.u        = u

class dummyMARPN:
    N       = None
    burn    = None
    NMC     = None
    TR      = None
    fs      = None
    amps    = None
    allalfas= None
    smp_q2  = None
    mnStds  = None
    B       = None
    u       = None
    y       = None

    def __init__(self, burn, NMC, TR, N, x, y, fs, amps, allalfas, q2, mnStds, B, u):
        self.burn     = burn
        self.NMC      = NMC
        self.TR       = TR
        self.N        = N
        self.x        = x
        self.y        = y
        self.fs       = fs
        self.amps     = amps
        self.allalfas = allalfas
        self.smp_q2   = q2
        self.smp_q2   = q2
        self.mnStds   = mnStds
        self.B        = B
        self.u        = u
