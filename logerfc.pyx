import pickle as _pkl
import numpy as _N
import scipy.stats as _ss
import os

cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double log(double)
    double abs(double)

class logerfc:
    _sqrt2   = sqrt(2)
    _twpi    = 2*_N.pi
    _logerfc = None    #  precomputed data
    _x0      = 0
    _x1      = None
    _R       = None
    _dx      = None
    #  useful constants

    def __init__(self):
        try:
            floc = os.environ["logerfc"]
        except ValueError:
            print "Need logerfc.dat file location.  Put it in shell startup script";    exit()
        try:
            fpk = open(floc, "r")
        except IOError:
            print "Didn't find logerfc.dat file at %s" % floc;  exit()

        dat = _pkl.load(fpk)
        self._logerfc = dat
        self._R, c     = self._logerfc.shape
        self._x1      = self._logerfc[self._R-1, 0]
        self._dx      = self._logerfc[1, 0] - self._logerfc[0, 0]
        fpk.close()

    def at(self, x):    #  logarithm of error function
        abx = _N.abs(x)

        if abx > self._x1:
            #print "abx  %(abx).3e   >  _x1  %(x1).3e" % {"abx" : abx, "x1" : _x1}
            return None

        n   = int(abx / self._dx)
        rx  = (abx - self._logerfc[n, 0]) / self._dx

        if n < self._R - 1:
            lval = self._logerfc[n, 1] + rx * (self._logerfc[n+1, 1] - self._logerfc[n, 1])
        else:
            lval = self._logerfc[n, 1]

        if x < 0:
            #  log(2 - val) = log(val x [2/val - 1])
            lval = log(2 - exp(lval))

        return lval

    def trncNrmNrmlz(self, a, b, u, sg):
        #   single variable Normal distribution
        #  Value of density is
        #  _N.exp(-(xs - u)**2 / (2*sg2) - lNC)

        #  Normalization is
        anmz = (a-u) / (self._sqrt2*sg)
        bnmz = (b-u) / (self._sqrt2*sg)
        sg2  = sg*sg

        #################################
        if (anmz < 0) and (bnmz > 0):    #  SIMPLE case
            NC = sqrt(2*_N.pi*sg2) * (_ss.norm.cdf((b - u) / sg) - \
                                      _ss.norm.cdf((a - u) / sg))
            return log(NC)
        if (anmz < 0) and (bnmz <= 0):
            tmp  = anmz
            anmz = -bnmz
            bnmz = -tmp
        #  NOW bnmz > anmz

        if bnmz > 19999:   #  when bnmz this large, eps almost 0 no matter anmz
            eps = 0
            print "(NO PROBLEM?)   bnmz large val   bnmz %(bnmz).3e  anmz %(anmz).3e" % {"bnmz" : bnmz, "anmz" : anmz}
        else:
            try:
                eps = exp(self.at(bnmz) - self.at(anmz))
            except TypeError:
                print "type error "
                print "anmz       %(1)e" % {"1" : anmz}
                print "bnmz       %(1)e" % {"1" : bnmz}
                print "at(anmz)   %(1)e" % {"1" : self.at(anmz)}
                print "at(bnmz)   %(1)e" % {"1" : self.at(bnmz)}

                raise

        lNC = log(0.5) + log(sqrt(self._twpi*sg2)) + self.at(anmz) - eps - 0.5*eps*eps - 0.166666666*eps*eps*eps - 0.04166666666*eps*eps*eps*eps
        return lNC
