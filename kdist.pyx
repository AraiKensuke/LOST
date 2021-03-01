# import scipy.stats as _ss
# import numpy as _N

# cdef extern from "math.h":
#     double exp(double)
#     double sqrt(double)
#     double log(double)
#     double abs(double)

# def truncnorm(a=1, b=None, u=0., std=1.):
#     """
#     a is left border, b is right border
#     """

#     if b == None:
#         if a > 0:
#             b = a + 100*std  #  
#         else:
#             b = a - 100*std  #
#     else:
#         if b <= a:
#             return None
        
#     a = (a - u) / std
#     b = (b - u) / std

#     #  the cases:
#     #  a, b both same sign, we deal only with positive side

#     if (a <= 0) and (b > 0):
#         smps = u + std*_ss.truncnorm.rvs(a, b)
#         return smps   #  simplest case sould return quickly

#     #  come here for one sided samples
#     bOneSddNeg = False

#     #print "a %(a).4f   b %(b).4f" % {"a" : a, "b" : b}

#     if (a < 0) and (b < 0):
#         a *= -1
#         b *= -1
#         tmp = a
#         a = b
#         b = tmp
#         bOneSddNeg = True
#     if (a > 0) and (b > 0):
#         #  case a > 5    (rejection sampling)
#         if a > 3.5:
#             w = _N.random.rand()
#             x = sqrt(a*a - 2*log(1 - w))
#             v = _N.random.rand()
#             while (v > x/a) or (x > b):
#                 w = _N.random.rand()
#                 x = sqrt(a*a - 2*log(1 - w))
#             smps = x
#         else:
#             smps = _ss.truncnorm.rvs(a, b)

#     if bOneSddNeg:
#         smps *= -1

#     return u + std*smps

# def truncnormC(double a, double b, double u, double std):
#     """
#     a is left border, b is right border
#     """
#     cdef double tmp, w, x, v, smps

#     a = (a - u) / std
#     b = (b - u) / std

#     #  the cases:
#     #  a, b both same sign, we deal only with positive side

#     if (a <= 0) and (b > 0):
#         smps = u + std*_ss.truncnorm.rvs(a, b)
#         return smps   #  simplest case sould return quickly

#     #  come here for one sided samples
#     cdef int bOneSddNeg = 0

#     #print "a %(a).4f   b %(b).4f" % {"a" : a, "b" : b}

#     if (a < 0) and (b < 0):
#         a *= -1
#         b *= -1
#         tmp = a
#         a = b
#         b = tmp
#         bOneSddNeg = 1
#     if (a > 0) and (b > 0):
#         #  case a > 5    (rejection sampling)
#         if a > 3.5:
#             w = _N.random.rand()
#             x = sqrt(a*a - 2*log(1 - w))
#             v = _N.random.rand()
#             while (v > x/a) or (x > b):
#                 w = _N.random.rand()
#                 x = sqrt(a*a - 2*log(1 - w))
#             smps = x
#         else:
#             smps = _ss.truncnorm.rvs(a, b)

#     if bOneSddNeg == 1:
#         smps *= -1

#     return u + std*smps

# def truncnormNV(a=1, b=None, u=0., std=1., size=1):
#     smps = _N.empty(size)

#     if b == None:
#         if a > 0:
#             b = a + 100*std  #  
#         else:
#             b = a - 100*std  #
#     else:
#         if b <= a:
#             return None
        
#     a = (a - u) / std
#     b = (b - u) / std

#     for smp in xrange(size):
#         x = _N.random.randn()
#         while (x < a) or (x > b):
#             x = _N.random.randn()
#         smps[smp] = x

#     if size == 1:
#         return u + std*smps[0]
#     return u + std*smps



import scipy.stats as _ss
import numpy as _N

cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double log(double)
    double abs(double)

def truncnorm(a=1, b=None, u=0., std=1.):
    """
    a is left border, b is right border
    """

    if b == None:
        if a > 0:
            b = a + 100*std  #  
        else:
            b = a - 100*std  #
    else:
        if b <= a:
            return None
        
    a = (a - u) / std
    b = (b - u) / std

    #  the cases:
    #  a, b both same sign, we deal only with positive side

    if (a <= 0) and (b > 0):
        smps = u + std*_ss.truncnorm.rvs(a, b)
        return smps   #  simplest case sould return quickly

    #  come here for one sided samples
    bOneSddNeg = False

    #print "a %(a).4f   b %(b).4f" % {"a" : a, "b" : b}

    if (a < 0) and (b < 0):
        a *= -1
        b *= -1
        tmp = a
        a = b
        b = tmp
        bOneSddNeg = True
    if (a > 0) and (b > 0):
        #  case a > 5    (rejection sampling)
        if a > 3.5:
            w = _N.random.rand()
            x = sqrt(a*a - 2*log(1 - w))
            v = _N.random.rand()
            while (v > x/a) or (x > b):
                w = _N.random.rand()
                x = sqrt(a*a - 2*log(1 - w))
            smps = x
        else:
            smps = _ss.truncnorm.rvs(a, b)

    if bOneSddNeg:
        smps *= -1

    return u + std*smps

def truncnormC(double a, double b, double u, double std):
    """
    a is left border, b is right border
    """
    cdef double tmp, w, x, v
    cdef double smps = 0   #  standardized normal or truncnorm.  

    a = (a - u) / std
    b = (b - u) / std

    #  the cases:
    #  a, b both same sign, we deal only with positive side

    if (a <= 0) and (b > 0):
        return u + std*_ss.truncnorm.rvs(a, b)
        #return smps   #  simplest case sould return quickly

    #  come here for one sided samples
    cdef int bOneSddNeg = 0

    #print "a %(a).4f   b %(b).4f" % {"a" : a, "b" : b}

    if (a < 0) and (b < 0):
        a *= -1
        b *= -1
        tmp = a
        a = b
        b = tmp
        bOneSddNeg = 1
    if (a > 0) and (b > 0):
        #  case a > 5    (rejection sampling)
        if a > 3.5:
            w = _N.random.rand()
            x = sqrt(a*a - 2*log(1 - w))
            v = _N.random.rand()
            while (v > x/a) or (x > b):
                w = _N.random.rand()
                x = sqrt(a*a - 2*log(1 - w))
            smps = x
        else:
            smps = _ss.truncnorm.rvs(a, b)

    if bOneSddNeg == 1:
        smps *= -1

    return u + std*smps

def truncnormNV(a=1, b=None, u=0., std=1., size=1):
    smps = _N.empty(size)

    if b == None:
        if a > 0:
            b = a + 100*std  #  
        else:
            b = a - 100*std  #
    else:
        if b <= a:
            return None
        
    a = (a - u) / std
    b = (b - u) / std

    for smp in xrange(size):
        x = _N.random.randn()
        while (x < a) or (x > b):
            x = _N.random.randn()
        smps[smp] = x

    if size == 1:
        return u + std*smps[0]
    return u + std*smps
