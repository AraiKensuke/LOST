import os

def setFN(fn, dir=None, create=False):
    """
    for programs run from the result directory
    """
    rD = ""

    if dir != None:
        lvls = dir.split("/")
        for lvl in lvls:
            rD += "%s/" % lvl
            if not os.access("%s" % rD, os.F_OK) and create:
                os.mkdir(rD)
        return "%(rd)s%(fn)s" % {"rd" : rD, "fn" : fn}
    return fn

def rename_self_vars(clsvars):
    """
    given list of names of class vars, returns a string like
    'var1 = self.var1; var2 = self.var2', which can be executed
    to make nicknames so we won't have to do self.var1, self.var2 to use.
    """
    str = ""
    for key in clsvars:
        str += "%(v)s = self.%(v)s; " % {"v" : key}
    return str
