import os

def resFN(fn, dir=None, create=False, env_dirname=None):
    #  ${LOST_dir}/Results/
    #  dir is inside the result directory
    #  look inside __LOST_ResultDir__ env var.
    env_dirname="__LOST_ResultDir__" if (env_dirname==None) else env_dirname

    __LOST_ResultDir__ = os.environ[env_dirname]
    rD = __LOST_ResultDir__

    if dir != None:
        lvls = dir.split("/")
        for lvl in lvls:
            rD += "/%s" % lvl
            if not os.access("%s" % rD, os.F_OK) and create:
                os.mkdir(rD)
    return "%(rd)s/%(fn)s" % {"rd" : rD, "fn" : fn}

def datFN(fn, dir=None, create=False):
    #  ${LOST_dir}/Results/
    __LOST_DataDir__ = os.environ["__LOST_DataDir__"]
    dD = __LOST_DataDir__

    if dir != None:
        dD = "%(dd)s/%(ed)s" % {"dd" : __LOST_DataDir__, "ed" : dir}
        if not os.access("%s" % dD, os.F_OK) and create:
            os.mkdir(dD)
    return "%(dd)s/%(fn)s" % {"dd" : dD, "fn" : fn}
