import pickle
import os
from kassdirs import resFN, datFN
import mcmcFigs as mF
import numpy as _N

def concat_smps(exptDir, runDir, xticks=None, ylimAmp=None):
    lms    = []

    i      = 0

    done   = False
    totalIter = 0
    its    = []

    while not done:
        i += 1
        fn = resFN("%(ed)s/%(rd)s/smpls-%(i)d.dump" % {"ed" : exptDir, "rd" : runDir, "i" : i})
        if not os.access(fn, os.F_OK):
            print "couldn't open %s" % fn
            done = True
        else:
            with open(fn, "rb") as f:
                lm = pickle.load(f)
            lms.append(lm)

            totalIter += lm["allalfas"].shape[0]
            its.append(lm["allalfas"].shape[0])

    allalfas = _N.empty((totalIter, lm["allalfas"].shape[1]), dtype=_N.complex)
    q2       = _N.empty((lm["q2"].shape[0], totalIter))
    h_coeffs = _N.empty((lm["h_coeffs"].shape[0], totalIter))
    Hbf      = lm["Hbf"]
    spkhist  = _N.empty((lm["spkhist"].shape[0], totalIter))
    aS       = _N.empty((totalIter, lm["aS"].shape[1]))
    fs       = _N.empty((totalIter, lm["fs"].shape[1]))
    ws       = lm["ws"]
    mnStds   = _N.empty(totalIter)
    amps     = _N.empty((totalIter, lm["fs"].shape[1]))
    smpx     = lm["smpx"]
    us       = _N.empty((lm["u"].shape[0], totalIter))
    B        = lm["B"]

    it       = 0
    for i in xrange(len(its)):
        it0  = it
        it1  = it0 + its[i]
        it   += it1 - it0
        allalfas[it0:it1]    = lms[i]["allalfas"]
        q2[:, it0:it1]       = lms[i]["q2"]
        h_coeffs[:, it0:it1] = lms[i]["h_coeffs"]
        spkhist[:, it0:it1]  = lms[i]["spkhist"]
        aS[it0:it1]          = lms[i]["aS"]
        fs[it0:it1]          = lms[i]["fs"]
        mnStds[it0:it1]      = lms[i]["mnStds"]
        amps[it0:it1]        = lms[i]["amps"]
        us[:, it0:it1]       = lms[i]["u"]

    concat_pkl = {}
    concat_pkl["allalfas"]   = allalfas
    concat_pkl["q2"]         = q2
    concat_pkl["h_coeffs"]   = h_coeffs
    concat_pkl["Hbf"]        = Hbf
    concat_pkl["spkhist"]    = spkhist
    concat_pkl["aS"]         = aS
    concat_pkl["fs"]         = fs
    concat_pkl["ws"]         = ws
    concat_pkl["mnStds"]     = mnStds
    concat_pkl["amps"]       = amps
    concat_pkl["smpx"]       = smpx
    concat_pkl["u"]          = us
    concat_pkl["B"]          = B
    concat_pkl["t0_is_t_since_1st_spk"] = lms[0]["t0_is_t_since_1st_spk"]

    fn = resFN("%(ed)s/%(rd)s/Csmpls.dump" % {"ed" : exptDir, "rd" : runDir})
    dmp = open(fn, "wb")
    pickle.dump(concat_pkl, dmp, -1)
    dmp.close()

    ofn = resFN("%(ed)s/%(rd)s/fs_amps" % {"ed" : exptDir, "rd" : runDir})

    mF.plotFsAmpDUMP(concat_pkl, totalIter, 0, xticks=xticks, yticksFrq=None, yticksMod=None, yticksAmp=None, fMult=2, dir=None, fn=ofn, ylimAmp=ylimAmp)
