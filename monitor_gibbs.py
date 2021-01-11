import numpy as _N
from filter import gauKer#, contiguous_pack2
import matplotlib.pyplot as _plt
import scipy.special as _ssp
import scipy.stats as _ss
#from scipy.spatial.distance import cdist, euclidean

_GAMMA         = 0
_INV_GAMMA     = 1

def stationary_from_Z_bckwd(smps, blksz=200):
    #  Detect when stationarity reached in Gibbs sampling
    #  Also, detect whether cluster is switching between local extremas
    #
    SMPS   = smps.shape[1]   #     smps   is 3 x ITERS

    wins         = SMPS/blksz
    wins_m1      = wins - 1

    print(wins*blksz)
    print(SMPS)
    rshpd     = smps.reshape((3, wins, blksz))

    mrshpd    = _N.median(rshpd, axis=2)   #  2 x wins_m1+1 x M  
    sdrshpd   = _N.std(rshpd, axis=2)

    mLst                =         mrshpd[:, wins_m1].reshape(3, 1)
    sdLst               =         sdrshpd[:, wins_m1].reshape(3, 1)
    sdNLst               =         sdrshpd[:, 0:-1].reshape(3, wins_m1)

    zL                =         (mrshpd[:, 0:-1] - mLst)/sdLst
    zNL               =         (mrshpd[:, 0:-1] - mLst)/sdNLst

    #  mn, std in each win
    #  u1=0.3, sd1=0.9      u2=0.4, sd2=0.8
    #  (u2-u1)/sd1  

    #  detect sudden changes


    win1stFound=(wins_m1-1)*blksz
    sameDist = 0
    i = 0

    thisWinSame     = False
    lastWinSame     = 0#wins_m1

    #  want 3 consecutive windows where distribution looks different

    #  amp and mnStd a bit more variable, so allow bigger deviations in previous windows, still consider similar distribution as last window
    bigExc = _N.array([0.75, 0.75, 0.75])   
    while (sameDist <= 3) and (i < wins_m1-1):
        i += 1
        it0 = i*blksz
        it1 = (i+1)*blksz

        thisWinSame = 0

        for d in range(3):
            bE = bigExc[d]
            if ((zL[d, i] < bE) and (zL[d, i] > -bE)) and \
               ((zNL[d, i] < bE) and (zNL[d, i] > -bE)):
                #  past window of 1 param is far from most recent window
                thisWinSame += 1

        if thisWinSame == 3:  #  all 3 wins must look similar
            if sameDist == 0:
                win1stFound = it0   #  start 
            lastWinSame = i

            sameDist += 1

        if (i - lastWinSame > 1) and (sameDist <= 2):
            #  there was a break in sterak of windows similar to last window
            sameDist = 0   #  reset
            win1stFound = (wins_m1-1)*blksz

    frms = win1stFound

    return frms+blksz


