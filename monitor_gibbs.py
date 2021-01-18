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

    wins         = SMPS//blksz
    wins_m1      = wins - 1

    rshpd     = smps.reshape((3, wins, blksz))

    mrshpd    = _N.mean(rshpd, axis=2)   #  2 x wins_m1+1 x M  
    sdrshpd   = _N.std(rshpd, axis=2)

    mLst                =         mrshpd[:, wins_m1].reshape(3, 1)
    sdLst               =         sdrshpd[:, wins_m1].reshape(3, 1)
    sdNLst               =         sdrshpd[:, 0:-1].reshape(3, wins_m1)

    zL                =         (mrshpd[:, 0:-1] - mLst)//sdLst
    zNL               =         (mrshpd[:, 0:-1] - mLst)//sdNLst

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
    bigExc = _N.array([1., 1., 1.])   
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

def shifted_arrays(inarray, pcs):
    N = inarray.shape[0]
    out_array = _N.empty((pcs, N), dtype=_N.int)
    blk = N // pcs
    for sh in range(pcs):
        for pc in range(pcs):
            spc1 = (pc+sh)
            spc2 = (pc+sh+1)
            if spc1 >= pcs:
                spc1 = spc1 % pcs
                spc2 = spc2 % pcs
            out_array[sh, spc1*blk:spc2*blk] = inarray[pc*blk:(pc+1)*blk]
    return out_array
    
#def stationary_test(amp0s, f0s, mnstds, currit, blksz=200, pts=20):
def stationary_test(amp0s, f0s, mnstds, currit, blocksize=200, points=20):
    #  stationary if over 3000 iterations, it is stationary
    #  we 
    b_amps   = _N.mean(amp0s[currit - points*blocksize:currit].reshape((points, blocksize)), axis=1)
    b_fs     = _N.mean(f0s[currit - points*blocksize:currit].reshape((points, blocksize)), axis=1)
    b_mnstds = _N.mean(mnstds[currit - points*blocksize:currit].reshape((points, blocksize)), axis=1)

    xpts = _N.arange(points)
    xpts4= shifted_arrays(xpts, 4)

    stat = 0
    for pc in range(4):
        pc_amps, pv_amps = _ss.pearsonr(xpts4[pc], b_amps)
        pc_fs, pv_fs = _ss.pearsonr(xpts4[pc], b_fs)
        pc_mnstds, pv_mnstds = _ss.pearsonr(xpts4[pc], b_mnstds)

        if (pv_mnstds > 0.01) and (pv_amps > 0.01) and (pv_fs > 0.01):
            stat += 1
        if stat == 4:
            return True
    return False
