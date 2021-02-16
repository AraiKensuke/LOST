arInd = range(C)

for c in arInd:   #  Filtering signal root first drastically alters the strength of the signal root upon update sometimes.  
        # rprior = prior
        # if c >= Cs:
        #    rprior = _cd.__COMP_REF__
        tttA = _tm.time()
        if c >= Cs:
            ph0L = -1
            ph0H = 0
        else:
            ph0L = sig_ph0L   # 
            ph0H = sig_ph0H #  R=0.97, reasonably oscillatory
            
        j = 2*c + 1
        p1a =  AR2lims[c, 0]
        p1b =  AR2lims[c, 1]

        # given all other roots except the jth.  This is PHI0
        jth_r1 = alpC.pop(j)    #  negative root   #  nothing is done with these
        jth_r2 = alpC.pop(j-1)  #  positive root
