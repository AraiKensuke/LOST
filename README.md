#  Latent Oscillation in Spike Train (LOST)

#####  Kensuke Arai and Robert E. Kass

##  Introduction
The timing of spikes in a neural spike train are sometimes observed to fire preferentially at certain phases of oscillatory components of brain signals such as the LFP or EEG, ie may be said to have an oscillatory modulation.    However, the temporal relationship is not exact, and there exists considerable variability in the spike-phase relationship.  Because the spikes themselves are often temporally sparse, assessing whether the spike train has oscillatory modulation, is challenging.  Further, the oscillatory modulation may not be uniformly present throughout the entire experiment.  We therefore developed a method to detect and also infer the instantaneous phase of the oscillation in the spike train, and also detection for non-stationarity in the modulation of the spikes.

##  Necessary packages
PP-AR
PP-AR/runTemplate.py
PP-AR/runTemplateBM.py
PyPG

##  Setup
LOST is written in python and cython, and requires python2.7, matplotlib, numpy, scipy, patsy.  pyPG is primarily written in C++, so his will need to be compiled.

##  Data format
LOST data is a flat ASCII file.  If the data is a simulation where we know the modulatory signal and the conditional intensity function (CIF)
simulation data
x, CIF, spks
[simulated data example](examples1.html)
real data
x, filtered x, spks, ph
[real data example](examples2.html)

however, we caution that the ONLY data actually used for LOST model fitting, is the spikes

##  Setup
First, environment variables need to be set.  If you're using Linux or OS X and also bash or sh, in he .profile file

```
export __LOST_BaseDir__="/Users/arai/Workspace/LOST"
export __LOST_ResultDir__="${__LOST_BaseDir__}/Results"
```
To run LOST, go to RESULTS directory
```
import os
__LOST_BaseDir__   = os.getenv("__LOST_BaseDir__", "")
__LOST_ResultDir__ = os.getenv("__LOST_ResultDir__", "")
```

Next, create a directory where your data file xprbsdN.dat will be.  If you're creating simulated data, if you rename the startup script `sim1.py`, a directory called `sim` will be automatically created for you, and it will contain the created `xprbsdN.dat`.


##  Creating simulated data

##  Running
Whether using real or simulated data, the template `cpRunTemplate`



##  TODO
As it stands, LOST is computationally expensive, and we might need an hour or so of sampling for inference to be made.  A significant bottle neck has been identified (inverting the covariance matrix for the forward filter in FFBS), and a possible solution is to parallelization, since all the trials are independent.  However, this would require the use of a non-GIL matrix inversion function (currently GIL numpy.linalg.inv), we would need to go directly to LAPACK.  However, LAPACK is a bit tricky for novices to use directly, but we hope to implement this soon.