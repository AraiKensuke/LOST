#  Latent Oscillation in Spike Train (LOST)

#####  Kensuke Arai and Robert E. Kass

##  Introduction
The timing of spikes in a neural spike train are sometimes observed to fire preferentially at certain phases of oscillatory components of brain signals such as the LFP or EEG, ie may be said to have an oscillatory modulation.    However, the temporal relationship is not exact, and there exists considerable variability in the spike-phase relationship.  Because the spikes themselves are often temporally sparse, assessing whether the spike train has oscillatory modulation, is challenging.  Further, the oscillatory modulation may not be uniformly present throughout the entire experiment.  We therefore developed a method to detect and also infer the instantaneous phase of the oscillation in the spike train, and also detection for non-stationarity in the modulation of the spikes.

##  Required packages
numpy
matplotlib
cython

PP-AR

[patsy](https://patsy.readthedocs.io/en/latest/)   used for splines to describe trial-averaged effect

[PyPG](https://github.com/AraiKensuke/pyPG)  Polya-Gamma random variables

##  Recommended packages 
[LOST_Results_exmplr](https://github.com/AraiKensuke/LOST_Results_exmplr)  run-script generator for LOST, as well as examples of a datafile and a HOWTO of how to run LOST and format input data.

##  Setup
LOST is written in python and cython, and requires python2.7, matplotlib, numpy, scipy, patsy and pyPG.  [pyPG](https://github.com/AraiKensuke/pyPG) is primarily written in C++ with python wrappers, and comes with its own python setup script, so please build it separately.

Once pyPG is built, LOST can be built and installed.  LOST is mainly written in python, with 1 cython file.  Choosing a destination directory `$LOSTDIR` and extract (doing the tar.gz case here)

```
gunzip LOST-x.tar.gz
tar xf LOST-x.tar
cd LOST-x
python setup.py install
```

It is now recommended to obtain [LOST_Results_exmplr](https://github.com/AraiKensuke/LOST_Results_exmplr).  It will walk you through how to set up a LOST analysis.

##  TODO
As it stands, LOST is computationally expensive, and we might need an hour or so of sampling for inference to be made.  A significant bottle neck has been identified (inverting the covariance matrix for the forward filter in FFBS), which takes around 80% of the processing time, and a possible solution is to parallelization, since all the trials are independent.  However, this would require the use of a non-GIL matrix inversion function (currently GIL numpy.linalg.inv), we would need to go directly to LAPACK.  However, LAPACK is a bit tricky for novices to use directly, but we hope to implement this soon.


