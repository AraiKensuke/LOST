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
LINK to example data file
real data
x, filtered x, spks, ph
LINK to example data file

however, we caution that the ONLY data actually used for LOST model fitting, is the spikes

##  Setup
