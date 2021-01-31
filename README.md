#  Latent Oscillation in Spike Train (LOST)

#####  Kensuke Arai and Robert E. Kass
[Inferring oscillatory modulation in neural spike trains](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005596) (2017)

##  Introduction
The timing of spikes in a neural spike train are sometimes observed to fire preferentially at certain phases of oscillatory components of brain signals such as the LFP or EEG, ie may be said to have an oscillatory modulation.    However, the temporal relationship is not exact, and there exists considerable variability in the spike-phase relationship.  Because the spikes themselves are often temporally sparse, assessing whether the spike train has oscillatory modulation, is challenging.  Further, the oscillatory modulation may not be uniformly present throughout the entire experiment.  We therefore developed a method to detect and also infer the instantaneous phase of the oscillation in the spike train, and also detection for non-stationarity in the modulation of the spikes.

##  Run LOST on Google Colab directly by following links in this repository.
_**LOST can be run entirely on Colab without having to download or install the software following the links in this repository**_.  We provide several example datasets and analysis results, as the inference results require some interpretation.  The examples will illustrate how to interpret the results, and also some guidelines in the choice of parameters that can affect the inferred oscillations.  Notebooks are found in the *Notebooks* directory above, as are the example data.  In particular, here are the parameters and settings that most influence the fit:

* Item 1  **AR(p) innovation variance**
Generally, allowing the innovation variance to be too large can result in a model of spiking probability that is near 0 where there are no spikes, and nearly 1 at spike times, and a very long post-spike history timescale will cause the oscillation to be partly explained by the post-spike history. A large innovation variance can cause large fluctuations in the AR(p), coupled with a large, negative offset and the nonlinearity in the link function, can result in probability being near 1 only near the spikes.  This kind of fit is easy to spot, and there are 2 hyperparamters 

* Item 2  **Timescale and knot locations of the post-spike history dependence**
We first address the AR(p) innovation variance.  

Both the post-spike history and the oscillation explain the history dependence of spiking, but there is no definitive demaraction between these two.  We included the post-spike history to account for the refractory period, a property of the neuron and its membrane excitability itself - something that can be observed in isolated electrophysiological experiments.  The oscillation we believe to be a network effect.  The effect of the latter can be captured somewhat by the fixed post-spike history.  We assume that the post-spike history is a more stereotyped effect, and sufficiently described by a fixed function of time after the spike, while the oscillation has more variability

