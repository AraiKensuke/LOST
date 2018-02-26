import numpy as _N
import matplotlib.pyplot as _plt
 
def raster(event_times_list, color='k', lw=1.5, t0=None, t1=None, w=10, h=8, filename=None, fsl=20, fst=20):
    """
    fsl = fontsize for label
    fst = fontsize for tick
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable    (list of arrays containing spike times)
    a list of event time iterables
    color : string
    color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    #ax = _plt.gca()
    fig = _plt.figure(figsize=(w, h))
    fig.subplots_adjust(hspace=0.15, wspace=0.15, left=0.15, right=0.95, bottom=0.15)
    for ith, trial in enumerate(event_times_list):
        _plt.vlines(trial, ith + .5, ith + 1.5, color=color, lw=lw)
    #_plt.ylim(0.5, len(event_times_list) + 0.5)
    _plt.ylim(-0.5, len(event_times_list) + 1.5)

    _plt.xlabel("Time (ms)", fontsize=fsl);
    _plt.ylabel("Trial #", fontsize=fsl)
    _plt.xticks(fontsize=fst);        _plt.yticks(fontsize=fst)

    if (t0 is not None) and (t1 is not None):
        _plt.xlim(t0-5, t1+5)
        
    if filename is not None:
        _plt.savefig(filename)
        _plt.close()

        
    #return ax 
