import numpy as np
from ..data  import DataStats
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.pyplot import *


# ==============================================================================
# GENERAL SETUP
# ==============================================================================

FIGSIZE = (10, 7)
FONTSIZE = {'S' : 16, 'M': 18, 'B': 20, 'BB': 22}
LABELS = {'x' : 't', 'y' : 'y', 'title' : 'Title'}

MARKERSIZE = '6'
CAPSIZE = 2
ELINEWIDTH = 0.9

# ==============================================================================
# INITIALIZATION
# ==============================================================================

# INITIALIZATION FOR PLOT 
def init_plot(*args, **kwargs):
    #plt.rcParams.update(mpl.rcParamsDefault)
    # see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html
    # see https://matplotlib.org/stable/api/matplotlib_configuration_api.html
    
    plt.rc('figure', titlesize=FONTSIZE['BB'], figsize=FIGSIZE)  # fontsize of the figure    
    plt.rc('axes',   titlesize=FONTSIZE['BB'], labelsize=FONTSIZE['S'])    # fontsize of the axes (created with subplots)
    plt.rc('legend', fontsize=FONTSIZE['S'])    # legend fontsize

    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "Helvetica",
    #     #"font.sans-serif": ["Helvetica"]
    # })
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Palatino"],
    # })
    
    # TODO: find a way to initialize plt.tight_layout() that holds for every plot
        
    #plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('xtick', labelsize=FONTSIZE['S'])    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONTSIZE['S'])    # fontsize of the tick labels

init_plot()



# IT WOULD BE NICE TO HAVE A TRACK OF THE AXIS I DEFINED, LIKE FOR EX axes_init
# THIS WOULD ALLOW ME TO APPLY COMMON CHANGES TO ALL OF THEM WITHOUT HAVING TO
# DO IT MANUALLY FOR EACH ONE...maybe set attributes of the object (should I define a class?)

def plot(*args, axis=None, scalex=True, scaley=True, data=None, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, DataStats):
            arg = arg.mean
        new_args.append(arg)
    if axis is None:
        axis = plt
    axis.plot(*new_args, scalex=scalex, scaley=scaley, data=data, **kwargs)

   
def errorbar(
    x, y, yerr=None, xerr=None, fmt='', axis=None, ecolor=None,
    elinewidth=ELINEWIDTH, capsize=CAPSIZE, barsabove=False, lolims=False,
    uplims=False, xlolims=False, xuplims=False, errorevery=1,
    capthick=None, *, data=None, **kwargs):
    
    if isinstance(x, DataStats) and xerr is None:
        xerr = x.err
        x = x.mean
    if isinstance(y, DataStats) and yerr is None:
        yerr = y.err
        y = y.mean  
    if axis is None:
        axis = plt      

    axis.errorbar(
        x, y, yerr=yerr, xerr=xerr, fmt=fmt, ecolor=ecolor,
        elinewidth=elinewidth, capsize=capsize, barsabove=barsabove, lolims=lolims,
        uplims=uplims, xlolims=xlolims, xuplims=xuplims, errorevery=errorevery,
        capthick=capthick, data=data, **kwargs
    )


class _MyAxes(Axes):

    def __init__(self, ax):
        # copy attributes
        self.__dict__ = ax.__dict__.copy()
        self.ax = ax

    def plot(self, *args, scalex=True, scaley=True, data=None, **kwargs):
        out = plot(
            *args, axis=self.ax, scalex=scalex, scaley=scaley, data=data,
            **kwargs
        )
        return out

    def errorbar(
        self, x, y, yerr=None, xerr=None, fmt='', ecolor=None,
        elinewidth=ELINEWIDTH, capsize=CAPSIZE, barsabove=False, lolims=False,
        uplims=False, xlolims=False, xuplims=False, errorevery=1,
        capthick=None, *, data=None, **kwargs):
        
        out = errorbar(
            x, y, yerr=yerr, xerr=xerr, fmt=fmt, axis=self.ax, ecolor=ecolor,
            elinewidth=elinewidth, capsize=capsize, barsabove=barsabove, lolims=lolims,
            uplims=uplims, xlolims=xlolims, xuplims=xuplims, errorevery=errorevery,
            capthick=capthick, data=data, **kwargs
        )
        return out
            
    
# INITIALIZATION FOR AXES
def init_axes(nrows=1, ncols=1, figsize=FIGSIZE, *args, **kwargs):
    #init_plot()
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols , figsize[1]*nrows),  *args, **kwargs)

    if nrows==1 and ncols==1:
        ax = _MyAxes(ax)
    elif nrows>1 and ncols==1:
        ax = np.array([_MyAxes(ax[r]) for r in range(nrows)])
    elif nrows==1 and ncols>1:
        ax = np.array([_MyAxes(ax[c]) for c in range(ncols)])
    else:
        ax = np.array([
            [_MyAxes(ax[r, c]) for r in range(nrows)]
            for c in range(ncols)
        ])
    # FIXME: fix tight layout
    fig.tight_layout(pad=5.0)
    return fig, ax
