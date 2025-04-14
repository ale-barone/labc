import numpy as np
from ..data  import DataStats, DataErr
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.pyplot import *


# ==============================================================================
# GENERAL SETUP
# ==============================================================================

FIGSIZE = (10, 7)
FONTSIZE = {'S' : 22, 'M': 24, 'B': 26, 'BB': 28}
LABELS = {'x' : 't', 'y' : 'y', 'title' : 'Title'}

MARKERSIZE = 8
CAPSIZE = 3
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
    plt.rc('axes',   titlesize=FONTSIZE['BB'], labelsize=FONTSIZE['M'])    # fontsize of the axes (created with subplots)
    plt.rc('legend', fontsize=FONTSIZE['S'])    # legend fontsize

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'serif',
        "font.serif": 'cm',
    })

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
    out = axis.plot(*new_args, scalex=scalex, scaley=scaley, data=data, **kwargs)
    return out

   
def errorbar(
    x, y, yerr=None, xerr=None, fmt='', axis=None, ecolor=None,
    markersize=MARKERSIZE, elinewidth=ELINEWIDTH, capsize=CAPSIZE,
    barsabove=False, lolims=False,
    uplims=False, xlolims=False, xuplims=False, errorevery=1,
    capthick=None, *, data=None, **kwargs):
    
    if isinstance(x, (DataStats, DataErr)) and xerr is None:
        xerr = x.err
        x = x.mean
    if isinstance(y, (DataStats, DataErr)) and yerr is None:
        yerr = y.err
        y = y.mean  
    if axis is None:
        axis = plt      

    out = axis.errorbar(
        x, y, yerr=yerr, xerr=xerr, fmt=fmt, ecolor=ecolor,
        markersize=markersize, elinewidth=elinewidth, capsize=capsize,
        barsabove=barsabove, lolims=lolims,
        uplims=uplims, xlolims=xlolims, xuplims=xuplims, errorevery=errorevery,
        capthick=capthick, data=data, **kwargs
    )
    return out

def fill_error_band(x, y, axis=None, *args, **kwargs):
    y1 = y.mean-y.err
    y2 = y.mean+y.err
    if axis is None:
        axis = plt   
    return axis.fill_between(x, y1, y2, *args, **kwargs)

def fill_error_bandx(x, y, axis=None, *args, **kwargs):
    y1 = y.mean-y.err
    y2 = y.mean+y.err
    if axis is None:
        axis = plt   
    return axis.fill_betweenx(x, y1, y2, *args, **kwargs)

class _MyAxes(Axes):

    def __init__(self, ax):
        # copy attributes FIXME I don't know how to efficiently copy and make
        # all the attributes work properly (see for example workaround on legend..)
        self.__dict__ = ax.__dict__
        self.ax = ax 

    def plot(self, *args, scalex=True, scaley=True, data=None, **kwargs):
        out = plot(
            *args, axis=self.ax, scalex=scalex, scaley=scaley, data=data,
            **kwargs
        )
        return out

    def errorbar(
        self, x, y, yerr=None, xerr=None, fmt='', ecolor=None,
        markersize=MARKERSIZE, elinewidth=ELINEWIDTH, capsize=CAPSIZE,
        barsabove=False, lolims=False,
        uplims=False, xlolims=False, xuplims=False, errorevery=1,
        capthick=None, *, data=None, **kwargs):
        
        out = errorbar(
            x, y, yerr=yerr, xerr=xerr, fmt=fmt, axis=self.ax, ecolor=ecolor,
            markersize=markersize, elinewidth=elinewidth, capsize=capsize,
            barsabove=barsabove, lolims=lolims,
            uplims=uplims, xlolims=xlolims, xuplims=xuplims, errorevery=errorevery,
            capthick=capthick, data=data, **kwargs
        )
        return out   
    
    def fill_error_band(self, x, y, *args, **kwargs):
        return fill_error_band(x, y, axis=self.ax, *args, **kwargs)
    
    def fill_error_bandx(self, x, y, *args, **kwargs):
        return fill_error_bandx(x, y, axis=self.ax, *args, **kwargs)
    
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
            [_MyAxes(ax[r, c]) for c in range(ncols)]
            for r in range(nrows)
        ])
    # FIXME: fix tight layout
    fig.tight_layout(pad=5.0)
    return fig, ax
