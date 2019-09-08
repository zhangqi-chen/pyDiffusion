"""
The plot module provides support for visualization of diffusion profile data
and diffusion coefficients data using matplotlib.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev
from pydiffusion.utils import SF

# Default label font size
label_fontsize = 13
# Default tick font size
tick_fontsize = 11
# Default legend font size
leg_fontsize = 13


def plot_lim(x1, x2, log=False):
    """
    Automatically generate plot range for diffusion plots

    Parameters
    ----------
    x1, x2 : float
        Plot range, can be concentration or diffusivity values.
    log : bool, optional
        Range in log scale or not.

    Returns
    -------
    x1_lim, x2_lim : float
        Plot range output, used for plt.xlim or plt.set_ylim.

    """
    x1, x2 = min(x1, x2), max(x1, x2)
    if not log:
        x1_lim = (x1*100//1)/100
        x2_lim = (x2*100//1+1)/100
    else:
        x1_lim = 10**np.floor(np.log10(x1))
        x2_lim = 10**(np.floor(np.log10(x2))+1)
    return x1_lim, x2_lim


def profileplot(profile, ax=None, err=None, **kwargs):
    """
    Plot diffusion profiles

    Parameters
    ----------
    profile : DiffProfile
        Diffusion profile object
    ax : matplotlib.Axes
        Default axes used if not specified
    kwargs : kwargs
        Passed to 'matplotlib.pyplot.plot', add label to name the profile.

    Returns
    -------
    matplotlib AxesSubplot

    Examples
    --------
    Plot two profiles on one axis:

    >>> ax = profileplot(profile1, c='b', label='profile1')
    >>> profileplot(profile2, ax, c='r', label='profile2')

    """
    dis, X = profile.dis, profile.X
    clw = {'lw': 2, 'label': profile.name}
    args = {**clw, **kwargs}
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(dis, X, **args)

    # Error analysis result plot
    if err is not None:
        profiles = err.profiles['error']
        disp = np.linspace(dis[0], dis[-1], 1e4)
        Xp = np.zeros((2, len(disp)))
        for i in range(2):
            Xp[i] = splev(disp, profiles[i])
        p = ax.plot(disp, Xp[0], '--', lw=2, label='Error')
        ax.plot(disp, Xp[1], '--', c=p[0].get_color(), lw=2)

    ax.set_xlabel('Distance (micron)', fontsize=label_fontsize)
    ax.set_ylabel('Mole fraction', fontsize=label_fontsize)
    ax.set_xlim(dis.min(), dis.max())
    ax.set_ylim(plot_lim(X.min(), X.max()))
    ax.tick_params(labelsize=tick_fontsize)
    leg = ax.legend(numpoints=1, fontsize=leg_fontsize)
    leg.get_frame().set_linewidth(0.0)
    leg.set_draggable(True)
    plt.tight_layout()

    return ax


def SFplot(profile, time, Xlim=[], ax=None, **kwargs):
    """
    Plot Sauer-Fraise calculated diffusion coefficients

    Parameters
    ----------
    profile : DiffProfile
        Diffusion profile object, passed to 'pydiffusion.Dmodel.SF'
    time : float
        Passed to 'pydiffusion.Dmodel.SF'
    Xlim : list
        Passed to 'pydiffusion.Dmodel.SF'
    ax : matplotlib.Axes
        Default axes used if not specified
    kwargs : kwargs
        Passed to 'matplotlib.pyplot.semilogy', , add label to name the D curve.

    Returns
    -------
    matplotlib AxesSubplot

    """
    X = profile.X
    sf = SF(profile, time, Xlim)
    clw = {'marker': '.', 'ls': 'none'}
    if 'label' not in kwargs:
        clw['label'] = profile.name+'_%.1fh_SF' % (time/3600)
    args = {**clw, **kwargs}
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.semilogy(X, sf, **args)
    ax.set_xlabel('Mole fraction', fontsize=label_fontsize)
    ax.set_ylabel('Diffusion Coefficients '+'$(m^2/s)$', fontsize=label_fontsize)
    ax.set_xlim(plot_lim(X.min(), X.max()))
    ax.tick_params(labelsize=tick_fontsize)
    leg = ax.legend(numpoints=1, fontsize=leg_fontsize)
    leg.get_frame().set_linewidth(0.0)
    leg.set_draggable(True)
    plt.tight_layout()

    return ax


def DCplot(diffsys, ax=None, err=None, **kwargs):
    """
    Plot diffusion coefficients

    Parameters
    ----------
    diffsys : DiffProfile
        Diffusion system object
    ax : matplotlib.Axes
        Default axes used if not specified
    err : DiffError
        Error analysis result
    kwargs : kwargs
        Passed to 'matplotlib.pyplot.semilogy', add label to name the D curve.

    Returns
    -------
    matplotlib AxesSubplot

    Compare two diffusion coefficients data on one axis:

    >>> ax = DCplot(dsys1, c='b', label='DC1')
    >>> DCplot(dsys2, ax, c='r', label='DC2')

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    Np, Xr, fD = diffsys.Np, diffsys.Xr, diffsys.Dfunc
    clw = {'lw': 2, 'label': diffsys.name}
    args = {**clw, **kwargs}
    # Diffusion Coefficients plot
    for i in range(Np):
        Xp = np.linspace(Xr[i, 0], Xr[i, 1], 51)
        Dp = np.exp(splev(Xp, fD[i]))
        if i == 0:
            Dmin, Dmax = min(Dp), max(Dp)
            p = ax.semilogy(Xp, Dp, **args)
            clw_nl = {'c': p[0].get_color(), 'label': '_nolegend_'}
            args_nl = {**args, **clw_nl}
        else:
            Dmin, Dmax = min(Dmin, min(Dp)), max(Dmax, max(Dp))
            ax.semilogy(Xp, Dp, **args_nl)
    plt.tight_layout()

    # Error analysis result plot
    if err is not None:
        loc, errors = err.loc, err.errors
        for i in range(Np):
            pid = np.where((loc >= Xr[i, 0]) & (loc <= Xr[i, 1]))[0]
            Dloc = np.exp(splev(loc[pid], fD[i]))
            if i == 0:
                p = ax.semilogy(loc[pid], Dloc * 10**errors[pid, 0], '--', lw=2, label='Error')
            else:
                p = ax.semilogy(loc[pid], Dloc * 10**errors[pid, 0], '--', lw=2)
            ax.semilogy(loc[pid], Dloc * 10**errors[pid, 1], '--', c=p[0].get_color(), lw=2)

    ax.set_xlim(plot_lim(Xr[0, 0], Xr[-1, 1]))
    ax.set_ylim(plot_lim(Dmin, Dmax, True))
    ax.set_xlabel('Mole fraction', fontsize=label_fontsize)
    ax.set_ylabel('Diffusion Coefficients '+'$(m^2/s)$', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    leg = ax.legend(numpoints=1, fontsize=leg_fontsize)
    leg.get_frame().set_linewidth(0.0)
    leg.set_draggable(True)
    plt.tight_layout()

    return ax


def colorcalc(X1, X2, r=1):
    """
    Calculate RGB color based on ternary compositions.

    Parameters
    ----------
    X1, X2 : array-like
        Composition 2D arrays.
        X1 = 1 corresponds to red.
        X2 = 1 corresponds to green.
        X1 = X2 = 0 corresponds to blue.
    r : float, >0
        r controls the color sensitivity to compositions.
        Higher r, higher sensitivity.

    """
    X1[X1 < 0], X2[X2 < 0] = 0, 0
    X1[X1 > 1], X2[X2 > 1] = 1, 1
    X3 = 1-X1-X2
    c1, c2, c3 = np.copy(X1), np.copy(X2), np.copy(X3)
    for i, (x1, x2, x3) in enumerate(itertools.permutations([X1, X2, X3])):
        ip = (x1 <= x2) & (x2 <= x3) & (x3 < 1)
        x, y = x3[ip]+.5*x1[ip], x1[ip]*3**.5/2
        p = (x-.5)/.5
        q = (.5-p**r)/(1-x)
        xn, yn = 1-q*(1-x), y*q
        a, b, c = yn*2/3**.5, 1-xn-yn/3**.5, xn-yn/3**.5
        if i == 0:
            c1[ip], c2[ip], c3[ip] = a, b, c
        elif i == 1:
            c1[ip], c3[ip], c2[ip] = a, b, c
        elif i == 2:
            c2[ip], c1[ip], c3[ip] = a, b, c
        elif i == 3:
            c2[ip], c3[ip], c1[ip] = a, b, c
        elif i == 4:
            c3[ip], c1[ip], c2[ip] = a, b, c
        else:
            c3[ip], c2[ip], c1[ip] = a, b, c
    color = np.stack([c1, c2, c3], axis=2)
    e = 1/color.max(2)
    return color*np.stack([e, e, e], axis=2)
