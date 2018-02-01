"""
The plot module provides support for virtualization of diffusion profile data
and diffusion coefficients data using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev
from pydiffusion.Dmodel import SF


def profileplot(profile, ax=None, **kwargs):
    """
    Plot diffusion profiles

    Parameters
    ----------
    profile : DiffProfile
        Diffusion profile object
    ax : matplotlib.Axes
        Default axes used if not specified
    kwargs : kwargs
        Passed to 'matplotlib.pyplot.plot'
    """
    dis, X = profile.dis, profile.X
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(dis, X, **kwargs)
    ax.set_xlabel('Distance (micron)', fontsize=15)
    ax.set_ylabel('Mole fraction', fontsize=15)
    ax.tick_params(labelsize=12)


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
        Passed to 'matplotlib.pyplot.semilogy'
    """
    X = profile.X
    sf = SF(profile, time, Xlim)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.semilogy(X, sf, **kwargs)
    ax.set_xlabel('Mole fraction', fontsize=15)
    ax.set_ylabel('Diffusion Coefficients '+'$\mathsf{(m^2/s)}$', fontsize=15)
    ax.tick_params(labelsize=12)


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
        Passed to 'matplotlib.pyplot.semilogy'
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    Np, Xr, fD = diffsys.Np, diffsys.Xr, diffsys.Dfunc

    # Diffusion Coefficients plot
    for i in range(Np):
        Xp = np.linspace(Xr[i, 0], Xr[i, 1], 100)
        Dp = np.exp(splev(Xp, fD[i]))
        if 'c' in kwargs or 'color' in kwargs:
            ax.semilogy(Xp, Dp, **kwargs)
        else:
            if i == 0:
                p = ax.semilogy(Xp, Dp, **kwargs)
            else:
                ax.semilogy(Xp, Dp, c=p[0].get_color(), **kwargs)

    # Error analysis result plot
    if err is not None:
        loc, errors = err.loc, err.errors
        for i in range(Np):
            pid = np.where((loc >= Xr[i, 0]) & (loc <= Xr[i, 1]))[0]
            Dloc = np.exp(splev(loc[pid], fD[i]))
            ax.semilogy(loc[pid], Dloc * 10**errors[pid, 0], 'r--', lw=2)
            ax.semilogy(loc[pid], Dloc * 10**errors[pid, 1], 'r--', lw=2)

    ax.set_xlabel('Mole fraction', fontsize=15)
    ax.set_ylabel('Diffusion Coefficients '+'$\mathsf{(m^2/s)}$', fontsize=15)
    ax.tick_params(labelsize=12)
