"""
The Dmodel module provide tools to fit a smooth diffusion coefficient curve
based on Sauer-Fraise calculation results and the tools to adjust it.
"""

import numpy as np
from scipy.interpolate import splrep, splev, UnivariateSpline
from pydiffusion.core import DiffSystem
from pydiffusion.utils import disfunc


def SF(profile, time, Xlim=[]):
    """
    Use Sauer-Fraise method to calculate diffusion coefficients from profile.

    Parameters
    ----------
    profile : DiffProfile
        Diffusion profile.
    time : float
        Diffusion time in seconds.
    Xlim : list (float), optional
        Indicates the left and right concentration limits for calculation.
        Default value = [profile.X[0], profile.X[-1]].

    Returns
    -------
    DC : numpy.array
        Diffusion coefficients.
    """
    dis, X = profile.dis, profile.X
    XL, XR = X[0], X[-1] if Xlim == [] else Xlim
    Y1 = (X-XL)/(XR-XL)
    Y2 = 1-Y1
    dYds = (Y1[2:]-Y1[:-2])/(dis[2:]-dis[:-2])
    intvalue = np.array([Y2[i]*np.trapz(Y1[:i], dis[:i])+Y1[i]*(np.trapz(Y2[i:], dis[i:])) for i in range(1, len(dis)-1)])
    DC = intvalue/dYds/2/time*1e-12
    DC = np.append(DC[0], np.append(DC, DC[-1]))
    return DC


def Dpcalc(X, DC, Xp):
    """
    Based on Sauer-Fraise calculated results, this function provides good
    estimation of diffusion coefficients at location picked.

    Parameters
    ----------
    X, DC : 1d-array
        Sauer-Fraise calculated results.
    Xp : 1d-array
        Locations to estimate diffusion coefficients.

    Returns
    -------
    Dp : 1d-array
        Estimated diffusion coefficients.
    """
    if len(Xp) == 1:
        fD = splrep(X, np.log(DC), k=1)
        Dp = np.exp(splev(Xp, fD))
    else:
        Dp = np.zeros(len(Xp))
        for i in range(len(Xp)):
            mark = np.zeros(2)
            if i == 0:
                mark[0] = Xp[i]
            else:
                mark[0] = Xp[i-1]
            if i == len(Xp)-1:
                mark[1] = Xp[i]
            else:
                mark[1] = Xp[i+1]
            pid = np.where((X > mark[0]) & (X < mark[1]))[0]
            p = np.polyfit(X[pid], np.log(DC[pid]), 2)
            Dp[i] = np.exp(np.polyval(p, Xp[i]))
    return Dp


def fDC_spl(Xp, Dp):
    """
    Return a spline function to model diffusion coefficients.
    The function can be constant(1), linear(2) or quadratic(>2) depending on
    the length of Xp.

    Parameters
    ----------
    Xp : 1d-array
        Composition list.
    Dp : 1d-array
        Corresponding diffusion coefficients at Xp.
    """
    if len(Xp) == 1:
        fDC = splrep([Xp, Xp*1.01], [np.log(Dp), np.log(Dp)], k=1)
    elif len(Xp) == 2:
        fDC = splrep(Xp, np.log(Dp), k=1)
    else:
        fDC = splrep(Xp, np.log(Dp), k=2)
    return fDC


def fDC_uspl(X, DC, Xp, Xr):
    """
    Use UnivariateSpline to model diffusion coefficients.

    Parameters
    ----------
    X, DC : 1d-array
        Diffusion coefficients data.
    Xp : 1d-array with shape (1, 2)
        UnivariateSpline range of X.
    Xr : 1d-array with shape (1, 2)
        Expanded range of UnivariateSpline, usually is the phase range.
    """
    pid = np.where((X >= Xp[0]) & (X <= Xp[1]))[0]
    fDC = UnivariateSpline(X[pid], np.log(DC[pid]), bbox=[Xr[0], Xr[1]], k=2)
    Xf = np.linspace(Xr[0], Xr[1], 30)
    return splrep(Xf, fDC(Xf), k=2)


def DCadjust(profile_ref, profile_sim, diffsys, ph, Xp=None, pp=True, deltaD=None, r=0.02):
    """
    Adjust diffusion coefficient fitting function by comparing simulated
    profile against reference profile. The purpose is to let simulated
    diffusion profile be similar to reference profile.

    Parameters
    ----------
    profile_ref : DiffProfile
        Reference diffusion profile
    profile_sim : DiffProfile
        Simulated diffusion profile
    diffsys : DiffSystem
        Diffusion system
    ph : int
        Phase # to be adjusted, 0 <= ph <= diffsys.Np-1
    Xp : 1d-array
        Reference composition to adjust their corresponding diffusivities.
        If provided, spline function Dfunc must be determined by [Xp, Dp]
        alone, where Dp = exp(Dfunc(Xp)).
    pp : bool, optional
        Per-point mode (True) or whole-phase mode (False). Per-point mode
        adjusts each Dp at Xp by itself. In whole-phase mode, all Dp are
        adjusted by the same rate, i.e. the diffusivity curve shape won't
        change.
    deltaD: float, optional
        Only useful at whole-phase mode. deltaD gives the rate to change
        diffusion coefficients DC. DC = DC * 10^deltaD
    r : float, optional
        Only useful at per-phase mode, default = 0.02, 0 < r < 1. r gives the
        range to calculate the concentration gradient around X, [X-r, X+r].

    """
    dref, Xref, Ifref = profile_ref.dis, profile_ref.X, profile_ref.If
    dsim, Xsim, Ifsim = profile_sim.dis, profile_sim.X, profile_sim.If

    if ph >= diffsys.Np:
        raise ValueError('Incorrect phase #, 0 <= ph <= %i' % diffsys.Np-1)
    if pp and Xp is None:
        raise ValueError('Xp must be provided in per-point mode')

    Dfunc, Xr, Np = diffsys.Dfunc[ph], diffsys.Xr[ph], diffsys.Np
    rate = 1

    # If there is phase consumed, increase adjustment rate
    if len(Ifref) != len(Ifsim):
        print('Phase consumed found, increase adjustment rate')
        rate = 2

    idref = np.where((Xref >= Xr[0]) & (Xref <= Xr[1]))[0]
    idsim = np.where((Xsim >= Xr[0]) & (Xsim <= Xr[1]))[0]

    if Xp is None:
        Xp = np.linspace(Xr[0], Xr[1], 30)
    Dp = np.exp(splev(Xp, Dfunc))

    # If this is consumed phase, increase DC by 2 or 10^deltaD
    if len(idsim) == 0:
        Dp = np.exp(splev(Xp, Dfunc))
        if deltaD is None:
            return splrep(Xp, np.log(Dp*2), k=2)
        else:
            return splrep(Xp, np.log(Dp*10**deltaD), k=2)

    dref, Xref = dref[idref], Xref[idref]
    dsim, Xsim = dsim[idsim], Xsim[idsim]

    # Per phase adjustment
    if not pp:
        if deltaD is not None:
            return splrep(Xp, np.log(Dp*10**deltaD), k=2)

        # Calculate deltaD by phase width
        # When it comes to first or last phase, data closed to end limits are not considered
        fdis_ref = disfunc(dref, Xref)
        fdis_sim = disfunc(dsim, Xsim)
        if ph == 0:
            X1 = Xr[0]*0.99 + Xr[1]*0.01
        if ph == Np-1:
            X2 = Xr[0]*0.01 + Xr[1]*0.99
        dref = splev([X1, X2], fdis_ref)
        dsim = splev([X1, X2], fdis_sim)
        wref = dref[1]-dref[0]
        wsim = dsim[1]-dsim[0]
        Dp *= np.sqrt(wref/wsim)
        return splrep(Xp, np.log(Dp), k=2)

    # Per point adjustment
    for i in len(Xp):
        # X1, X2 is the lower, upper bound to collect profile data
        # X1, X2 cannot exceed phase bound Xr
        X1, X2 = max(Xp[i]-r, Xr[0]), min(Xp[i]+r, Xr[1])

        # Calculate the gradient inside [X1, X2] by linear fitting
        fdis_ref = disfunc(dref, Xref)
        fdis_sim = disfunc(dsim, Xsim)
        Xf = np.linspace(X1, X2, 10)
        pref = np.polyfit(splev(Xf, fdis_ref), Xf, 1)[0]
        psim = np.polyfit(splev(Xf, fdis_sim), Xf, 1)[0]

        # Adjust DC by gradient difference
        Dp[i] *= (psim/pref)**rate
    return splrep(Xp, np.log(Dp), k=2)


def Dmodel(profile, time, Xlim=[]):
    dis, X = profile.dis, profile.X
    DC = SF(profile, time, Xlim)
    Xlim = [X[0], X[-1]] if Xlim == [] else Xlim
    Xr = np.array(Xlim)
    for i in range(len(dis)-1):
        if dis[i] == dis[i+1]:
            Xr = np.insert(Xr, -1, [X[i], X[i+1]])
    Np = len(Xr)//2
    Xr = Xr.reshape(Np, 2)
    fD = [0]*Np
    for i in range(Np):
        msg = 'Enter the UnivariateSpline factor p [0.9] for phase %i' % (i+1)
        msg += ' (p=0.9 means UnivariateSpline only fit 5-95% of data):\n'
        ipt = input(msg)
        p = float(ipt) if ipt != '' else 0.9
        Xdiff = Xr[i, 1]-Xr[i, 0]
        start, end = Xr[i, 0]+Xdiff*(1-p)/2, Xr[i, 1]-Xdiff*(1-p)/2
        pid = np.where((X >= start) & (X <= end))[0]
        fDC = UnivariateSpline(X[pid], np.log(DC[pid]), bbox=[Xr[i, 0], Xr[i, 1]], k=2)
        Xf = np.linspace(Xr[i, 0], Xr[i, 1], 20)
        fD[i] = splrep(Xf, fDC(Xf), k=2)
    return DiffSystem(Xr, Dfunc=fD)
