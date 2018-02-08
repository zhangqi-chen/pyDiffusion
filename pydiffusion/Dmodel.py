"""
The Dmodel module provides tools to fit a smooth diffusion coefficient curve
based on Sauer-Fraise calculation results and the tools to adjust it.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, UnivariateSpline
from pydiffusion.core import DiffSystem
from pydiffusion.utils import disfunc
from pydiffusion.io import ita_start, ita_finish, ask_input


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
    try:
        time = float(time)
    except TypeError:
        print('Cannot convert time to float')

    dis, X = profile.dis, profile.X
    [XL, XR] = [X[0], X[-1]] if Xlim == [] else Xlim
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


def Dfunc_spl(Xp, Dp):
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
        fDC = splrep([Xp[0], Xp[0]*1.01], [np.log(Dp[0]), np.log(Dp[0])], k=1)
    elif len(Xp) == 2:
        fDC = splrep(Xp, np.log(Dp), k=1)
    else:
        fDC = splrep(Xp, np.log(Dp), k=2)
    return fDC


def Dfunc_uspl(X, DC, Xp, Xr):
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
    pid = np.where((X >= Xp[0]) & (X <= Xp[-1]))[0]
    fDC = UnivariateSpline(X[pid], np.log(DC[pid]), bbox=[Xr[0], Xr[1]], k=2)
    Xf = np.linspace(Xr[0], Xr[1], 30)
    return splrep(Xf, fDC(Xf), k=2)


def Dadjust(profile_ref, profile_sim, diffsys, ph, pp=True, deltaD=None, r=0.02):
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
    if pp and 'Xspl' not in dir(diffsys):
        raise ValueError('diffsys must have Xspl properties in per-point mode')

    Dfunc, Xr, Np = diffsys.Dfunc[ph], diffsys.Xr[ph], diffsys.Np
    rate = 1

    # If there is phase consumed, increase adjustment rate
    if len(Ifref) != len(Ifsim):
        print('Phase consumed found, increase adjustment rate')
        rate = 2

    idref = np.where((Xref >= Xr[0]) & (Xref <= Xr[1]))[0]
    idsim = np.where((Xsim >= Xr[0]) & (Xsim <= Xr[1]))[0]

    if 'Xspl' in dir(diffsys):
        Xp = diffsys.Xspl[ph]
    else:
        Xp = np.linspace(Xr[0], Xr[1], 30)
    Dp = np.exp(splev(Xp, Dfunc))

    # If this is consumed phase, increase DC by 2 or 10^deltaD
    if len(idsim) == 0:
        Dp = np.exp(splev(Xp, Dfunc))
        if deltaD is None:
            return Dfunc_spl(Xp, Dp*2)
        else:
            return Dfunc_spl(Xp, Dp*10**deltaD)

    dref, Xref = dref[idref], Xref[idref]
    dsim, Xsim = dsim[idsim], Xsim[idsim]

    # Per phase adjustment
    if not pp:
        if deltaD is not None:
            return Dfunc_spl(Xp, Dp*10**deltaD)

        # Calculate deltaD by phase width
        # When it comes to first or last phase, data closed to end limits are not considered
        fdis_ref = disfunc(dref, Xref)
        fdis_sim = disfunc(dsim, Xsim)
        X1, X2 = Xr[0], Xr[1]
        if ph == 0:
            X1 = Xr[0]*0.9 + Xr[1]*0.1
        if ph == Np-1:
            X2 = Xr[0]*0.1 + Xr[1]*0.9
        ref = splev([X1, X2], fdis_ref)
        sim = splev([X1, X2], fdis_sim)
        wref = ref[1]-ref[0]
        wsim = sim[1]-sim[0]
        Dp *= np.sqrt(wref/wsim)
        return Dfunc_spl(Xp, Dp)

    # Per point adjustment
    for i in range(len(Xp)):
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
    return Dfunc_spl(Xp, Dp)


def Dmodel(profile, time, Xspl=None, Xlim=[]):
    """
    Given the diffusion profile and diffusion time, modeling the diffusion
    coefficients for each phase.

    Parameters
    ----------
    profile : DiffProfile
        Diffusion profile. Multiple phase profile must be after datasmooth to
        identify phase boundaries.
    time : float
        Diffusion time
    Xlim : list, optional
        Left and Right limit of diffusion coefficients. Xlim is also passed to
        SF function to calculate diffusion coefficients initially.
    """
    if not isinstance(Xlim, list):
        raise TypeError('Xlim must be a list')
    if len(Xlim) != 2 and Xlim != []:
        raise ValueError('Xlim must be an empty list or a list with length = 2')

    # Initial set-up of Xr (phase boundaries)
    dis, X = profile.dis, profile.X
    Xlim = [X[0], X[-1]] if Xlim == [] else Xlim
    DC = SF(profile, time, Xlim)
    Xr = np.array(Xlim, dtype=float)
    for i in range(len(dis)-1):
        if dis[i] == dis[i+1]:
            Xr = np.insert(Xr, -1, [X[i], X[i+1]])
    Np = len(Xr)//2
    Xr = Xr.reshape(Np, 2)
    fD = [0]*Np

    ita_start()

    # Choose Spline or UnivariateSpline
    plt.figure()
    plt.semilogy(X, DC, 'b.')
    ipt = ask_input('Use Spline (y) or UnivariateSpline (n) to model diffusion coefficients? [y]\n')
    choice = False if 'N' in ipt or 'n' in ipt else True

    # Xspl provided, no need for manually picking Xspl
    if Xspl is not None:
        if len(Xspl) != Np:
            raise ValueError('Xspl must has a length of phase number')

        for i in range(Np):
            pid = np.where((X >= Xr[i, 0]) & (X <= Xr[i, 1]))[0]

            # Spline
            if choice:
                Dp = Dpcalc(X, DC, Xspl[i])
                fD[i] = Dfunc_spl(Xspl[i], Dp)

            # UnivariateSpline
            else:
                fD[i] = Dfunc_uspl(X, DC, Xspl[i], Xr[i])

        print('DC modeling finished, Xspl info:')
        print(Xspl)

        plt.cla()
        plt.title('DC Modeling Result')
        plt.semilogy(X, DC, 'b.')
        for i in range(Np):
            Xf = np.linspace(Xr[i, 0], Xr[i, 1], 30)
            plt.semilogy(Xf, np.exp(splev(Xf, fD[i])), 'r-')
        plt.pause(1.0)
        plt.show()

        ita_finish()

        return DiffSystem(Xr, Dfunc=fD, Xspl=Xspl)

    Xspl = [0] * Np if choice else None

    for i in range(Np):
        pid = np.where((X >= Xr[i, 0]) & (X <= Xr[i, 1]))[0]

        # Spline
        if choice:
            while True:
                plt.cla()
                plt.semilogy(X[pid], DC[pid], 'b.')
                plt.draw()
                msg = '# of spline points: 1 (constant), 2 (linear), >2 (spline)\n'
                ipt = ask_input(msg+'input # of spline points\n')
                plt.title('Select %i points of Spline' % int(ipt))
                plt.pause(1.0)
                Xp = np.array(plt.ginput(int(ipt)))[:, 0]
                Dp = Dpcalc(X, DC, Xp)
                fD[i] = Dfunc_spl(Xp, Dp)
                Xspl[i] = list(Xp)
                Xf = np.linspace(Xr[i, 0], Xr[i, 1], 30)
                plt.cla()
                plt.semilogy(X[pid], DC[pid], 'b.')
                plt.semilogy(Xf, np.exp(splev(Xf, fD[i])), 'r-', lw=2)
                plt.draw()
                ipt = ask_input('Continue to next phase? [y]')
                redo = False if 'N' in ipt or 'n' in ipt else True
                if redo:
                    break

        # UnivariateSpline
        else:
            while True:
                plt.cla()
                plt.semilogy(X[pid], DC[pid], 'b.')
                plt.draw()
                ipt = ask_input('input 2 boundaries for UnivariateSpline\n')
                Xp = np.array([float(x) for x in ipt.split(' ')])
                fD[i] = Dfunc_uspl(X, DC, Xp, Xr[i])
                Xf = np.linspace(Xr[i, 0], Xr[i, 1], 30)
                plt.semilogy(Xf, np.exp(splev(Xf, fD[i])), 'r-', lw=2)
                plt.draw()
                ipt = ask_input('Continue to next phase? [y]')
                redo = False if 'N' in ipt or 'n' in ipt else True
                if redo:
                    break

    print('DC modeling finished, Xspl info:')
    print(Xspl)

    plt.cla()
    plt.title('DC Modeling Result')
    plt.semilogy(X, DC, 'b.')
    for i in range(Np):
        Xf = np.linspace(Xr[i, 0], Xr[i, 1], 30)
        plt.semilogy(Xf, np.exp(splev(Xf, fD[i])), 'r-')
    plt.pause(1.0)
    plt.show()

    ita_finish()

    return DiffSystem(Xr, Dfunc=fD, Xspl=Xspl)
