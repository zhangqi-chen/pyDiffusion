"""
The Dtools module provides tools to calculate diffusion coefficients based on
a diffusion profile, like Sauer-Freise method, Hall method and Forward
Simulation Analysis (FSA). This module also provides the construction of
DiffSystem by fitting a smooth diffusion coefficient curve based on
Sauer-Fraise calculation results and the tools to adjust it.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, UnivariateSpline
from scipy.special import erfinv
from pydiffusion.core import DiffSystem
from pydiffusion.utils import disfunc, matanocalc, error_profile, step, automesh, SF
from pydiffusion.io import ita_start, ita_finish, ask_input
from pydiffusion.plot import profileplot, DCplot, SFplot
from pydiffusion.simulation import mphSim


def SauerFreise(profile, time, Xlim=[]):
    """
    Use Sauer-Freise method to calculate and virtualize diffusion coefficients.

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
    SFplot(profile=profile, time=time, Xlim=Xlim)
    DC = SF(profile=profile, time=time, Xlim=Xlim)
    return DC


def Hall(profile, time, Xlim=[], a=0.25):
    """
    Use Hall method to estimate diffusion coefficients nearby concentration limits.

    Parameters
    ----------
    profile : DiffProfile
        Diffusion profile.
    time : float
        Diffusion time in seconds.
    Xlim : list (float), optional
        Indicates the left and right concentration limits for calculation.
        Default value = [profile.X[0], profile.X[-1]].
    a : float, optional
        Potion of solubility range will be plotted to choose the linear fitting range
        of u vs. lambda. default a =0.25, 0 < a < 1.
        e.g. XL = 0, XR = 1, a = 0.25. [0, 0.25] and [0.75, 1] are the plotting range.

    Returns
    -------
    DC_left, DC_right : numpy.array
        Diffusion coefficients by Hall method at left end and right end.
        Note: Only left part of DC_left and right part of DC_right are valid.

    """
    dis, X = profile.dis, profile.X
    [XL, XR] = [X[0], X[-1]] if Xlim == [] else Xlim

    # Select range for plot
    X1, X2 = XL*(1-a)+XR*a, XL*a+XR*(1-a)
    id1, id2 = (np.abs(X-X1)).argmin(), (np.abs(X-X2)).argmin()

    # Calculate lambda & u
    Y = (X-XL)/(XR-XL)
    matano = matanocalc(profile, Xlim)
    lbd = (dis-matano)/np.sqrt(time)/1e6
    u = erfinv(2*Y-1)

    plt.figure('Hall')
    plt.cla()
    plt.title('LEFT side, select 2 points for linear fitting.')
    plt.plot(lbd[:id1], u[:id1], 'b.')
    plt.xlabel('$\lambda$')
    plt.ylabel('u')
    plt.pause(0.1)
    lbd1 = np.array(plt.ginput(2))[:, 0]

    plt.cla()
    plt.title('RIGHT side, select 2 points for linear fitting.')
    plt.plot(lbd[id2:], u[id2:], 'b.')
    plt.xlabel('$\lambda$')
    plt.ylabel('u')
    plt.pause(0.1)
    lbd2 = np.array(plt.ginput(2))[:, 0]

    sp = np.where((lbd < max(lbd1)) & (lbd > min(lbd1)))[0]
    h1, k1 = np.polyfit(lbd[sp], u[sp], 1)
    sp = np.where((lbd < max(lbd2)) & (lbd > min(lbd2)))[0]
    h2, k2 = np.polyfit(lbd[sp], u[sp], 1)

    DC_left = 1/4/h1**2*(1+2*k1/np.sqrt(np.pi)*np.exp(u**2)*Y)
    DC_right = 1/4/h2**2*(1-2*k2/np.sqrt(np.pi)*np.exp(u**2)*(1-Y))
    DC = SF(profile, time, Xlim)

    plt.cla()
    plt.semilogy(X, DC, 'b.')
    plt.semilogy(X[:id1], DC_left[:id1], 'r--', lw=2)
    plt.semilogy(X[id2:], DC_right[id2:], 'r--', lw=2)
    plt.xlabel('Mole fraction', fontsize=15)
    plt.ylabel('Diffusion Coefficients '+'$(m^2/s)$', fontsize=15)
    plt.xlim(X.min(), X.max())

    return DC_left, DC_right


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
        Point Mode (True) or Phase Mode (False). Point Mode
        adjusts each Dp at Xp by itself. In Phase Mode, all Dp are
        adjusted by the same rate, i.e. the diffusivity curve shape won't
        change.
    deltaD: float, optional
        Only useful at Phase Mode. deltaD gives the rate to change
        diffusion coefficients DC. DC = DC * 10^deltaD
    r : float, optional
        Only useful at Phase Mode, default = 0.02, 0 < r < 1. r gives the
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

    # Point Mode adjustment
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


def Dmodel(profile, time, Xspl=None, Xlim=[], output=True, name=''):
    """
    Given the diffusion profile and diffusion time, modeling the diffusion
    coefficients for each phase. Please do not close any plot window during
    the modeling process.

    Dmodel() is the second step of the forward simulation analysis (FSA).

    Parameters
    ----------
    profile : DiffProfile
        Diffusion profile. Multiple phase profile must be after datasmooth to
        identify phase boundaries.
    time : float
        Diffusion time in seconds.
    Xspl : list, optional
        If Xspl is given, Dmodel will be done automatically.
    Xlim : list, optional
        Left and Right limit of diffusion coefficients. Xlim is also passed to
        SF function to calculate diffusion coefficients initially.
    output : bool, optional
        Plot Dmodel result or not. Can be False only if Xspl is given.
    name : str, optional
        Name of the output DiffSystem

    Examples
    --------
    After datasmooth(), a initial diffusion coefficients will be established before
    FSA():

    >>> ds = datasmooth(exp)
    >>> dsys = Dmodel(ds, time)

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
    Xr.sort()
    Xr = Xr.reshape(Np, 2)
    fD = [0]*Np

    ita_start()

    # Choose Spline or UnivariateSpline
    if Xspl is None or output:
        ax = plt.figure().gca()
        ax.semilogy(X, DC, 'b.')
        ax.set_title('Sauer-Freise result')
        ax.set_xlabel('Mole fraction')
        ax.set_ylabel('Diffusion Coefficients '+'$(m^2/s)$')
        plt.tight_layout()

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
                try:
                    Dp = Dpcalc(X, DC, Xspl[i])
                    fD[i] = Dfunc_spl(Xspl[i], Dp)
                except (ValueError, TypeError) as error:
                    ita_finish()
                    raise error

            # UnivariateSpline
            else:
                try:
                    fD[i] = Dfunc_uspl(X, DC, Xspl[i], Xr[i])
                except (ValueError, TypeError) as error:
                    ita_finish()
                    raise error

        print('DC modeling finished, Xspl info:')
        print(Xspl)

        if output:
            ax.cla()
            ax.set_title('DC Modeling Result')
            ax.semilogy(X, DC, 'b.')
            for i in range(Np):
                Xf = np.linspace(Xr[i, 0], Xr[i, 1], 30)
                ax.semilogy(Xf, np.exp(splev(Xf, fD[i])), 'r-')
            ax.set_xlabel('Mole fraction')
            ax.set_ylabel('Diffusion Coefficients '+'$(m^2/s)$')
            plt.tight_layout()
            plt.pause(0.1)
            plt.show()

        ita_finish()

        return DiffSystem(Xr, Dfunc=fD, Xspl=Xspl)

    Xspl = [0] * Np if choice else None

    for i in range(Np):
        pid = np.where((X >= Xr[i, 0]) & (X <= Xr[i, 1]))[0]

        # Spline
        if choice:
            while True:
                DC_real = [k for k in DC[pid] if not np.isnan(k) and not np.isinf(k)]
                DCmean = np.mean(DC_real)
                for k in pid:
                    if np.isnan(DC[k]) or np.isinf(DC[k]) or abs(np.log10(DC[k]/DCmean)) > 5:
                        DC[k] = DCmean
                ax.cla()
                ax.semilogy(X[pid], DC[pid], 'b.')
                ax.set_xlabel('Mole fraction')
                ax.set_ylabel('Diffusion Coefficients '+'$(m^2/s)$')
                ax.set_title('Phase %i' % (i+1))
                plt.draw()
                msg = '# of spline points: 1 (constant), 2 (linear), >2 (spline)\n'
                ipt = ask_input(msg+'input # of spline points\n')
                ax.set_title('Phase %i: Select %i points of Spline' % (i+1, int(ipt)))
                plt.pause(0.1)
                Xp = np.array(plt.ginput(int(ipt)))[:, 0]
                try:
                    Dp = Dpcalc(X, DC, Xp)
                    fD[i] = Dfunc_spl(Xp, Dp)
                except (ValueError, TypeError) as error:
                    ita_finish()
                    raise error
                Xspl[i] = list(Xp)
                Xf = np.linspace(Xr[i, 0], Xr[i, 1], 30)
                ax.semilogy(Xf, np.exp(splev(Xf, fD[i])), 'r-', lw=2)
                plt.draw()
                ipt = ask_input('Continue to next phase? [y]')
                redo = False if 'N' in ipt or 'n' in ipt else True
                if redo:
                    break

        # UnivariateSpline
        else:
            while True:
                ax.cla()
                ax.semilogy(X[pid], DC[pid], 'b.')
                ax.set_xlabel('Mole fraction')
                ax.set_ylabel('Diffusion Coefficients '+'$(m^2/s)$')
                #ax.set_title('Phase %i' % (i+1))
                ax.set_title('Phase %i: Select 2 boundaries for UnivariateSpline' % (i+1))
                plt.draw()
                plt.pause(0.1)
                Xp = np.array(plt.ginput(2))[:, 0]
                #ipt = ask_input('input 2 boundaries for UnivariateSpline\n')
                #Xp = np.array([float(x) for x in ipt.split()])
                try:
                    fD[i] = Dfunc_uspl(X, DC, Xp, Xr[i])
                except (ValueError, TypeError) as error:
                    ita_finish()
                    raise error
                Xf = np.linspace(Xr[i, 0], Xr[i, 1], 30)
                ax.semilogy(Xf, np.exp(splev(Xf, fD[i])), 'r-', lw=2)
                plt.draw()
                ipt = ask_input('Continue to next phase? [y]')
                redo = False if 'N' in ipt or 'n' in ipt else True
                if redo:
                    break

    ita_finish()

    print('DC modeling finished, Xspl info:')
    print(Xspl)

    ax.cla()
    ax.set_title('DC Modeling Result')
    ax.semilogy(X, DC, 'b.')
    for i in range(Np):
        Xf = np.linspace(Xr[i, 0], Xr[i, 1], 30)
        ax.semilogy(Xf, np.exp(splev(Xf, fD[i])), 'r-')
    ax.set_xlabel('Mole fraction')
    ax.set_ylabel('Diffusion Coefficients '+'$(m^2/s)$')
    plt.tight_layout()
    plt.pause(0.1)
    plt.show()

    if name == '':
        name = profile.name+'_%.1fh_Dmodeled' % (time/3600)

    return DiffSystem(Xr, Dfunc=fD, Xspl=Xspl, name=name)


def FSA(profile_exp, profile_sm, diffsys, time, Xlim=[], n=[400, 500], w=None, f=None, alpha=0.3, name=''):
    """
    Forward Simulation Analysis
    Extract diffusion coefficients based on a diffusion profile.
    Please do not close any plot window during the FSA process.

    This is the final step of FSA.

    Parameters
    ----------
    profile_exp : DiffProfile
        Experimental diffusion profile, used for comparison with simulation
        results.
    profile_sm : DiffProfile
        Diffusion profile after data smooth on experimental profile.
    diffsys : DiffSystem
        Diffusion coefficients
    time : float
        Diffusion time in seconds
    Xlim : list (float), optional
        Passed to 'pydiffusion.Dtools.SF', 'pydiffusion.utils.step'.
        Indicates the left and right concentration limits for calculation.
        Default value = [profile.X[0], profile.X[-1]].
    n : list. optional
        Passed to 'pydiffusion.utils.automesh'.
        Meshing number range, default = [400, 500].
    w : list, optional
        Weights of each phase to calculate error.
        Passed to 'pydiffusion.utils.error_profile'.
    f : function of Meshing
        Keyword argument of automesh()
    alpha : float
        Keyword argument of automesh()
    name : str, optional
        Name the output DiffProfile

    Returns
    -------
    profile_sim : DiffProfile
        Simulated diffusion profile after FSA.
    diffsys_sim : DiffSystem
        Calculated diffusion efficients by FSA.

    Examples
    --------
    After datasmooth() and Dmodel(), FSA can be performed to calculate accurate diffusion coefficients:

    >>> ds = datasmooth(exp)
    >>> dsys = Dmodel(ds, time)
    >>> fsa = FSA(exp, ds, dsys, time)

    """
    # Create step profile on meshed grids
    dism = automesh(profile=profile_sm, diffsys=diffsys, n=n, f=f, alpha=alpha)
    matano = matanocalc(profile_sm, Xlim)
    if Xlim == [] and profile_sm.X[-1] < profile_sm.X[0]:
        profile_init = step(dism, matano, diffsys, [diffsys.Xr[-1, 1], diffsys.Xr[0, 0]])
    else:
        profile_init = step(dism, matano, diffsys, Xlim)

    # Determine the stop criteria of forward simulations
    error_sm = error_profile(profile_sm, profile_exp)
    ipt = input('Default error = %.6f\nInput the stop criteria of error: [%.6f]\n'
                % (error_sm, error_sm*2))
    error_stop = error_sm*2 if ipt == '' else float(ipt)

    # If there is no Xspl info in diffsys, use Phase Mode
    # else: ask if use Phase or Point Mode
    if diffsys.Xspl is not None:
        ipt = input('Use Phase Mode? [n]\n(The shape of diffusivity curve does not change)\n')
        pp = False if 'y' in ipt or 'Y' in ipt else True
    else:
        pp = False

    if name == '':
        name = profile_exp.name+'_FSA'

    # Diffusion coefficients used for forward simulations
    diffsys_sim = DiffSystem(diffsys.Xr, diffsys.Dfunc, Xspl=diffsys.Xspl, name=name)

    # Plot FSA status
    fig = plt.figure('FSA', figsize=(16, 6))
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    profileplot(profile_exp, ax1, ls='none', marker='o', c='b', fillstyle='none')
    profileplot(profile_sm, ax1, ls='-', c='g', lw=1)
    SFplot(profile_sm, time, Xlim, ax2, ls='none', c='b', marker='.')
    DCplot(diffsys_sim, ax2, ls='-', c='r', lw=2)
    plt.draw()
    plt.tight_layout()
    plt.pause(0.1)

    n_sim = 0
    while True:

        # Simulation
        n_sim += 1
        profile_sim = mphSim(profile_init, diffsys_sim, time, name=name)
        error_sim = error_profile(profile_sim, profile_exp, w)
        print('Simulation %i, error = %f(%f)' % (n_sim, error_sim, error_stop))

        # Plot simulation results
        ax1.cla()
        ax2.cla()
        profileplot(profile_exp, ax1, ls='none', marker='o', c='b', fillstyle='none')
        profileplot(profile_sm, ax1, ls='-', c='g', lw=1)
        profileplot(profile_sim, ax1, ls='-', c='r', lw=2)
        SFplot(profile_sm, time, Xlim, ax2, ls='none', c='b', marker='.')
        DCplot(diffsys_sim, ax2, ls='-', c='r', lw=2)
        plt.draw()
        plt.tight_layout()

        # DC adjust
        Dfunc_adjust = [0] * diffsys_sim.Np

        # If error > stop criteria, continue simulation by auto DC adjustment
        if error_sim > error_stop:
            for ph in range(diffsys_sim.Np):
                try:
                    Dfunc_adjust[ph] = Dadjust(profile_sm, profile_sim, diffsys_sim, ph, pp)
                except (ValueError, TypeError) as error:
                    ita_finish()
                    raise error
            diffsys_sim.Dfunc = Dfunc_adjust

        # If error < stop criteria or simulate too many times
        if error_sim <= error_stop or n_sim > 9:

            ita_start()

            # Ask if exit
            ipt = ask_input('Satisfied with FSA? [n]')
            if 'y' in ipt or 'Y' in ipt:
                ita_finish()
                break

            # If use Point Mode
            if diffsys_sim.Xspl is not None:
                ipt = ask_input('Use Point Mode (y) or Phase Mode (n)? [y]')
                pp = False if 'n' in ipt or 'N' in ipt else True
                if pp:
                    for ph in range(diffsys_sim.Np):
                        try:
                            Dfunc_adjust[ph] = Dadjust(profile_sm, profile_sim, diffsys_sim, ph, pp)
                        except (ValueError, TypeError) as error:
                            ita_finish()
                            raise error
                    diffsys_sim.Dfunc = Dfunc_adjust

                    DCplot(diffsys_sim, ax2, ls='-', c='m', lw=2)
                    plt.draw()
                    plt.pause(0.1)
                    ita_finish()
                    continue

            # Phase Mode, ask if use manual input for each phase
            pp = False
            ipt = input('Phase Mode\nManually input for each phase? [n]')
            manual = True if 'y' in ipt or 'Y' in ipt else False
            for ph in range(diffsys_sim.Np):
                if manual:
                    ipt = input('Input deltaD for phase # %i:\n(DC = DC * 10^deltaD, default deltaD = auto)\n' % (ph+1))
                    deltaD = float(ipt) if ipt != '' else None
                else:
                    deltaD = None
                try:
                    Dfunc_adjust[ph] = Dadjust(profile_sm, profile_sim, diffsys_sim, ph, pp, deltaD)
                except (ValueError, TypeError) as error:
                    ita_finish()
                    raise error

            # Apply the adjustment to diffsys_sim
            diffsys_sim.Dfunc = Dfunc_adjust

            DCplot(diffsys_sim, ax2, ls='-', c='m', lw=2)
            plt.draw()
            plt.pause(0.1)
            ita_finish()

    return profile_sim, diffsys_sim
