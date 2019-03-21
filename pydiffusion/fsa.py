"""
    Copyright (c) 2018-2019 Zhangqi Chen

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

The fsa module provides tools to perform Forward Simulation Analysis (FSA).
"""
import matplotlib.pyplot as plt
from pydiffusion.core import DiffSystem
from pydiffusion.utils import error_profile, step, matanocalc, automesh
from pydiffusion.Dtools import Dadjust
from pydiffusion.plot import profileplot, DCplot, SFplot
from pydiffusion.simulation import mphSim
from pydiffusion.io import ita_start, ita_finish, ask_input


def FSA(profile_exp, profile_sm, diffsys, time, Xlim=[], n=[400, 500], w=None, name=''):
    """
    Forward Simulation Analysis
    Extract diffusion coefficients based on a diffusion profile.
    Please do not close any plot window during the FSA process.

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
    name : str, optional
        Name of the output DiffProfile and DiffSystem

    Returns
    -------
    profile_sim : DiffProfile
        Simulated diffusion profile after FSA.
    diffsys_sim : DiffSystem
        Calculated diffusion efficients by FSA.
    """
    # Create step profile on meshed grids
    dism = automesh(profile_sm, diffsys, n)
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
    plt.pause(1.0)

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
                    plt.pause(1.0)
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
            plt.pause(1.0)
            ita_finish()

    return profile_sim, diffsys_sim
