"""
The simulation module contains diffusion simulation function and
simulation-based error analysis tools.
"""

import numpy as np
from scipy.interpolate import splev, splrep
from pydiffusion.core import DiffProfile, DiffError
from pydiffusion.utils import error_profile, DCbias


def sphSim(profile, diffsys, time, output=True):
    """
    Single-Phase Diffusion Simulation

    Parameters
    ----------
    profile : DiffProfile
        Initial diffusion profile before simulation.
    diffsys : DiffSystem
        Diffusion coefficients.
    time : float
        time in seconds.
    output : boolean, optional
        Print simulation progress, default = True.

    Returns
    -------
    profile : DiffProfile
        Simulated diffusion profile
    """
    dis, Xs = profile.dis.copy()/1e6, profile.X.copy()
    fD = diffsys.Dfunc
    d = dis[1:]-dis[:-1]
    dj = 0.5*(d[1:]+d[:-1])
    t, m = 0.0, 0
    while t < time:
        Xm = 0.5*(Xs[1:]+Xs[:-1])
        DCs = np.exp(splev(Xm, fD[0]))
        dt = min(d**2/DCs/2)
        dt = time-t if t+dt > time else dt*0.95
        t += dt
        m += 1
        Jf = -DCs*(Xs[1:]-Xs[:-1])/d
        Xs[1:-1] = Xs[1:-1]-dt*(Jf[1:]-Jf[:-1])/dj
        Xs[0] -= Jf[0]/d[0]*dt*2
        Xs[-1] += Jf[-1]/d[-1]*dt*2
        if output and np.mod(m, 3e4) == 0:
            print('%.3f/%.3f hrs simulated' % (t/3600, time/3600))
    if output:
        print('Simulation Complete')
    return DiffProfile(dis*1e6, Xs)


def mphSim(profile, diffsys, time, liquid=0, output=True):
    """
    Single/Multiple-Phase Diffusion Simulation. Liquid phase can be attached
    at left or right end. For thin film simulation, please set up the
    interface nearby the end in the initial profile.

    Parameters
    ----------
    profile : DiffProfile
        Initial profile before simulation.
    diffsys : DiffSystem
        Diffusion coefficients.
    time : float
        time in seconds.
    liquid : 1, 0 or -1, optional
        Liquid phase provide infinite mass flux during simulation, can only be
        attached at left or right end.
        0 : No liquid phase attached.
        -1 : Liquid phase attached at left.
        1 : Liquid phase attached at right.
    output : boolean, optional
        Print simulation progress, default = True.

    Returns
    -------
    profile : DiffProfile
        Simulated diffusion profile
    """
    dis, Xs = profile.dis.copy()/1e6, profile.X.copy()
    Ip, If = profile.Ip.copy(), profile.If.copy()/1e6
    Np, Xr = diffsys.Np, diffsys.Xr.copy()
    fD = [f for f in diffsys.Dfunc]

    if len(If) != Np+1:
        raise ValueError('Number of phases mismatches between profile and diffusion system')
    if liquid not in [-1, 0, 1]:
        raise ValueError('liquid can only be 0 1 or -1')
    try:
        time = float(time)
    except TypeError:
        print('Wrong type for time variable')

    d = dis[1:]-dis[:-1]
    dj = 0.5*(d[1:]+d[:-1])
    Jf, DCs = np.zeros(len(dis)), np.zeros(len(dis))
    JIf = np.zeros([Np+1, 2])
    vIf = np.zeros(Np+1)
    t, m = 0.0, 0
    dt0 = time/1e3

    while t < time:

        Xm = (Xs[:-1]+Xs[1:])/2
        dt = dt0

        # Jf JIf calculation
        # dt limited by simulation stability inside each phases
        for i in range(Np):
            if Ip[i+1] > Ip[i]+1:
                Ipr = np.arange(Ip[i], Ip[i+1]-1)
                DCs[Ipr] = np.exp(splev(Xm[Ipr], fD[i]))
                Jf[Ipr] = -DCs[Ipr]*(Xs[Ipr+1]-Xs[Ipr])/d[Ipr]
                dt = min(dt, min(abs(d[Ipr]**2/DCs[Ipr]/2)))
            if Ip[i+1] == Ip[i]:
                X0 = np.mean(Xr[i])
                JIf[i, 1] = JIf[i+1, 0] = -(Xr[i, 1]-Xr[i, 0])/(If[i+1]-If[i])*np.exp(splev(X0, fD[i]))
            else:
                if i < Np-1:
                    X0 = 0.5*(Xr[i, 1]+Xs[Ip[i+1]-1])
                    JIf[i+1, 0] = -(Xr[i, 1]-Xs[Ip[i+1]-1])/(If[i+1]-dis[Ip[i+1]-1])*np.exp(splev(X0, fD[i]))
                if i > 0:
                    X0 = 0.5*(Xr[i, 0]+Xs[Ip[i]])
                    JIf[i, 1] = -(Xs[Ip[i]]-Xr[i, 0])/(dis[Ip[i]]-If[i])*np.exp(splev(X0, fD[i]))

        # vIf calculation, dt limited by 'No interface passing by'
        for i in range(1, Np):
            vIf[i] = (JIf[i, 1]-JIf[i, 0])/(Xr[i, 0]-Xr[i-1, 1])
            vid = [Ip[i]-2] if Ip[i] > 1 else []
            vid += [Ip[i]] if Ip[i] < len(dis) else []
            dt = min(dt, abs(min(d[vid]/vIf[i])))
            if i > 1 and vIf[i-1] > vIf[i]:
                dt = min(dt, (If[i]-If[i-1])/(vIf[i-1]-vIf[i])/2)

        # dt limited by grid nearby interfaces cannot exceed solubility limit
        for i in range(Np):
            if Ip[i+1] == Ip[i]:
                continue
            elif Ip[i+1] == Ip[i]+1:
                if JIf[i, 1] > JIf[i+1, 0]:
                    dt = min(dt, (Xr[i, 1]-Xs[Ip[i]])*dj[Ip[i]-1]/(JIf[i, 1]-JIf[i+1, 0]))
                elif JIf[i, 1] < JIf[i+1, 0]:
                    dt = min(dt, (Xs[Ip[i]]-Xr[i, 0])*dj[Ip[i]-1]/(JIf[i+1, 0]-JIf[i, 1]))
            else:
                if i < Np-1 and JIf[i+1, 0] < Jf[Ip[i+1]-2]:
                    dt = min(dt, -(Xr[i, 1]-Xs[Ip[i+1]-1])/(JIf[i+1, 0]-Jf[Ip[i+1]-2])*dj[Ip[i+1]-2])
                if i > 0 and Jf[Ip[i]] > JIf[i, 1]:
                    dt = min(dt, (Xs[Ip[i]]-Xr[i, 0])/(Jf[Ip[i]]-JIf[i, 1])*dj[Ip[i]-1])

        dt = time-t if t+dt > time else dt*0.95

        # If first or last phase will be consumed
        if If[1] < dis[1] and vIf[1] < 0:
            dt = min(dt, (dis[0]-If[1])/vIf[1])
        elif If[-2] > dis[-2] and vIf[-2] > 0:
            dt = min(dt, (dis[-1]-If[-2])/vIf[-2])

        t += dt
        m += 1

        # Ficks 2nd law inside each phase
        for i in range(Np):
            if Ip[i+1] == Ip[i]:
                continue
            elif Ip[i+1] == Ip[i]+1:
                Xs[Ip[i]] -= dt*(JIf[i+1, 0]-JIf[i, 1])/dj[Ip[i]-1]
            else:
                if i > 0:
                    Xs[Ip[i]] -= dt*(Jf[Ip[i]]-JIf[i, 1])/dj[Ip[i]-1]
                if i < Np-1:
                    Xs[Ip[i+1]-1] -= dt*(JIf[i+1, 0]-Jf[Ip[i+1]-2])/dj[Ip[i+1]-2]
                if Ip[i+1] > Ip[i]+2:
                    Ipr = np.arange(Ip[i]+1, Ip[i+1]-1)
                    Xs[Ipr] -= dt*(Jf[Ipr]-Jf[Ipr-1])/dj[Ipr-1]

        # Composition changes at first & last grid
        # If there is liquid phase attached, composition unchanged.
        if liquid != -1:
            Xs[0] -= Jf[0]/d[0]*dt*2
        if liquid != 1:
            Xs[-1] += Jf[-1]/d[-1]*dt*2

        # If one phase comsumed, delete this phase
        if If[1]+vIf[1]*dt <= dis[0]:
            Xs[0] = Xr[1, 0]
            Np -= 1
            Xr, If, Ip, fD = Xr[1:], If[1:], Ip[1:], fD[1:]
            Ip[0] = 0
            JIf = np.zeros([Np+1, 2])
            vIf = np.zeros(Np+1)
            if output:
                print('First phase consumed, %i phase(s) left, time = %.3f hrs' % (Np, t/3600))
        elif If[-2]*vIf[-2]*dt >= dis[-1]:
            Xs[-1] = Xr[-2, 1]
            Np -= 1
            Xr, If, Ip, fD = Xr[:-1], If[:-1], Ip[:-1], fD[:-1]
            Ip[-1] = len(dis)
            JIf = np.zeros([Np+1, 2])
            vIf = np.zeros(Np+1)
            if output:
                print('Last phase consumed, %i phase(s) left, time = %.3f hrs' % (Np, t/3600))

        # Interface move across one grid, passed grid has value change
        for i in range(1, Np):
            If[i] += vIf[i]*dt
            if If[i] < dis[Ip[i]-1]:
                Ip[i] -= 1
                if If[i+1] < dis[Ip[i]+1]:
                    Xs[Ip[i]] = splev(dis[Ip[i]],
                                      splrep([If[i], If[i+1]], Xr[i], k=1))
                else:
                    Xs[Ip[i]] = splev(dis[Ip[i]],
                                      splrep([If[i], dis[Ip[i]+1]],
                                             [Xr[i, 0], Xs[Ip[i]+1]], k=1)
                                      )
            elif If[i] > dis[Ip[i]]:
                Ip[i] += 1
                if If[i-1] > dis[Ip[i]-2]:
                    Xs[Ip[i]-1] = splev(dis[Ip[i]-1],
                                        splrep([If[i-1], If[i]], Xr[i-1], k=1))
                else:
                    Xs[Ip[i]-1] = splev(dis[Ip[i]-1],
                                        splrep([dis[Ip[i]-2], If[i]],
                                               [Xs[Ip[i]-2], Xr[i-1, 1]], k=1)
                                        )

        if output and np.mod(m, 3e4) == 0:
            print('%.3f/%.3f hrs simulated' % (t/3600, time/3600))
    if output:
        print('Simulation Complete')

    # Insert interface informations
    for i in list(range(Np-1, 0, -1)):
        dis = np.insert(dis, Ip[i], [If[i], If[i]])
        Xs = np.insert(Xs, Ip[i], [Xr[i-1, 1], Xr[i, 0]])

    return DiffProfile(dis*1e6, Xs, If[1:-1]*1e6)


def ErrorAnalysis(profile_exp, profile_init, diffsys, time, loc=10,
                  accuracy=1e-3, output=True):
    """
    Error analysis of diffusion coefficients through comparison with experimental
    data.

    Parameters
    ----------
    profile_exp : DiffProfile
        Experiment measured diffusion profile, which is used to compared against
        each simulation result.
    profile_init : DiffProfile
        The initial profile for diffusion simulation, usually a step profile.
    diffsys : DiffSystem
        Reference diffusion coefficients. Simulation based on this datasets will
        be the reference for the error analysis.
        Bias will then be applied to this diffusivity datasets before each
        simulation.
    time : float
        Diffusion time in seconds.
    loc : list or int
        loc indicates the locations to perform error analysis. If loc is an
        integer, loc points are selected inside each phase. Each point has
        both positive and negative error to be calculated.
    accuracy : float
        Stop criterion of each simulation: within error_cap * (1+-accuracy).
        Low accuracy value may increase simulation times for each point.
    output : boolean, optional
        Print analysis progress, default = True.

    Returns
    -------
    differror : DiffError
        Diffusion error object
    """
    if len(profile_init.If) != diffsys.Np+1:
        raise ValueError('Number of phases mismatches between profile and diffusion system')
    try:
        time = float(time)
    except TypeError:
        print('Wrong type for time variable')

    profile_ref = mphSim(profile_init, diffsys, time)
    error_ref = error_profile(profile_ref, profile_exp)
    ipt = input('Reference error= % .6f. Input cap error: [% .6f]' % (error_ref, error_ref*1.01))
    error_cap = error_ref*1.01 if ipt == '' else float(ipt)
    print('Cap error = % .6f' % error_cap)

    if isinstance(loc, int):
        n = loc
        loc = np.array([])
        for i in range(diffsys.Np):
            loc = np.append(loc, np.linspace(diffsys.Xr[i, 0], diffsys.Xr[i, 1], n))

    profiles = []
    errors = np.zeros((len(loc), 2))
    deltaD = -0.5
    for i in range(len(loc)):
        X = loc[i]
        profile_at_X = []
        for p in range(2):
            n_sim = 0
            deltaD = -deltaD
            De, Xe = [0], [error_ref]
            while True:
                n_sim += 1
                diffsys_error = DCbias(diffsys, X, deltaD)
                profile_error = mphSim(profile_init, diffsys_error, time, output=False)
                error_sim = error_profile(profile_error, profile_exp)
                if output:
                    print('At %.3f, simulation #%i, deltaD = %f, profile difference = %f(%f)'
                          % (X, n_sim, deltaD, error_sim, error_cap))

                if abs(error_sim-error_cap) < error_cap*accuracy:
                    break

                if len(De) == 1:
                    if error_sim > error_cap:
                        De += [deltaD]
                        Xe += [error_sim]
                        fe = splrep(Xe, De, k=1)
                        deltaD = float(splev(error_cap, fe))
                    else:
                        De, Xe = [deltaD], [error_sim]
                        deltaD *= 2
                else:
                    j = 1 if error_sim > error_cap else 0
                    De[j], Xe[j] = deltaD, error_sim

                    if (Xe[1]-Xe[0])/abs(De[1]-De[0]) > 100:
                        deltaD = De[0]
                        if output:
                            print('Jump between %f and %f' % (De[0], De[1]))
                        break
                    elif (Xe[1]-Xe[0]) > abs(De[1]-De[0]) and n_sim > 4:
                        deltaD = np.mean(De)
                    else:
                        fe = splrep(Xe, De, k=1)
                        deltaD = float(splev(error_cap, fe))
            profile_at_X += [profile_error]
            errors[i, p] = deltaD
        profiles += [profile_at_X]
        data = {}
        data['exp'] = profile_exp
        data['ref'] = profile_ref
        data['error'] = profiles

    differror = DiffError(diffsys, loc, errors, data)

    return differror
