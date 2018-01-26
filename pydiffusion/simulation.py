import numpy as np
from scipy.interpolate import splev, splrep
from pydiffusion import DiffProfile


def sphSim(profile, diffsys, time):
    """
    Single-Phase Diffusion Simulation

    Parameters
    ----------
    profile : pydiffusion.diffusion.DiffProfile
        Initial diffusion profile before simulation.
    diffsys : pydiffusion.diffusion.DiffSystem
        Diffusion coefficients.
    time : float
        time in seconds.

    Returns
    -------
    profile : pydiffusion.diffusion.DiffProfile
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
        if np.mod(m, 3e4) == 0:
            print('%.3f/%.3f hrs simulated' % (t/3600, time/3600))
    print('Simulation Complete')
    return DiffProfile(dis, Xs)


def mphSim(profile, diffsys, time):
    """
    Multiple-Phase Diffusion Simulation.

    Parameters
    ----------
    profile : pydiffusion.diffusion.DiffProfile
        Initial diffusion profile before simulation.
    diffsys : pydiffusion.diffusion.DiffSystem
        Diffusion coefficients.
    time : float
        time in seconds.

    Returns
    -------
    profile : pydiffusion.diffusion.DiffProfile
        Simulated diffusion profile
    """
    dis, Xs = profile.dis.copy()/1e6, profile.X.copy()
    Ip, If = profile.Ip.copy(), profile.If.copy()/1e6
    Np, Xr, fD = diffsys.Np, diffsys.Xr, diffsys.Dfunc
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
            dt = min(dt, abs(min(d[Ip[i]-2:Ip[i]+1])/vIf[i]))
            if i >= 2 and vIf[i-1] > vIf[i]:
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
        Xs[0] -= Jf[0]/d[0]*dt*2
        Xs[-1] += Jf[-1]/d[-1]*dt*2

        # Interface motion
        If += vIf*dt

        # Interface move across one grid, passed grid has value change
        for i in range(1, Np):
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

        if np.mod(m, 3e4) == 0:
            print('%.3f/%.3f hrs simulated' % (t/3600, time/3600))
    print('Simulation Complete')

    # Insert interface informations
    for i in list(range(Np-1, 0, -1)):
        dis = np.insert(dis, Ip[i], [If[i], If[i]])
        Xs = np.insert(Xs, Ip[i], [Xr[i-1, 1], Xr[i, 0]])

    return DiffProfile(dis, Xs, If[1:-1])
