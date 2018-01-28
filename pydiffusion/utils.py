import numpy as np
from scipy.interpolate import splev, splrep, UnivariateSpline
from pydiffusion import DiffProfile, DiffSystem


def mesh(profile, diffsys, n=[400, 500], f=lambda X: X**0.3):
    """
    Meshing for fast simulation similiar to existing profile.

    Parameters
    ----------
    profile : pydiffusion.diffusion.DiffProfile
        The profile meshing is based on. It should be similar to final
        simulated profile. e.g. Smoothed experimental profile.
    diffsys : pydiffusion.diffusion.DiffSystem
        The diffusion coefficients for simulation.
    n : list
        meshing number range, default = [400, 500]
    f : function of meshing
        Meshing grid size is propotional to f(DC), default = DC**0.3
        DC is diffusion coefficients.

    Returns
    -------
    dism: numpy.array
        Distance information after meshing.

    """
    dis, X = profile.dis, profile.X
    Xr, fD = diffsys.Xr, diffsys.Dfunc
    nmin, nmax = n
    dmin = dis[-1]/nmax/2

    # Create profile function fX
    disn = dis.copy()
    for i in range(len(X)-1):
        if disn[i] == disn[i+1]:
            disn[i] -= (disn[i]-dis[i-1])/1e5
            disn[i+1] += (disn[i+2]-disn[i+1])/1e5
    fX = splrep(disn, X, k=1)

    # Create universel D function fDC
    Xf, Df = np.array([]), np.array([])
    for i in range(len(fD)):
        Xnew = np.linspace(Xr[i, 0], Xr[i, 1], 20)
        Xf = np.append(Xf, Xnew)
        Df = np.append(Df, np.exp(splev(Xnew, fD[i])))
    fDC = splrep(Xf, np.log(Df), k=2)

    # Meshing
    while True:
        dism = [dis[0]]
        dseed = dmin/min(f(np.exp(splev(X, fDC))))
        while dism[-1] < dis[-1]:
            disDC = np.exp(splev(splev(dism[-1], fX), fDC))
            dnew = dmin if disDC < 1e-17 else dseed*f(disDC)
            dism += [dism[-1]+dnew]
        dism += [dis[-1]+dnew]
        meshnum = len(dism)
        if meshnum < nmin:
            dmin /= 2
        elif meshnum < nmax:
            break
        else:
            dmin *= 1.1
    print('Meshing Num=%i, Minimum grid=%f um' % (meshnum, dmin))
    return np.array(dism)


def step(dis, matano, diffsys):
    """
    Create a step profile at the matano plane.

    Parameters
    ----------
    dis : numpy.array
        Distance information.
    matano : float
        Matano plane location (micron).
    diffsys : pydiffusion.diffusion.DiffSystem
        The diffusion system information for initial setups before simulation.

    Returns
    -------
    profile : pydiffusion.diffusion.DiffProfile
        Step profile can be used for input of pydiffusion.simulation.mphSim
    """
    Np = diffsys.Np
    XL, XR = diffsys.Xr[0][0], diffsys.Xr[-1][1]
    X = np.ones(len(dis))*XL
    X[dis > matano] = XR
    if Np == 1:
        return DiffProfile(dis, X)
    else:
        If = [0]*(Np-1)
        Ip = np.where(X == XR)[0][0]
        d = dis[Ip] - dis[Ip-1]
        for i in range(Np-1):
            If[i] = dis[Ip-1] + d/(Np+1)*(i+1)
        return DiffProfile(dis, X, If)


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


def profilefunc(profile):
    "Create a tck interpolation (k=1) of the diffusion profile"
    disn, Xn = profile.dis.copy(), profile.X.copy()
    for i in range(len(disn)-1):
        if disn[i] == disn[i+1]:
            disn[i] -= (disn[i]-disn[i-1])/1e5
            disn[i+1] += (disn[i+2]-disn[i+1])/1e5
    return splrep(disn, Xn, k=1)


def SF(profile, time, Xlim=[]):
    """
    Use Sauer-Fraise method to calculate diffusion coefficients from profile

    Parameters
    ----------
    profile : pydiffusion.diffusion.DiffProfile
        Diffusion profile
    time : float
        Diffusion time in seconds
    Xlim : list (float), optional
        Indicates the left and right concentration limits for calculation.
        Default value = [profile.X[0], profile.X[-1]]

    Returns
    -------
    DC : numpy.array
        Diffusion coefficients
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


def matanocalc(profile, Xlim=None):
    """
    Matano Plane calculation.

    Parameters
    ----------
    profile : pydiffusion.diffusion.DiffProfile
        Diffusion Profile.
    Xlim : list
        The left and right end concentration of the profile.

    Returns:
    --------
    matano : float
        Matano Plane location.
    """
    dis, X = profile.dis, profile.X
    if Xlim is None:
        XL, XR = X[0], X[-1]
    elif isinstance(Xlim, list) and len(Xlim) == 2:
        XL, XR = Xlim
    else:
        raise ValueError('Xlim is a list with length = 2')
    return (np.trapz(X, dis)-dis[-1]*XR+dis[0]*XL)/(XL-XR)


def error_profile(profilesim, profileexp):
    """
    Calculate the difference (in mole fraction) between experimental profile
    and simulated profile. This function take profilesim as reference, i.e. 
    compare profileexp against profilesim.

    Parameters
    ----------
    profilesim : pydiffusion.diffusion.DiffProfile
        Simulated diffusion profile.
    profileexp : pydiffusion.diffusion.DiffProfile
        Experiemntal measured profile.

    Returns
    -------
    error : float
        Averaged difference in mole fraction.
    """
    fX = profilefunc(profilesim)
    dissim, Xsim = profilesim.dis, profilesim.X
    disexp, Xexp = profileexp.dis, profileexp.X
    error = 0
    for i in range(len(disexp)):
        if disexp[i] < dissim[0]:
            error += abs(Xexp[i]-Xsim[0])
        elif disexp[i] > dissim[-1]:
            error += abs(Xexp[i]-Xsim[-1])
        else:
            error += abs(Xexp[i]-splev(disexp[i], fX))
    return error / len(disexp)


def efunc(X, Xf, r):
    "Default efunc to create bias for diffusion coefficients"
    if abs(Xf-X) <= r/2:
        deltae = 1-2*(X-Xf)**2/r**2
    elif Xf < X-r/2:
        deltae = 2*(Xf-X+r)**2/r**2
    else:
        deltae = 2*(Xf-X-r)**2/r**2
    return deltae


def DCbias(diffsys, X, deltaD, r=0.3, efunc=efunc):
    """
    This function creates bias for a diffusion coefficients profile

    Parameters
    ----------
    diffsys : pydiffusion.diffusion.DiffSystem
        Original diffusion coefficients.
    X : float
        Concentration location which has largest bias.
    deltaD : float
        Scale of the bias. D *= 10**deltaD is the maximum of bias.
    r : float, optional
        Bias at X will create smaller bias in range of (X-r, X+r)
    efunc : function
        efunc(X, Xf, r)
        efunc should return 1 as the maximum when X == Xf,
        and return 0 when abs(X-Xf) == r.

    Returns
    -------
    fDbias : pydiffusion.diffusion.DiffSystem
        Diffusion coefficients with bias.

    """
    Xr, Np, fD = diffsys.Xr, diffsys.Np, diffsys.Dfunc
    fDbias = []
    for i in range(Np):
        if X >= Xr[i, 0] and X <= Xr[i, 1]:
            Xf = np.linspace(Xr[i, 0], Xr[i, 1], 30)
            Df = np.exp(splev(Xf, fD[i]))
            eid = np.where((Xf >= X-r) & (Xf <= X+r))[0]
            for j in eid:
                Df[j] *= 10**(deltaD * efunc(X, Xf[j], r))
            fDbias += [splrep(Xf, np.log(Df))]
        else:
            fDbias += [fD[i]]
    return DiffSystem(Xr, fDbias)
