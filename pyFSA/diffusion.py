import numpy as np
from scipy.interpolate import splrep


class DiffProfile(object):
    """
    Diffusion Profile

    Parameters
    ----------
    dis, X : numpy.array
        Diffusion profile data.
    itf : numpy.array, optional
        N+1 interfaces locations (in micron) for N phases system.
        Default value is [dis[0]-d1, dis[-1]+d2], d1 d2 are small values.
    """

    def __init__(self, dis, X, itf=[]):
        assert len(dis) == len(X), 'length of dis and X is not equal'
        self.dis = dis
        self.X = X
        if itf == []:
            self.Ip = np.array([0, len(dis)])
            d1, d2 = 0.5*(dis[1]-dis[0]), 0.5*(dis[-1]-dis[-2])
            self.If = np.array([dis[0]-d1, dis[-1]+d2])
        else:
            self.If = itf
            self.Ip = np.zeros(len(self.If), dtype=int)
            for i in range(len(self.Ip)-1):
                self.Ip[i] = np.where(self.dis > self.If[i])[0][0]
            self.Ip[-1] = len(dis)


class DiffSystem(object):
    """
    Diffusion System with diffusion coefficients modeling.

    Parameters
    ----------
    Xr : numpy.array with shape (,2), optional
        Concentration range for each phase, default = [0,1].
        Save Xr.shape[0] to phase number DiffSystem.Np.
    Dfunc : list of tck tuple, optional
        Np of tck tuple discribe the diffusion coefficient function for each phase.
    X, DC : numpy.array, optional
        Use exsiting data to model Dfunc.
    """

    def __init__(self, Xr=[0, 1], Dfunc=[], X=[], DC=[]):
        if isinstance(Xr, list):
            self.Np = 1
            self.Xr = np.array([Xr])
        else:
            self.Np = Xr.shape[0]
            self.Xr = Xr
        if Dfunc != []:
            assert len(Dfunc) == self.Np, 'Incorrect phase function'
            self.Dfunc = Dfunc
        elif X != [] and DC != []:
            assert len(X) == len(DC), 'length of X and DC are not equal'
            fD = [0]*self.Np
            for i in range(self.Np):
                pid = np.where((X >= Xr[i, 0]) & (X <= Xr[i, 1]))[0]
                fD[i] = splrep(X[pid], np.log(DC[pid]))
            self.Dfunc = fD
