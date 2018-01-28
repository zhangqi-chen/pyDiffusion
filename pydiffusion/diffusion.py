import numpy as np
from scipy.interpolate import splrep


class DiffProfile(object):
    """
    Diffusion Profile

    Parameters
    ----------
    dis : numpy.array
        Distance data in microns.
    X : numpy.array
        Concentration data in mole fraction (0~1).
    If : list, optional
        N-1 interfaces locations (in micron) for N phases system.
        Default value is [dis[0]-d1, dis[-1]+d2], d1 d2 are small values.
    """

    def __init__(self, dis, X, If=[]):
        assert len(dis) == len(X), 'length of dis and X is not equal'
        self.dis = np.array(dis)
        self.X = np.array(X)
        d1, d2 = 0.5*(self.dis[1]-self.dis[0]), 0.5*(self.dis[-1]-self.dis[-2])
        self.If = np.array([self.dis[0]-d1] + list(If) + [self.dis[-1]+d2])
        self.Ip = np.zeros(len(self.If), dtype=int)
        for i in range(1, len(self.Ip)-1):
            self.Ip[i] = np.where(self.dis > self.If[i])[0][0]
        self.Ip[-1] = len(dis)


class DiffSystem(object):
    """
    Diffusion System with diffusion coefficients modeling

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

    def __init__(self, Xr=[0, 1], Dfunc=None, X=None, DC=None):
        if isinstance(Xr, list):
            self.Np = 1
            self.Xr = np.array([Xr])
        else:
            self.Np = Xr.shape[0]
            self.Xr = Xr
        if Dfunc is not None:
            assert len(Dfunc) == self.Np, 'Incorrect phase function'
            self.Dfunc = Dfunc
        elif X is not None and DC is not None:
            assert len(X) == len(DC), 'length of X and DC are not equal'
            X, DC = np.array(X), np.array(DC)
            fD = [0]*self.Np
            for i in range(self.Np):
                pid = np.where((X >= self.Xr[i, 0]) & (X <= self.Xr[i, 1]))[0]
                fD[i] = splrep(X[pid], np.log(DC[pid]))
            self.Dfunc = fD
