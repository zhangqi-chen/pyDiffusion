"""
The core module gives definition of main classes in pyDiffusion.
"""

import numpy as np
from scipy.interpolate import splrep, splev


class DiffProfile(object):
    """
    Diffusion Profile

    Parameters
    ----------
    dis : array-like
        Distance data in microns.
    X : array-like
        Concentration data in mole fraction (0~1).
    If : list, optional
        N-1 interfaces locations (in micron) for N phases system.
        Default value is [dis[0]-d1, dis[-1]+d2], d1 d2 are small values.
    name : str, optional
        Name of the current diffusion profile.
    """

    def __init__(self, dis, X, If=[], name='Profile'):
        try:
            self.dis = np.array(dis, dtype=float)
            self.X = np.array(X, dtype=float)
        except TypeError:
            print('Can not convert input into 1d-array')

        try:
            self.name = str(name)
        except TypeError:
            print('name must be able to convert to str type')

        if self.dis.ndim != 1 or self.X.ndim != 1:
            raise ValueError('1d data is required')

        if len(self.dis) != len(self.X):
            raise ValueError('length of dis and X is not equal')

        # dis must be ascending
        if not np.all(self.dis[1:] >= self.dis[:-1]):
            self.X = np.array([x for _, x in sorted(zip(self.dis, self.X))])
            self.dis.sort()

        d1, d2 = 0.5*(self.dis[1]-self.dis[0]), 0.5*(self.dis[-1]-self.dis[-2])

        try:
            If = np.array(If, dtype=float)
        except TypeError:
            print('If must be a list or 1d array')

        if If.ndim != 1:
            raise TypeError('If must be a list or 1d array')

        self.If = np.array([self.dis[0]-d1] + list(If) + [self.dis[-1]+d2])
        self.Ip = np.zeros(len(self.If), dtype=int)
        for i in range(1, len(self.Ip)-1):
            self.Ip[i] = np.where(self.dis > self.If[i])[0][0]
        self.Ip[-1] = len(dis)

    def copy(self, dismax=None, Xmax=None):
        """
        Method to copy DiffProfile
        Distance data or concentration data can be reversed

        Parameters
        ----------
        dismax : if given, output distance data = dismax - dis
        Xmax : if given, output concentration data = Xmax - X
        """
        if dismax is not None:
            try:
                dis = dismax-self.dis
                If = dismax-self.If
            except TypeError:
                print('dismax must be a number')
        else:
            dis = self.dis
            If = self.If
        if Xmax is not None:
            try:
                X = Xmax-self.X
            except TypeError:
                print('Xmax must be a number')
        else:
            X = self.X
        return DiffProfile(dis=dis, X=X, If=If[1:-1], name=self.name)


class DiffSystem(object):
    """
    Diffusion System with diffusion coefficients modeling

    Parameters
    ----------
    Xr : List with length of 2 or array-like with shape (,2), optional
        Concentration range for each phase, default = [0,1].
        Save Xr.shape[0] to phase number DiffSystem.Np.
    Dfunc : list of tck tuple, optional
        Np of tck tuple describe the diffusion coefficient function for each phase.
    X, DC : array-like, optional
        Use existing data to model Dfunc.
    Xspl : list of array, optional
        Xspl is the reference locations to create Dfunc, useful for forward
        simulation analysis.
    name : str, optional
        Name of the current diffusion system.
    """

    def __init__(self, Xr=[0, 1], Dfunc=None, X=None, DC=None, Xspl=None, name='System'):
        try:
            self.Xr = np.array(Xr, dtype=float)
        except TypeError:
            print('Cannot convert Xr to array')
        if self.Xr.shape == (2,):
            self.Xr = np.array([Xr])
            self.Np = 1
        elif self.Xr.shape[1] == 2:
            self.Np = self.Xr.shape[0]
        else:
            raise ValueError('Xr must has a shape (,2) or (2,)')

        try:
            self.name = str(name)
        except TypeError:
            print('name must be able to convert to str type')

        if Dfunc is not None:
            if len(Dfunc) != self.Np:
                raise ValueError('Length of Dfunc must be equal to phase number Np')
            else:
                self.Dfunc = Dfunc
        elif X is not None and DC is not None:
            try:
                X = np.array(X, dtype=float)
                DC = np.array(DC, dtype=float)
            except TypeError:
                print('Can not convert input into 1d-array')
            if X.ndim != 1 or DC.ndim != 1:
                raise ValueError('1d data is required')
            if len(X) != len(DC):
                raise ValueError('length of X and DC is not equal')
            fD = [0]*self.Np
            for i in range(self.Np):
                pid = np.where((X >= self.Xr[i, 0]) & (X <= self.Xr[i, 1]))[0]
                if len(pid) > 2:
                    fD[i] = splrep(X[pid], np.log(DC[pid]), k=2)
                else:
                    fD[i] = splrep(X[pid], np.log(DC[pid]), k=1)
            self.Dfunc = fD
        if Xspl is not None:
            if len(Xspl) != self.Np:
                raise ValueError('Length of Xspl must be equal to phase number Np')
        self.Xspl = Xspl

    def copy(self, Xmax=None):
        """
        Method to copy DiffSystem
        Concentration can be reversed

        Parameters
        ----------
        Xmax : if given, output concentration = Xmax - X
        """
        if Xmax is None:
            return DiffSystem(Xr=self.Xr, Dfunc=self.Dfunc, Xspl=self.Xspl, name=self.name)
        else:
            Xr = Xmax-self.Xr.flatten()[::-1]
            Xr = Xr.reshape((self.Np, 2))
            fD = [0]*self.Np
            for i in range(self.Np):
                k = self.Dfunc[-i-1][2]
                X = np.linspace(Xr[i, 0], Xr[i, 1], 30)
                DC = splev(Xmax-X, self.Dfunc[-i-1])
                fD[i] = splrep(X, DC, k=k)
            Xspl = None if self.Xspl is None else Xmax-self.Xspl[::-1]
            return DiffSystem(Xr=Xr, Dfunc=fD, Xspl=Xspl, name=self.name)


class DiffError(object):
    """
    Error analysis result of diffusion system

    Parameters
    ----------
    diffsys : DiffSystem
        Diffusion system object
    loc : array_like
        Locations performed error analysis
    errors : 2d-array
        Error calculated at loc.
    data : dict
        A dictionary contains all profiles data during error analysis.
        data['exp'] : Experimental collected data.
        data['ref'] : Reference simulated profile.
        data['error'] : Profiles that reaches error_cap during analysis.
    """

    def __init__(self, diffsys, loc, errors, profiles=None):
        self.diffsys = diffsys
        self.loc = np.array(loc)
        self.errors = errors
        if profiles is not None:
            self.profiles = profiles
