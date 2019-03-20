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

The core module gives definition of main classes in pyDiffusion.
"""

import numpy as np
from scipy.interpolate import splrep


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
