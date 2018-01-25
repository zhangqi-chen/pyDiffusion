import numpy as np
import pandas as pd
from scipy.interpolate import splev
from pyFSA.diffusion import DiffProfile, DiffSystem


def save_csv(name, profile=None, diffsys=None):
    "save data as csv"
    if profile is None and diffsys is None:
        raise ValueError('No data entered')
    elif profile is None:
        Xr, fD = diffsys.Xr, diffsys.Dfunc
        X, DC = np.array([]), np.array([])
        for i in diffsys.Np:
            Xnew = np.linspace(Xr[i, 0], Xr[i, 1], 30)
            Dnew = np.exp(splev(Xnew, fD[i]))
            X = np.append(X, Xnew)
            DC = np.append(DC, Dnew)
        data = pd.DataFrame({'X': X, 'DC': DC})
        data.to_csv(name, index=False)
    elif diffsys is None:
        data = pd.DataFrame({'dis': profile.dis, 'X': profile.X})
        data.to_csv(name, index=False)
    else:
        dis, X, Xr = profile.dis, profile.X, diffsys.Xr
        DC = np.zeros(len(dis))
        for i in range(diffsys.Np):
            pid = np.where((X >= Xr[i, 0]) & (X <= Xr[i, 1]))[0]
            DC[pid] = np.exp(splev(X[pid], fD[i]))
        data = pd.DataFrame({'dis': dis, 'X': X, 'DC': DC})
        data.to_csv(name, index=False)


def read_csv(name, Xr=None):
    "read profile and DC data from csv"
    data = pd.read_csv(name)
    if 'X' not in data.columns:
        raise ValueError('No column X in csv file')
    X = np.array(data['X'])
    if 'dis' in data.columns:
        dis = np.array(data['dis'])
        If = []
        XIf = []
        for i in range(len(dis)-1):
            if dis[i] == dis[i+1]:
                If += [dis[i]]
                XIf += [X[i], X[i+1]]
        profile = DiffProfile(dis, X, If)
        XIf = np.array([X[0]] + XIf + [X[-1]])
        Xr = XIf.reshape((len(XIf)//2, 2))
    if 'DC' in data.columns:
        DC = np.array(data['DC'])
    else:
        X = DC = None
    diffsys = DiffSystem(Xr, X=X, DC=DC)
    return profile, diffsys
