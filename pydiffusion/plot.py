import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev


def profileplot(diffprofile, compare=None, data=None):
    "Plot diffusion profile"
    dis, X = diffprofile.dis, diffprofile.X
    plt.figure('Diffusion Profile')
    if data is not None:
        plt.plot(data.dis, data.X, 'go', fillstyle='none')
    plt.plot(dis, X, 'b-', lw=2)
    if compare is not None:
        plt.plot(compare.dis, compare.X, 'r-', lw=2)
    plt.xlabel('Distance (micron)')
    plt.ylabel('Mole fraction')
    plt.show()


def DCplot(diffsys, loc=None, error=None):
    "Plot diffusion coefficients"
    plt.figure('Diffusion Coefficients')
    for i in range(diffsys.Np):
        Xf = np.linspace(diffsys.Xr[i, 0], diffsys.Xr[i, 1], 100)
        Df = np.exp(splev(Xf, diffsys.Dfunc[i]))
        plt.semilogy(Xf, Df, 'b-', lw=2)
    if loc is not None and error is not None:
        assert error.shape == (len(loc), 2), 'Wrong error data'
        for i in range(diffsys.Np):
            pid = np.where((loc >= diffsys.Xr[i, 0]) & (loc <= diffsys.Xr[i, 1]))[0]
            Xf = loc[pid]
            Dfp = np.exp(splev(Xf, diffsys.Dfunc[i])) * 10**error[pid, 0]
            Dfn = np.exp(splev(Xf, diffsys.Dfunc[i])) * 10**error[pid, 1]
            plt.semilogy(Xf, Dfp, 'r-', Xf, Dfn, 'r-', lw=2)
    plt.xlabel('Mole fraction')
    plt.ylabel('Diffusion Coefficients (m2/s)')
    plt.show()
