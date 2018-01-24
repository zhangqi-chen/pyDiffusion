import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev


def profileplot(diffprofile):
    "Plot diffusion profile"
    plt.figure('DiffProfile')
    plt.plot(diffprofile.dis, diffprofile.X, 'b-', lw=2)
    for If in diffprofile.If:
        Ifid = np.where(diffprofile.dis == If)[0]
        plt.plot(diffprofile.dis[Ifid], diffprofile.X[Ifid], 'r-', lw=2)
    plt.xlabel('Distance (micron)')
    plt.ylabel('Mole fraction')
    plt.show()


def DCplot(diffsys):
    "Plot diffusion coefficients"
    plt.figure('DiffSystem')
    for i in range(diffsys.Np):
        Xf = np.linspace(diffsys.Xr[i, 0], diffsys.Xr[i, 1], 100)
        Df = np.exp(splev(Xf, diffsys.Dfunc[i]))
        plt.semilogy(Xf, Df, 'b-', lw=2)
    plt.xlabel('Mole fraction')
    plt.ylabel('Diffusion Coefficients (m2/s)')
    plt.show()
