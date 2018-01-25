import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev


def profileplot(diffprofile):
    "Plot diffusion profile"
    dis, X = diffprofile.dis, diffprofile.X
    plt.figure('Diffusion Profile')
    plt.plot(dis, X, 'b-', lw=2)
    plt.xlabel('Distance (micron)')
    plt.ylabel('Mole fraction')
    plt.show()


def DCplot(diffsys):
    "Plot diffusion coefficients"
    plt.figure('Diffusion Coefficients')
    for i in range(diffsys.Np):
        Xf = np.linspace(diffsys.Xr[i, 0], diffsys.Xr[i, 1], 100)
        Df = np.exp(splev(Xf, diffsys.Dfunc[i]))
        plt.semilogy(Xf, Df, 'b-', lw=2)
    plt.xlabel('Mole fraction')
    plt.ylabel('Diffusion Coefficients (m2/s)')
    plt.show()
