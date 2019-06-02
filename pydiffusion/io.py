"""
The io module provides support for reading and writing diffusion profile data
and diffusion coefficients data to csv files.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import splev
from pydiffusion.core import DiffProfile, DiffSystem
import matplotlib.pyplot as plt
import threading

# To solve the problem when matplotlib figure freezes when input used
# https://stackoverflow.com/questions/34938593/matplotlib-freezes-when-input-used-in-spyder
prompt = False
promptText = ""
done = False
waiting = False
response = ""

regular_input = input


def threadfunc():
    global prompt
    global done
    global waiting
    global response

    while not done:
        if prompt:
            prompt = False
            response = regular_input(promptText)
            waiting = True


def ask_input(text):
    global waiting
    global prompt
    global promptText

    promptText = text
    prompt = True

    while not waiting:
        plt.pause(1.0)
    waiting = False

    return response


def ita_start():
    global done
    done = False
    thread = threading.Thread(target=threadfunc)
    thread.start()


def ita_finish():
    global done
    done = True


def save_csv(name=None, profile=None, diffsys=None):
    """
    Save diffusion data as csv file.

    Parameters
    ----------
    name : str
        csv file name, default name is the profile name of diffsys name.
    profile : DiffProfile
        DiffProfile to save.
    diffsys : DiffSystem
        DiffSystem to save. diffsys can be saved by itself or with profile.
    """
    if profile is None and diffsys is None:
        raise ValueError('No data entered')
    if name is not None and not name.endswith('.csv'):
        name += str(name)+'.csv'
    elif profile is None:
        Xr, fD = diffsys.Xr, diffsys.Dfunc
        X, DC = np.array([]), np.array([])
        for i in range(diffsys.Np):
            Xnew = np.linspace(Xr[i, 0], Xr[i, 1], 30)
            Dnew = np.exp(splev(Xnew, fD[i]))
            X = np.append(X, Xnew)
            DC = np.append(DC, Dnew)
        data = pd.DataFrame({'X': X, 'DC': DC})
        if name is None:
            name = diffsys.name+'.csv'
        data.to_csv(name, index=False)
    elif diffsys is None:
        data = pd.DataFrame({'dis': profile.dis, 'X': profile.X})
        if name is None:
            name = profile.name+'.csv'
        data.to_csv(name, index=False)
    else:
        dis, X, Xr, fD = profile.dis, profile.X, diffsys.Xr, diffsys.Dfunc
        DC = np.zeros(len(dis))
        for i in range(diffsys.Np):
            pid = np.where((X >= Xr[i, 0]) & (X <= Xr[i, 1]))[0]
            DC[pid] = np.exp(splev(X[pid], fD[i]))
        data = pd.DataFrame({'dis': dis, 'X': X, 'DC': DC})
        if name is None:
            name = profile.name+'.csv'
        data.to_csv(name, index=False)


def read_csv(filename, Xlim=None, name=''):
    """
    Read diffusion data from csv file.

    Parameters
    ----------
    filename : str
        csv file path.
    Xlim : list (length of 2), optional
        A list to determine the two ends solubilities.
    name : str, optional
        Name the output DiffProfile and DiffSystem

    Returns
    -------
    profile : DiffProfile
        output DiffProfile object.
    diffsys : DiffSystem
        output DiffSystem object.
    """
    data = pd.read_csv(filename)
    if 'X' not in data.columns:
        raise ValueError('No column X in csv file')
    X = np.array(data['X'])

    # Auto rename
    if name == '':
        if '/' in filename:
            r = filename.rfind('/')
        elif '\\' in filename:
            r = filename.rfind('\\')
        else:
            r = -1
        if filename.endswith('.csv'):
            name = filename[r+1:-4]
        else:
            name = filename[r+1:]

    if 'dis' in data.columns:
        dis = np.array(data['dis'])
        If = []
        XIf = []
        for i in range(len(dis)-1):
            if dis[i] == dis[i+1]:
                If += [dis[i]]
                XIf += [X[i], X[i+1]]
        profile = DiffProfile(dis, X, If, name=name)
        if Xlim is None:
            XIf = np.array([X[0]] + XIf + [X[-1]])
        else:
            XIf = np.array([Xlim[0]] + XIf + [Xlim[-1]])
        Xr = XIf.reshape((len(XIf)//2, 2))
    if 'DC' in data.columns:
        DC = np.array(data['DC'])
    else:
        X = DC = None
    if Xr is not None:
        diffsys = DiffSystem(Xr, X=X, DC=DC, name=name)
    return profile, diffsys
