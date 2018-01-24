import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from pyFSA.utils import DiffProfile


def movingradius(dis, X, r):
    dmin, dmax = dis[0], dis[-1]
    n = np.size(dis)
    f = splrep(dis, X, k=1)
    Xnew = np.zeros(n)
    for i in range(n):
        h = min(abs(dmin-dis[i]), abs(dmax-dis[i]), r)
        Xnew[i] = np.mean(splev(np.linspace(dis[i]-h, dis[i]+h, 100), f))
    return Xnew


def phasesmooth(dis, X):
    Xsm = X.copy()
    smoo = True
    while smoo:
        plt.figure('phase')
        plt.cla()
        plt.plot(dis, Xsm, 'bo')
        plt.show()
        ipt = input(
            'Enter the zoomed in start and end location, if no, enter nothing\n')
        zm = [float(i) for i in ipt.split(
            ' ')] if ipt != '' else [dis[0], dis[-1]]
        zmid = np.where((dis >= zm[0]) & (dis <= zm[1]))[0]
        plt.cla()
        plt.plot(dis[zmid], Xsm[zmid], 'bo')
        plt.show()
        sm = True
        while sm:
            Xsmn = np.copy(Xsm[zmid])
            msg = 'Enter Start and End Composition for this region: '
            msg += str(Xsmn[0])+' '+str(Xsmn[-1])+'\n'
            ipt = input(msg)
            if ipt != '':
                Xsmn[0], Xsmn[-1] = [float(i) for i in ipt.split(' ')]
            msg = 'Smooth Radius and Times: [1 1]\n'
            ipt = input(msg)
            if ipt != '':
                ipt = ipt.split(' ')
                r, t = float(ipt[0]), int(ipt[1])
            else:
                r, t = 1.0, 1
            for i in range(t):
                Xsmn = movingradius(dis[zmid], Xsmn, r)
            plt.cla()
            plt.plot(dis[zmid], Xsm[zmid], 'bo', dis[zmid], Xsmn, 'ro')
            plt.show()
            ipt = input('Redo this smooth? (y/[n])')
            sm = True if 'y' in ipt or 'Y' in ipt else False
            if not sm:
                Xsm[zmid] = Xsmn
        plt.cla()
        plt.plot(dis, X, 'bo', dis, Xsm, 'ro')
        plt.show()
        ipt = input('Further smooth for this phase? (y/[n])')
        smoo = True if 'y' in ipt or 'Y' in ipt else False
    return Xsm


def datasmooth(dis, X, interface=[], n=2000):
    dis, X = np.array(dis), np.array(X)
    assert len(dis) == len(X), 'Nonequal length of distance and composition data'
    Np = len(interface)+1
    If = np.zeros(Np+1)
    If[1:-1] = interface
    If[0], If[-1] = dis[0]-0.5, dis[-1]+0.5
    Ip = [0]*(Np+1)
    disn, Xn = dis.copy(), X.copy()
    for i in range(Np):
        pid = np.where((disn > If[i]) & (disn < If[i+1]))[0]
        Xn[pid] = phasesmooth(disn[pid], Xn[pid])
    for i in range(1, Np):
        pid = np.where(disn > If[i])[0][0]
        start = max(pid-5, np.where(disn > If[i-1])[0][0])
        end = min(pid+5, np.where(disn < If[i+1])[0][-1])
        fX1 = splrep(disn[start:pid], Xn[start:pid], k=2)
        fX2 = splrep(disn[pid:end], Xn[pid:end], k=2)
        disn = np.insert(disn, pid, [If[i], If[i]])
        Xn = np.insert(Xn, pid, [splev(If[i], fX1), splev(If[i], fX2)])
        Ip[i] = pid+1
    Ip[-1] = len(Xn)
    disni, Xni = disn.copy(), Xn.copy()
    if n > 0:
        ni = [int(n*(If[i]-If[0])//(If[-1]-If[0])) for i in range(Np)]+[n]
        disni, Xni = np.zeros(n), np.zeros(n)
        for i in range(Np):
            fX = splrep(disn[Ip[i]:Ip[i+1]], Xn[Ip[i]:Ip[i+1]], k=1)
            disni[ni[i]:ni[i+1]] = np.linspace(disn[Ip[i]], disn[Ip[i+1]-1], ni[i+1]-ni[i])
            Xni[ni[i]:ni[i+1]] = splev(disni[ni[i]:ni[i+1]], fX)
    plt.figure('Smooth Result')
    plt.cla()
    plt.plot(dis, X, 'bo', fillstyle='none')
    plt.plot(disni, Xni, 'r-', lw=2)
    plt.show()

    return DiffProfile(disni, Xni, If=If)
