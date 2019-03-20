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

The test_utils module contains tests for pydiffusion utilities.
"""

import numpy as np
from scipy.interpolate import splev
from pydiffusion.core import DiffSystem
from pydiffusion.io import read_csv
from pydiffusion.simulation import sphSim
from pydiffusion.utils import mesh, automesh, step, profilefunc, disfunc


def test_mesh():
    """
    mesh should return a meshed array with increasing/decreasing grid size.
    """
    dis = mesh(n=100, a=1)
    d = dis[1:]-dis[:-1]
    assert len(dis) == 100
    assert np.all(d[1:] > d[:-1])

    dis = mesh(n=100, a=-1)
    d = dis[1:]-dis[:-1]
    assert np.all(d[1:] < d[:-1])


def test_automesh():
    """
    automesh should return a meshed array whose length is within its range.
    """
    profile, diffsys = read_csv('dataset.csv', Xlim=[0, 1])
    dism = automesh(profile, diffsys, n=[300, 400])

    assert len(dism) >= 300 and len(dism) <= 400


def test_dispfunc():
    """
    disfunc and profilefunc should give functions to copy profile data.
    """
    diffsys = DiffSystem(Xr=[0, 1], X=[0, 1], DC=[1e-14, 1e-13])
    dis = mesh(0, 1000, 201)
    profile_init = step(dis, 500, diffsys)
    time = 200 * 3600
    profile = sphSim(profile_init, diffsys, time)

    fX = profilefunc(profile)
    fdis = disfunc(profile.dis, profile.X)

    assert np.all(abs(splev(dis, fX)-profile.X) < 0.01)
    assert np.all(abs(splev(profile.X, fdis)-dis) < 0.1)
