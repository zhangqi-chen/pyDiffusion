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

The test_model module contains tests for DiffProfile and DiffSystem objects.
"""

import numpy as np
from pydiffusion.core import DiffProfile, DiffSystem


def test_profile(): 
    """
    DiffProfile constructor
    """
    dis = np.linspace(100, 0, 101)
    X = np.linspace(0, 1, 101)
    If = [30.5, 60.5]
    profile = DiffProfile(dis=dis, X=X, If=If)

    assert len(profile.dis) == len(profile.X)
    assert isinstance(profile.name, str)
    assert np.all(profile.dis[1:] >= profile.dis[:-1])
    assert len(profile.If) == len(If)+2
    assert len(profile.Ip) == len(If)+2
    assert profile.Ip[-1] == len(profile.dis)
    assert profile.If[0] < profile.dis[profile.Ip[0]]
    assert profile.If[-1] > profile.dis[profile.Ip[-1]-1]
    for i in range(1, len(If)):
        assert profile.If[i] > profile.dis[profile.Ip[i]-1]
        assert profile.If[i] < profile.dis[profile.Ip[i]]


def test_system():
    """
    DiffSystem constructor
    """
    # Construct with X, DC
    Xr = [[0, .4], [.6, 1]]
    X = np.linspace(0, 1, 101)
    DC = np.linspace(1, 10, 101)*1e-14
    diffsys = DiffSystem(Xr=Xr, X=X, DC=DC)

    assert diffsys.Xr.shape == (diffsys.Np, 2)
    assert isinstance(diffsys.name, str)
    assert len(diffsys.Dfunc) == diffsys.Np
    for i in range(diffsys.Np):
        assert len(diffsys.Dfunc[i][0]) == len(diffsys.Dfunc[i][1])

    # Construct with Dfunc
    diffsys1 = DiffSystem(Xr=Xr, Dfunc=diffsys.Dfunc)

    assert diffsys1.Xr.shape == (diffsys1.Np, 2)
    assert len(diffsys1.Dfunc) == diffsys1.Np
