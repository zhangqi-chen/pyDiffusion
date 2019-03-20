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

The test_plot verifies that the plot module produce plots without error.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pydiffusion.core import DiffSystem, DiffProfile
from pydiffusion.utils import mesh, step
from pydiffusion.simulation import sphSim
from pydiffusion.plot import profileplot, DCplot, SFplot
plt.switch_backend('Agg')


def test_profileplot():
    """
    profileplot should return an axes object when a DiffProfile is passed
    """
    dis = np.linspace(0, 100, 101)
    X = np.linspace(0, 1, 101)
    If = [30.5, 60.5]
    profile = DiffProfile(dis=dis, X=X, If=If)
    ax = profileplot(profile, label='test')

    assert isinstance(ax, Axes)


def test_DCplot():
    """
    DCplot should return an axes object when a DiffSystem is passed
    """
    Xr = [[0, .4], [.6, 1]]
    X = np.linspace(0, 1, 101)
    DC = np.linspace(1, 10, 101)*1e-14
    diffsys = DiffSystem(Xr=Xr, X=X, DC=DC)
    ax = DCplot(diffsys, label='test')

    assert isinstance(ax, Axes)


def test_SFplot():
    """
    SFplot should return an axes object when a DiffProfile and time is passed
    """
    diffsys = DiffSystem(Xr=[0, 1], X=[0, 1], DC=[1e-14, 1e-13])
    dis = mesh(0, 1000, 201)
    profile_init = step(dis, 500, diffsys)
    time = 200 * 3600
    profile = sphSim(profile_init, diffsys, time)
    ax = SFplot(profile, time, Xlim=[0, 1], label='test')

    assert isinstance(ax, Axes)
