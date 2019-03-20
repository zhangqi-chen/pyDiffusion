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
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pydiffusion.io import read_csv
from pydiffusion.plot import profileplot, DCplot, SFplot
plt.switch_backend('Agg')


def test_profileplot():
    """
    profileplot should return an axes object when a DiffProfile is passed
    """
    profile, _ = read_csv('dataset.csv', Xlim=[0, 1])
    ax = profileplot(profile, label='test')

    assert isinstance(ax, Axes)


def test_DCplot():
    """
    DCplot should return an axes object when a DiffSystem is passed
    """
    _, diffsys = read_csv('dataset.csv', Xlim=[0, 1])
    ax = DCplot(diffsys, label='Dtest')

    assert isinstance(ax, Axes)


def test_SFplot():
    """
    SFplot should return an axes object when a DiffProfile and time is passed
    """
    profile, _ = read_csv('dataset.csv', Xlim=[0, 1])
    time = 1000*3600
    ax = SFplot(profile, time, Xlim=[0, 1], label='SFtest')

    assert isinstance(ax, Axes)
