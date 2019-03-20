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

The test_simulation module contains tests for simulation in pydiffusion.
"""

import numpy as np
from pydiffusion.core import DiffSystem, DiffProfile
from pydiffusion.utils import mesh, step, matanocalc
from pydiffusion.simulation import sphSim, mphSim


def test_sphsim():
    """
    Single-phase system simulation.
    Offset of the simulated matano plane should be very small.
    """
    diffsys = DiffSystem(Xr=[0, 1], X=[0, 1], DC=[1e-14, 1e-14])
    dis = mesh(0, 1000, 201)
    profile_init = step(dis, 500, diffsys)
    time = 200 * 3600
    profile_final = sphSim(profile_init, diffsys, time)

    mpi = matanocalc(profile_init, [0, 1])
    mpf = matanocalc(profile_final, [0, 1])

    assert isinstance(profile_final, DiffProfile)
    assert len(profile_final.If) == diffsys.Np+1
    assert abs(mpi-mpf) < 1


def test_mphsim():
    """
    Multiple-phase system simulation.
    Offset of the simulated matano plane should be very small.
    """
    Xr = [[0, .4], [.6, 1]]
    X = np.linspace(0, 1, 101)
    DC = np.linspace(1, 2, 101)*1e-14
    diffsys = DiffSystem(Xr=Xr, X=X, DC=DC)
    dis = mesh(0, 1000, 201)
    profile_init = step(dis, 500, diffsys)
    time = 200 * 3600
    profile_final = mphSim(profile_init, diffsys, time)

    mpi = matanocalc(profile_init, [0, 1])
    mpf = matanocalc(profile_final, [0, 1])

    assert isinstance(profile_final, DiffProfile)
    assert len(profile_final.If) == diffsys.Np+1
    assert abs(mpi-mpf) < 1
