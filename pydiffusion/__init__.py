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

pyDiffusion combines tools like diffusion simulation, diffusion data smooth,
forward simulation analysis (FSA), etc. to help people analyze diffusion data
efficiently.
"""
__version__ = '0.1.6'


from pydiffusion.core import DiffProfile, DiffSystem
from pydiffusion.simulation import mphSim
from pydiffusion.smooth import datasmooth
from pydiffusion.fsa import FSA
from pydiffusion.plot import profileplot, DCplot, SFplot
from pydiffusion.utils import matanocalc, mesh, step
from pydiffusion.Dtools import SF, Hall, Dmodel
from pydiffusion.io import read_csv, save_csv
