"""
pyDiffusion combines tools like diffusion simulation, diffusion data smooth,
forward simulation analysis (FSA), etc. to help people analyze diffusion data
efficiently.
"""
from pydiffusion.core import DiffProfile, DiffSystem
from pydiffusion.simulation import mphSim, ErrorAnalysis
from pydiffusion.fsa import FSA
from pydiffusion.plot import profileplot, DCplot
from pydiffusion.utils import matanocalc
from pydiffusion.Dmodel import SF, Dmodel
