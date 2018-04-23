import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pydiffusion.core import DiffSystem
from pydiffusion.utils import step, mesh
from pydiffusion.simulation import mphSim
from pydiffusion.plot import profileplot, DCplot
from pydiffusion.io import read_csv

# Create diffusion system with constant DC
diffsys = DiffSystem(Xr=[0, 1], X=[0, 1], DC=[1e-14, 1e-14])

# Create initial step profile
dis = mesh(0, 1000, 501)
profile_init = step(dis, 500, diffsys)

fig = plt.figure(figsize=(16, 6))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
ax1.set_title('Diffusion Coefficients', fontsize=15)
ax2.set_title('Initial Step Profile', fontsize=15)
DCplot(diffsys, ax1)
profileplot(profile_init, ax2)
plt.show()

# Diffusion simulation using the setups
time = 200 * 3600
profile_final = mphSim(profile_init, diffsys, time)

ax = plt.figure(figsize=(8, 6)).add_subplot(111)
profileplot(profile_init, ax, ls='--')
profileplot(profile_final, ax, c='r')
plt.show()

# Read diffusion coefficients data of Ni-Mo system
data = pd.read_csv('data/NiMo.csv')
X, DC = data['X'], data['DC']
Xr = np.array([[0, 0.25],
               [0.49, 0.53],
               [0.97, 1]])
diffsys_NiMo = DiffSystem(Xr=Xr, X=X, DC=DC)

# Read diffusion coefficients from saved data file by pydiffusion.io.save_csv
_, diffsys_NiMo = read_csv('data/NiMo.csv', [0, 1])

# Create initial step profile
dis = mesh(0, 400, 301)
profile_NiMo_init = step(dis, 300, diffsys_NiMo)

# Perform simulation
profile_NiMo = mphSim(profile_NiMo_init, diffsys_NiMo, 800*3600)

# Plot results
fig = plt.figure(figsize=(16, 6))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
ax1.set_title('Diffusion Coefficients of Ni-Mo at 1100C', fontsize=15)
ax2.set_title('Diffusion Simulation for 800 hours', fontsize=15)
DCplot(diffsys_NiMo, ax1)
profileplot(profile_NiMo_init, ax2, ls='--')
profileplot(profile_NiMo, ax2, c='r')
plt.show()
