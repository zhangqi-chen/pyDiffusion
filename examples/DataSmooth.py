import pandas as pd
import matplotlib.pyplot as plt
from pydiffusion.core import DiffProfile
from pydiffusion.io import read_csv, save_csv
from pydiffusion.plot import profileplot
from pydiffusion.smooth import datasmooth

# Read raw data
data = pd.read_csv('examples/data/NiMo_exp.csv')
dis, X = data['dis'], data['X']
NiMo_exp = DiffProfile(dis=dis, X=X)

# Another way to read profile data, .csv must be created by pydiffusion.io.save_csv
NiMo_exp, _ = read_csv('examples/data/NiMo_exp.csv')

ax = plt.figure().add_subplot(111)
ax.set_title('Ni-Mo 1100C 1000hrs')
profileplot(NiMo_exp, ax, c='b', marker='o', ls='none', fillstyle='none')
plt.show(block=False)

# Data smoothing
NiMo_sm = datasmooth(NiMo_exp, [311.5, 340.5], n=500)

# Plot result
profileplot(NiMo_sm, ax, c='r')
plt.pause(1.0)
plt.show()

# Save result
save_csv('examples/data/NiMo_sm.csv', profile=NiMo_sm)
