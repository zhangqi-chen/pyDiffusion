import pandas as pd
from pydiffusion.core import DiffProfile
from pydiffusion.io import read_csv, save_csv
from pydiffusion.plot import profileplot
from pydiffusion.smooth import datasmooth

# Read raw data
data = pd.read_csv('examples/data/NiMo_exp.csv')
dis, X = data['dis'], data['X']
NiMo_exp = DiffProfile(dis=dis, X=X, name='NiMo_exp')

# Another way to read profile data, .csv must be created by pydiffusion.io.save_csv
NiMo_exp, _ = read_csv('examples/data/NiMo_exp.csv')

ax = profileplot(NiMo_exp, c='b', marker='o', ls='none', fillstyle='none')

# Data smoothing
NiMo_sm = datasmooth(NiMo_exp, [311.5, 340.5], n=500)

# Plot result
profileplot(NiMo_sm, ax, c='r')

# Save result
save_csv('examples/data/NiMo_sm.csv', profile=NiMo_sm)
