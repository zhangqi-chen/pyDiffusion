import matplotlib.pyplot as plt
from pydiffusion.io import read_csv
from pydiffusion.plot import profileplot
from pydiffusion.smooth import datasmooth

# Read Ni-Mo 1100C 1000 hours raw data
NiMo_exp, _ = read_csv('NiMo_exp.csv')
ax = plt.figure().add_subplot(111)
ax.set_title('Ni-Mo 1100C 1000hrs')
profileplot(NiMo_exp, ax, c='b', marker='o', ls='none', fillstyle='none')
plt.show(block=False)

# Data smoothing
NiMo_sm = datasmooth(NiMo_exp, [311.5, 340.5])

# Plot results
profileplot(NiMo_sm, ax, c='r')
plt.pause(1.0)
plt.show()
