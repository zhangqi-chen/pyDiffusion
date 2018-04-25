import matplotlib.pyplot as plt
from pydiffusion.io import read_csv, save_csv
from pydiffusion.plot import profileplot, DCplot, SFplot
from pydiffusion.Dmodel import Dmodel

# Read smoothed NiMo data
NiMo_sm, _ = read_csv('data/NiMo_sm.csv')

ax = plt.figure(figsize=(8, 6)).add_subplot(111)
profileplot(NiMo_sm, ax)
plt.show()

# DC modeling
time = 1000 * 3600
diffsys_init = Dmodel(NiMo_sm, time, Xlim=[0, 1])

# Plot results
ax = plt.figure(figsize=(8, 6)).add_subplot(111)
SFplot(NiMo_sm, time, Xlim=[0, 1], ax=ax)
DCplot(diffsys_init, ax, c='r')
plt.show()

# DC modeling automatically
Xspl = [[.05, .2],
        [.5, .515],
        [.985]]
diffsys_init_auto = Dmodel(NiMo_sm, time, Xspl=Xspl, Xlim=[0, 1])

# Save result
save_csv('NiMo_DC_init.csv', profile=NiMo_sm, diffsys=diffsys_init_auto)
