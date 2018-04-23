import matplotlib.pyplot as plt
from pydiffusion.io import read_csv
from pydiffusion.plot import profileplot, DCplot, SFplot
from pydiffusion.fsa import FSA

# Read required data for FSA
NiMo_sm, diffsys_init = read_csv('data/NiMo_DC_init.csv')
NiMo_exp, _ = read_csv('data/NiMo_exp.csv')
Xp = [[.05, .2],
      [.5, .515],
      [.985]]
diffsys_init.Xspl = Xp
time = 3600*1000

fig = plt.figure(figsize=(16, 6))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
profileplot(NiMo_exp, ax1, ls='none', marker='o', fillstyle='none')
profileplot(NiMo_sm, ax1, c='g', lw=2)
SFplot(NiMo_sm, time, ax=ax2, Xlim=[0, 1])
DCplot(diffsys_init, ax2, c='r', lw=2)
plt.show()

# FSA
NiMo_sim, diffsys_fsa = FSA(NiMo_exp, NiMo_sm, diffsys_init, time, Xlim=[0, 1], n=[300, 350])

# Plot the results
fig = plt.figure(figsize=(16, 6))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
profileplot(NiMo_exp, ax1, ls='none', marker='o', fillstyle='none')
profileplot(NiMo_sm, ax1, c='g', lw=2)
profileplot(NiMo_sim, ax1, c='r', lw=2)
SFplot(NiMo_sm, time, ax=ax2, Xlim=[0, 1])
DCplot(diffsys_fsa, ax2, c='r', lw=2)
plt.show()
