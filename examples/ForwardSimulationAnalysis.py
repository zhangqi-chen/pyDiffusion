import matplotlib.pyplot as plt
from pydiffusion.io import read_csv, save_csv
from pydiffusion.plot import profileplot, DCplot, SFplot
from pydiffusion.Dtools import FSA

# Read required data for FSA
NiMo_sm, diffsys_init = read_csv('examples/data/NiMo_DC_init.csv')
NiMo_exp, _ = read_csv('examples/data/NiMo_exp.csv')
Xp = [[.05, .2],
      [.5, .515],
      [.985]]
diffsys_init.Xspl = Xp
time = 3600*1000

fig = plt.figure(figsize=(16, 6))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
profileplot(NiMo_exp, ax1, ls='none', c='b', marker='o', fillstyle='none')
profileplot(NiMo_sm, ax1, c='g', lw=2, label='NiMo_exp_smoothed')
SFplot(NiMo_sm, time, Xlim=[0, 1], ax=ax2, c='b', label='Sauer-Freise')
DCplot(diffsys_init, ax2, c='r', lw=2)
plt.pause(1.0)

# FSA
NiMo_sim, diffsys_fsa = FSA(NiMo_exp, NiMo_sm, diffsys_init, time, Xlim=[0, 1], n=[250, 300])

# Plot the results
fig = plt.figure(figsize=(16, 6))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
profileplot(NiMo_exp, ax1, ls='none', c='b', marker='o', fillstyle='none')
profileplot(NiMo_sm, ax1, c='g', lw=2, label='NiMo_exp_smoothed')
profileplot(NiMo_sim, ax1, c='r', lw=2, label='FSA simulated')
SFplot(NiMo_sm, time, Xlim=[0, 1], ax=ax2, c='b', label='Sauer-Freise')
DCplot(diffsys_fsa, ax2, c='r', lw=2, label='FSA')

# Save FSA result
save_csv('examples/data/NiMo.csv', NiMo_sim, diffsys_fsa)
