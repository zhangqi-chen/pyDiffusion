import matplotlib.pyplot as plt
from pydiffusion.io import read_csv
from pydiffusion.utils import step, mesh, matanocalc, DCbias
from pydiffusion.simulation import ErrorAnalysis
from pydiffusion.plot import DCplot, profileplot

# Read data, create bias
profile_fsa, diffsys_TiZr = read_csv('TiZr.csv', [0, 1])
profile_exp, _ = read_csv('TiZr_exp.csv')
diffsys_bias = DCbias(diffsys_TiZr, 0.2, 0.1)

ax = plt.figure(figsize=(8, 6)).add_subplot(111)
DCplot(diffsys_TiZr, ax, label='original')
DCplot(diffsys_bias, ax, c='r', ls='--', label='bias')
plt.legend(fontsize=15)
plt.show()

# Error analysis with low accuracy
dism = mesh(profile_fsa, diffsys_TiZr, [300, 350])
mp = matanocalc(profile_fsa, [0, 1])
profile_init = step(dism, mp, diffsys_TiZr)
time = 100*3600
error_result = ErrorAnalysis(profile_exp, profile_init, diffsys_TiZr, time, loc=3, accuracy=1e-2, output=True)

fig = plt.figure(figsize=(16, 6))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
DCplot(diffsys_TiZr, ax1, error_result)
profileplot(profile_fsa, ax2, error_result)
profileplot(profile_exp, ax2, marker='o', ls='none', fillstyle='none')
plt.show()

# Error analysis with high accuracy
error_result2 = ErrorAnalysis(profile_exp, profile_init, diffsys_TiZr, time, loc=21, accuracy=1e-3, output=False)

fig = plt.figure(figsize=(16, 6))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
DCplot(diffsys_TiZr, ax1, error_result2)
profileplot(profile_fsa, ax2, error_result2)
profileplot(profile_exp, ax2, marker='o', ls='none', fillstyle='none')
plt.show()
