from pydiffusion.io import read_csv, save_csv
from pydiffusion.plot import profileplot, DCplot, SFplot
from pydiffusion.Dtools import Dmodel

# Read smoothed NiMo data
NiMo_sm, _ = read_csv('examples/data/NiMo_sm.csv')
profileplot(NiMo_sm)

# DC modeling
time = 1000 * 3600
diffsys_init = Dmodel(NiMo_sm, time, Xlim=[0, 1])

# Plot results
ax = SFplot(NiMo_sm, time, Xlim=[0, 1])
DCplot(diffsys_init, ax)

# DC modeling automatically
Xspl = [[.05, .2],
        [.5, .515],
        [.985]]
diffsys_init_auto = Dmodel(NiMo_sm, time, Xspl=Xspl, Xlim=[0, 1])

# Save result
save_csv('examples/data/NiMo_DC_init.csv', profile=NiMo_sm, diffsys=diffsys_init_auto)
