#Script for generating some outflow solutions based on the coronal temperature and mf constant

import sys, random, time
import os
from datetime import datetime, timedelta
from pathlib import Path
import outflowpy
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from cycler import cycler

plt.rcParams['axes.prop_cycle'] = cycler('linestyle', ['-', '--', '-.', ':'])
# plt.rcParams.update({
#     "text.usetex": True,                 # Use LaTeX for all text
#     "font.family": "serif",              # Use serif (Computer Modern)
#     "font.serif": ["Computer Modern"],   # Explicitly choose CM
#     "axes.unicode_minus": False          # Fix minus sign in labels
# })

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # LaTeX-like Computer Modern
    "font.family": "serif",
})

colors = sns.color_palette('colorblind')

def generate_fn(corona_temp, mf_constant):

    mf_in_sensible_units = mf_constant*(6.957e10)**2 #In seconds/solar radius
    sound_speed = np.sqrt(1.38064852e-23*corona_temp/1.67262192e-27) #Sound speed in m/s
    r_c = (6.67408e-11*1.98847542e30/(2*sound_speed**2))/(6.957e8)   #Critical radius in solar radii (code units)
    c_s = mf_in_sensible_units*sound_speed/6.957e8  #Sound speed in seconds/solar radius (code units)

    print('Radius', r_c)
    print('Speed', c_s)

generate_fn(1e6, 5e-17)

ns = 60
nphi = 120
nrho = 120

obs_time = "1999-01-01T00:00:00"

print('Downloading data')

hmi_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi)   

temps = [0.5e6, 1e6, 1.5e6, 2e6, 3e6]

fig = plt.figure(figsize = (6.9, 3.5))

for i, corona_temp in enumerate(temps):
    mf_in_sensible_units = 5e-17*(6.957e10)**2 #In seconds/solar radius
    sound_speed = np.sqrt(1.38064852e-23*corona_temp/1.67262192e-27) #Sound speed in m/s
    r_c = (6.67408e-11*1.98847542e30/(2*sound_speed**2))/(6.957e8)   #Critical radius in solar radii (code units)
    c_s = mf_in_sensible_units*sound_speed/6.957e8  #Sound speed in seconds/solar radius (code units)

    outflow_in = outflowpy.Input(hmi_map, nrho, 5.0, mf_constant = 5e-17, corona_temp = corona_temp)

    plt.plot(np.exp(outflow_in.grid.rg), outflow_in.vg, c= colors[i], label = f"$T_0 = {corona_temp*1e-6} \\times 10^6$", linewidth = 2.)

plt.xlabel('Radius $r$ ($R_\odot$)')
plt.ylabel('Outflow speed $v_(r)$ ($s/R_\odot$)')
plt.legend()
plt.tight_layout()
plt.savefig('0_outflow_speeds.pdf')
plt.show()
