#Script for generating PFSS and outflow fields and plotting the open flux as a function of altitude

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


#Set up parameters
mfs = [0.0, 5e-17, 5e-17]
temps = [1e6, 1.5e6, 2.5e6]

rsses = np.linspace(1.5,5.0,8)

colors = sns.color_palette('colorblind')

ns = 180
nphi = 360
nrho = 120

obs_time = "2017-08-21T00:00:00"

hmi_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = True)

def find_oflux_profile(br):
    """
    Outputs the open flux profile (in the correct units) as an array aligned with radial grid centres
    """
    rsun_cm = 6.957e10
    ofluxes = np.zeros(np.shape(br)[2])
    for i in range(np.shape(br)[2]):
        surface_area = 4*np.pi*(np.exp(outflow_in.grid.rg[i])*rsun_cm)**2
        ofluxes[i] = np.sum(np.abs(outflow_out.br)[:,:,i])*surface_area/(nphi*ns)
    return ofluxes


fig = plt.figure(figsize = (6.9, 3.5))

for mi, model in enumerate(mfs[:]):
    mf_constant = mfs[mi]
    corona_temp = temps[mi]
    for ri, rss in enumerate(rsses[:]):
        print(mf_constant, corona_temp, rss)
        outflow_in = outflowpy.Input(hmi_map, nrho, rss, mf_constant = mf_constant, corona_temp = corona_temp)
        outflow_out = outflowpy.outflow_fortran(outflow_in)

        ofluxes = find_oflux_profile(outflow_out.br)

        if ri == 0:
            if mi == 0:
                plt.plot(np.exp(outflow_in.grid.rg), ofluxes, c = colors[mi], zorder = 0, linewidth = 1.5, label = "PFSS")
            else:
                plt.plot(np.exp(outflow_in.grid.rg), ofluxes, c = colors[mi], zorder = 0, linewidth = 1.5, label = f"$T_0 = {corona_temp*1e-6} \\times 10^6$ K")
        else:
            plt.plot(np.exp(outflow_in.grid.rg), ofluxes, c = colors[mi], zorder = 0, linewidth = 1.5)
        plt.scatter(np.exp(outflow_in.grid.rg)[-1], ofluxes[-1], color = colors[mi], edgecolor = 'black', s = 25)

plt.ylim(ymin = 0, ymax = 1.2e23)
plt.xlabel('Altitude $r/R_\odot$)')
plt.ylabel('Open Flux (Mx)')
plt.legend()
plt.tight_layout()
plt.savefig('2_openflux_comparison.pdf')
plt.show()
