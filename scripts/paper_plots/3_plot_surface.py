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


colors = sns.color_palette('colorblind')

ns = 180
nphi = 360
nrho = 60

obs_time = "2017-08-21T00:00:00"

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

hmi_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = True)
outflow_in = outflowpy.Input(hmi_map, nrho, 2.5, mf_constant = 0.0)

cmap = sns.diverging_palette(220, 20, as_cmap=True)

fig, axs = plt.subplots(2,1, subplot_kw = {"projection": "mollweide"}, figsize = (6.9, 5.0))

#Find plot scales (including stretching -- IMPORTANT)
raw_data = np.load('./data/raw_data_2017.npy')

downscale_factor = 4
nxl = np.shape(raw_data)[0]//downscale_factor
nyl = np.shape(raw_data)[1]//downscale_factor
raw_data = raw_data.reshape(nxl, downscale_factor, nyl, downscale_factor).mean(axis=(1,3))

ax = axs[0]

lon_edges = np.linspace(-np.pi, np.pi, np.shape(raw_data)[1] + 1)
lat_edges = np.arccos(np.linspace(1., -1., np.shape(raw_data)[0] + 1)) - np.pi/2
im = ax.pcolormesh(lon_edges, lat_edges, raw_data, vmin = -50, vmax = 50, cmap = cmap, rasterized=True)
ax.patch.set_linewidth(3)   # or any thickness you want
ax.patch.set_edgecolor("black")
#
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im, label = 'Magnetic field strength (Gauss)')

ax.set_title('Raw magnetic field data')

ax = axs[1]

data = outflow_in.br
lon_edges = np.linspace(-np.pi, np.pi, nphi+1)
lat_edges = np.arccos(np.linspace(1., -1., ns+1)) - np.pi/2
im = ax.pcolormesh(lon_edges, lat_edges, data, vmin = -50, vmax = 50, cmap = cmap, rasterized=True)
ax.patch.set_linewidth(3)   # or any thickness you want
ax.patch.set_edgecolor("black")
#
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im, label = 'Magnetic field strength (Gauss)')

ax.set_title('Smoothed magnetic field data')

plt.tight_layout()
plt.savefig('3_plot_surface.pdf')
plt.show()
