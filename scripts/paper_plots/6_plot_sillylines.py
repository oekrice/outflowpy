#This one for making simple plots of the field line shapes. Let's see how these come out...

import sys, random, time
import os
from datetime import datetime, timedelta
from pathlib import Path
import outflowpy
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import astropy.constants as const
import matplotlib.patches as patches
from cycler import cycler


plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # LaTeX-like Computer Modern
    "font.family": "serif",
})


colors = sns.color_palette('colorblind')

ns = 30#180
nphi = 90#360
nrho = 60


obs_time = "2017-08-21T00:00:00"
mfs = [0.0, 5e-17]
temps = [1e6, 2.6255e6]
rsses = [1.5898, 2.5]

hmi_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = True)

fig, axs = plt.subplots(1,2, figsize = (6.9, 4.0))

model_labs = ["PFSS, $r_{ss} = 1.59R_\odot$", "Outflow, $T_0 = 2.63$MK"]

for mi, model in enumerate(mfs[:]):

    mf_constant = mfs[mi]
    corona_temp = temps[mi]
    rss = rsses[mi]

    ax = axs[mi]
    print(mf_constant, corona_temp, rss)
    outflow_in = outflowpy.Input(hmi_map, nrho, rss, mf_constant = mf_constant, corona_temp = corona_temp)
    outflow_out = outflowpy.outflow_fortran(outflow_in)

    seeds = outflowpy.utils.equal_seed_sampler(outflow_out, 100, 1.0+(rss - 1.0)*0.5)

    tracer = outflowpy.tracing.FastTracer()

    field_lines = tracer.trace(seeds, outflow_out, save_flag = True)

    transformed_lines = []

    for fi, fline in enumerate(field_lines):
        coords = fline.coords
        coords.representation_type = 'cartesian'
        line = np.zeros((2, len(coords)))
        line[0,:] = coords.y/const.R_sun; line[1,:] = coords.z/const.R_sun
        transformed_lines.append([line[0,:], line[1,:]])

        ax.plot(line[0,:], line[1,:], c = colors[mi * 2], linewidth = 0.5, zorder = 0)

    circle = patches.Circle(
        (0.0, 0.0),
        1.0,
        facecolor="#fc9e23",
        edgecolor="black",
        linewidth=1
    )

    ax.add_patch(circle)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.set_title(f"{model_labs[mi]}")
cmap = sns.diverging_palette(220, 20, as_cmap=True)



plt.tight_layout()
plt.savefig('6_plot_sillylines.pdf')
plt.show()
