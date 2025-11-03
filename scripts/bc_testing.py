"""
This is a test script to compare the pfsspy routine with various boundary conditions in the fortran code. Will need to recompile using f2py each time I tweak it I suppose.
"""

import os
import sys

import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
import numpy as np

import outflowpy
import outflowpy.utils

import pfsspy
import pfsspy.utils

#Testing (with extreme outflow) the various options for the precise lower boundary condition
colors = plt.get_cmap("tab10")

if False:
    nrho = 30
    nphi = nrho*6
    ns = nrho*3

    phi = np.linspace(0, 2 * np.pi, nphi)
    s = np.linspace(-1, 1, ns)
    s, phi = np.meshgrid(s, phi)

    def dipole_Br(r, s):
        return 2 * s

    br = dipole_Br(1, s)

    nrho = nrho
    rss = 2.5

    header = outflowpy.utils.carr_cea_wcs_header(Time('2020-1-1'), br.shape)
    input_map = sunpy.map.Map((br.T, header))

else:
    ns_target = 90; nphi_target = 180
    input_map = outflowpy.obtain_data.prepare_hmi_crot(2210, ns_target, nphi_target, smooth = 5e-2/nphi_target)   #Outputs the set of data corresponding to this particular Carrington rotation.
    #Calculate pfss field, for now. So I don't have to write any code to do the basic testing
    nrho = 30
    rss = 2.5

def find_oflux_profile(outflow_out):
    br_out = outflow_out.br
    ofluxes = np.zeros(np.shape(br_out)[2])
    for ri in range(np.shape(br_out)[2]):
        surface_area = 4*np.pi*np.exp(outflow_in.grid.rg[ri])**2
        oflux = np.sum(np.abs(br_out)[:,:,ri])*surface_area
        ofluxes[ri] = oflux
    return ofluxes

fix, axs = plt.subplots(1,3, figsize = (10,7))

for ai, ax in enumerate(axs):
    mfs = [0.0,5e-17,1.0]
    outflow_in = outflowpy.Input(input_map, nrho, rss, mf_constant = mfs[ai])

    #outflow_out = outflowpy.outflow(outflow_in)
    outflow_out = outflowpy.outflow_fortran(outflow_in)

    ax.set_aspect('equal')

    # Take 32 start points spaced equally in theta
    r = 1.01 * const.R_sun
    lon = np.pi / 2 * u.rad
    lat = np.linspace(-np.pi / 2, np.pi / 2, 33) * u.rad
    seeds = SkyCoord(lon, lat, r, frame=outflow_out.coordinate_frame)

    tracer = outflowpy.tracing.FortranTracer()
    field_lines = tracer.trace(seeds, outflow_out)

    for field_line in field_lines:
        coords = field_line.coords
        coords.representation_type = 'cartesian'
        color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
        ax.plot(coords.y / const.R_sun,
                coords.z / const.R_sun, color=color)

    # Add inner and outer boundary circles
    ax.add_patch(mpatch.Circle((0, 0), 1, color='k', fill=False))
    ax.add_patch(mpatch.Circle((0, 0), outflow_in.grid.rss, color='k', linestyle='--',
                            fill=False))


    ofluxes = find_oflux_profile(outflow_out)

plt.show()




