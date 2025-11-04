"""
This is a test script to compare the pfsspy routine with various boundary conditions in the fortran code. Will need to recompile using f2py each time I tweak it I suppose.

The conditions which can reasonably be changed are
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

colors = plt.get_cmap("tab10")

nrs = [30]

#Basic plot of open flux with height, as a first order comparison?
#Need to multiply the field strength by the area (just keep in Rsuns for simplicity)

def find_oflux_profile(outflow_out):
    br_out = outflow_out.br
    ofluxes = np.zeros(np.shape(br_out)[2])
    for ri in range(np.shape(br_out)[2]):
        surface_area = 4*np.pi*np.exp(outflow_in.grid.rg[ri])**2
        oflux = np.sum(np.abs(br_out)[:,:,ri])*surface_area
        ofluxes[ri] = oflux
    return ofluxes

fig = plt.figure()

for count, nrho in enumerate(nrs):

    #Test the comparison of python, fortran and pfsspy.
    #Now appears to be matching correctly, but is still some ambiguity in the numerics for the lower boundary condition
    nphi = nrho*6
    ns = nrho*3

    phi = np.linspace(0, 2 * np.pi, nphi)
    s = np.linspace(-1, 1, ns)
    s, phi = np.meshgrid(s, phi)

    def dipole_Br(r, s):
        return 2 * s**7

    br = dipole_Br(1, s)

    nrho = nrho
    rss = 2.5

    header = outflowpy.utils.carr_cea_wcs_header(Time('2020-1-1'), br.shape)
    input_map = sunpy.map.Map((br.T, header))

    outflow_in = outflowpy.Input(input_map, nrho, rss, mf_constant = 0.0)

    pfss_out = outflowpy.pfss(outflow_in)
    pfss_profile = find_oflux_profile(pfss_out)
    pfss_profile = pfss_profile/pfss_profile[0]
    plt.plot(np.exp(outflow_in.grid.rg), pfss_profile, label = 'pfsspy')

    outflow_out = outflowpy.outflow(outflow_in)
    oflux_profile = find_oflux_profile(outflow_out)
    oflux_profile = oflux_profile/oflux_profile[0]
    plt.plot(np.exp(outflow_in.grid.rg), oflux_profile, label = 'outflowpy python')

    outflow_out = outflowpy.outflow_fortran(outflow_in)
    oflux_profile_fort = find_oflux_profile(outflow_out)
    oflux_profile_fort = oflux_profile_fort/oflux_profile_fort[0]
    plt.plot(np.exp(outflow_in.grid.rg), oflux_profile_fort, label = 'outflowpy fortran')

    print('Squared error', np.sum((oflux_profile-pfss_profile)**2)/len(oflux_profile))

plt.legend()
plt.show()



