"""
Writing a test script to test this all works (very basically) with everything set up to be identical/very compatible with pfsspy stuff
I think it's very important people can just change the import and things just work -- but the tests should take care of that maybe?
Will need to learn how that actually works though
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

# import pfsspy
# import pfsspy.utils

if False: #Dipole test
    nphi = 180
    ns = 90

    phi = np.linspace(0, 2 * np.pi, nphi)
    s = np.linspace(-1, 1, ns)
    s, phi = np.meshgrid(s, phi)

    def dipole_Br(r, s):
        return 2 * s / r**3

    br = dipole_Br(1, s)

    nrho = 30
    rss = 2.5

    header = outflowpy.utils.carr_cea_wcs_header(Time('2020-1-1'), br.shape)
    input_map = sunpy.map.Map((br.T, header))

else:   #Hmi test
    ns_target = 90; nphi_target = 180
    input_map = outflowpy.obtain_data.prepare_hmi_crot(2210, ns_target, nphi_target, smooth = 5e-2/nphi_target)   #Outputs the set of data corresponding to this particular Carrington rotation.
    #Calculate pfss field, for now. So I don't have to write any code to do the basic testing
    nrho = 30
    rss = 2.5

outflow_in = outflowpy.Input(input_map, nrho, rss, mf_constant = 5e-17)

# outflow_out = outflowpy.outflow(outflow_in)
# python_test = outflow_out.br, outflow_out.bs, outflow_out.bp

outflow_out = outflowpy.outflow_fortran(outflow_in)
fortran_test = outflow_out.br, outflow_out.bs, outflow_out.bp

# for ti, test_field in enumerate(python_test):
#     print(np.max(np.abs(test_field - fortran_test[ti])))
#     print(np.allclose(test_field, fortran_test[ti], atol = 1e-8))

if False:  #Plot the outflow function
    plt.plot(np.exp(outflow_in.grid.rg), outflow_in.vg)
    plt.plot(np.exp(outflow_in.grid.rcx), outflow_in.vcx)
    plt.plot(np.exp(outflow_in.grid.rcx), outflow_in.vdcx)
    plt.show()

ss_br = outflow_out.source_surface_br

# Create the figure and axes
fig = plt.figure()
ax = plt.subplot(projection=ss_br)
# Plot the source surface map
ss_br.plot()
# Plot formatting
plt.colorbar()
ax.set_title('Source surface magnetic field')

plt.show()
#
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
#
# # Take 32 start points spaced equally in theta
# r = 2.49 * const.R_sun
# lon = np.pi / 2 * u.rad
# lat = np.linspace(-np.pi/2, np.pi/2, 33) * u.rad
# seeds = SkyCoord(lon, lat, r, frame=outflow_out.coordinate_frame)
#
# tracer = pfsspy.tracing.FortranTracer()
# field_lines = tracer.trace(seeds, outflow_out)
#
# for field_line in field_lines:
#     coords = field_line.coords
#     coords.representation_type = 'cartesian'
#     color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
#     ax.plot(coords.y / const.R_sun,
#             coords.z / const.R_sun, color=color)
#
# # Add inner and outer boundary circles
# ax.add_patch(mpatch.Circle((0, 0), 1, color='k', fill=False))
# ax.add_patch(mpatch.Circle((0, 0), outflow_in.grid.rss, color='k', linestyle='--',
#                            fill=False))
#
# ax.set_title('Test solution')
#
# plt.show()







