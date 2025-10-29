"""
Writing a test script to test this all works (very basically) with everything set up to be identical/very compatible with pfsspy stuff
I think it's very important people can just change the import and things just work -- but the tests should take care of that maybe?
Will need to learn how that actually works though
"""


import os

import astropy.units as u
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
import numpy as np

import outflowpy
import outflowpy.utils

import pfsspy
import pfsspy.utils

import drms

"""
Going to run through the HMI example from the readthedocs, checking all the code and modifying as necessary. Not sure how far to go here.

I think (unlike pfsspy, I think?) I'll provide a means to create such a map without any faffing around? Don't need emails etc. the way I do it!
"""

hmi_map = outflowpy.obtain_data.download_hmi_crot(2210)   #Outputs the set of data corresponding to this particular Carrington rotation.

print('Data shape: ', hmi_map.data.shape)
    
hmi_map = hmi_map.resample([90, 60] * u.pix)   #This is the default PFSSpy resampling, but it looks a bit rubbish compared to Anthony's method.
print('New shape: ', hmi_map.data.shape)

#Calculate pfss field, for now. So I don't have to write any code to do the basic testing
nrho = 30
rss = 2.5



pfss_in = pfsspy.Input(hmi_map, nrho, rss)
pfss_out = pfsspy.pfss(pfss_in)
ss_br = pfss_out.source_surface_br

# outflow_in = outflowpy.Input(hmi_map, nrho, rss, corona_temp = 2e6, mf_constant = 0.0)
# outflow_out = outflowpy.outflow(outflow_in)
# outflow_br = outflow_out.source_surface_br

# Create the figure and axes
fig = plt.figure()
ax = plt.subplot(projection=ss_br)
# Plot the source surface map
ss_br.plot()
# Plot formatting
plt.colorbar()
ax.set_title('Source surface magnetic field')

plt.show()