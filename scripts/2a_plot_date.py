import astropy.constants as const
import outflowpy
from outflowpy.plotting import plot_pyvista
import numpy as np
import astropy.units as u
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.time import Time
import sunpy
import random
import sys

max_rss = 5

rss_values = [2.5]
model_options = ["PFSS", "Outflow"]

model = 1

for plot_count, rss in enumerate(rss_values):

    corona_temp = 2e6
    if model == 0:
        mf_constant = 0.0#1e-16
    else:
        mf_constant = 5e-17

    nphi = 180
    ns = 90

    phi = np.linspace(0, 2 * np.pi, nphi)
    s = np.linspace(-1, 1, ns)
    s, phi = np.meshgrid(s, phi)

    def dipole_Br(r, s):
        return s**7#2 * s / r**3

    br = dipole_Br(1, s)

    br[:45,:] = -br[:45,:]
    nrho = 30
    rss = 2.5

    header = outflowpy.utils.carr_cea_wcs_header(Time('2020-1-1'), br.shape)
    input_map = sunpy.map.Map((br.T, header))

    obs_time = "2008-08-01T00:00:00"
    nrho = 60
    ns = 90
    nphi = 180
    rss = 5.0

    print('Run parameters', obs_time, ns, nphi,1.0*5e-2/nphi, corona_temp, mf_constant)
    #br = np.loadtxt("./data/dipole_smooth.txt")
    # br = np.loadtxt("./data/hmi_2210_smooth.txt")
    #br = np.loadtxt("./data/mdi_2000_smooth.txt")
    #
    #
    # header = outflowpy.utils.carr_cea_wcs_header(Time('2020-1-1'), br.T.shape)
    # input_map = sunpy.map.Map((br, header))

    input_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = False)   #Outputs the set of data corresponding to this particular Carrington rotation.

    sys.exit()

    outflow_in = outflowpy.Input(input_map, nrho, rss, corona_temp = corona_temp, mf_constant = mf_constant)
    outflow_out = outflowpy.outflow_fortran(outflow_in)

    def _coord_to_cart(r, lon, lat):
        #Returns a list of cartesian positions (to be read into the SkyCoord object) based on these coordinates
        return np.array([r*np.sin(lon)*np.sin(lat),r*np.cos(lon)*np.cos(lat),r*np.cos(lat)]).T

    # Start points from near the source surface
    r = const.R_sun*rss*0.9
    lon = np.pi / 2 * u.rad
    lat = np.linspace(-np.pi / 2, np.pi / 2, 5) * u.rad
    r = np.ones(len(lat))*r
    lon = np.ones(len(lat))*lon

    lon_all = np.hstack((lon, -lon))
    lat_all = np.hstack((lat, lat))
    r_all = np.hstack((r, r))

    #Start points from the lower surface
    r = const.R_sun*1.05
    lons, lats, rs = [], [], []

    for seed in range(10000):
        lons.append(random.uniform(0,2*np.pi) * u.rad)
        lats.append(random.uniform(-np.pi/2,np.pi/2) * u.rad)
        rs.append(random.uniform(1.0,2.5))

    lon = np.array(lon)
    lat = np.array(lat)
    rs = np.array(rs) * r

    seeds = SkyCoord(lons,lats,rs, frame=outflow_out.coordinate_frame)   #This can take three arrays (of the same length) for all the coordinates.

    tracing_options = ['Fast', 'Python', 'Fortran']

    for option in range(1):
        if option == 0:
            tracer = outflowpy.tracing.FastTracer()
        elif option == 1:
            tracer = outflowpy.tracing.PythonTracer()
        else:
            tracer = outflowpy.tracing.FortranTracer()
        field_lines = tracer.trace(seeds, outflow_out)

        if False:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

            for field_line in field_lines:
                coords = field_line.coords
                coords.representation_type = 'cartesian'
                color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
                linewidth = {0: 0.5, -1: 1.0, 1: 1.0}.get(field_line.polarity)
                #Filter out open field lines which aren't in the plane of sky (messy)
                ax.plot(coords.y / const.R_sun,
                        coords.z / const.R_sun, color=color, linewidth = linewidth)

            # Add inner and outer boundary circles
            ax.add_patch(mpatch.Circle((0, 0), 1, color='k', fill=False))
            ax.add_patch(mpatch.Circle((0, 0), outflow_in.grid.rss, color='k', linestyle='--',
                                    fill=False))
            ax.set_xlim(-max_rss, max_rss)
            ax.set_ylim(-max_rss, max_rss)
            ax.set_title(f"{model_options[model]} solution, rss = {rss:.1f}")
            ax.set_axis_off()
            plt.savefig(f'./plots/eclipse_08_{model_options[model]}_{plot_count:03d}.png')
            plt.close()


        else:
            outflowpy.plotting.plot_pyvista(outflow_out, field_lines)








