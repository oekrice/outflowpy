#This script is for the comparison of field line shapes, with the ability to generate new fields as it goes, one hopes.
#As the real data is quite messy I think taking the average values in radial and poloidal bins BUT blocking some of them out is probably a good idea.
#That should provide a nicer comparison between the nice eclipses and the nasty eclipses.
#Need to produce a new field line seed distributor too, as just in-plane isn't really up to the job
import edge_detection
import matplotlib.pyplot as plt
import numpy as np
import outflowpy
import astropy.constants as const
from astropy.time import Time
import sunpy

#Now look into the synthetic angle distributions

def find_open_flux(eclipse_year, rss = 2.5):
    field_root = f"./data/output_{eclipse_year}"

    nrho = 120
    rss = rss
    ns = 180
    nphi = 360

    nseeds = 2000

    image_extent = 2.5
    image_resolution = 512

    obs_time = outflowpy.utils.find_eclipse_time(eclipse_year)

    year_options = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]
    poly_values = [0.0,0.0,0.0,0.0,0.0]

    allpolys = np.load(f"batch_logs/optimums.npy")

    #Load the correct polynomial coefficients.
    eclipse_index = year_options.index(eclipse_year)
    mf_constant = np.abs(allpolys[eclipse_index,0])*1e-17
    corona_temp = np.abs(allpolys[eclipse_index,1])*1e6

    print('Individual eclipse parameters', mf_constant, corona_temp)
    poly_values = allpolys[eclipse_index]

    input_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = True)   #Outputs the set of data corresponding to this particular Carrington rotation.

    #outflow_in = outflowpy.Input(input_map, nrho, rss, polynomial_coeffs = poly_values, polynomial_type = source)
    outflow_in = outflowpy.Input(input_map, nrho, rss, mf_constant = mf_constant, corona_temp = corona_temp)

    outflow_out = outflowpy.outflow_fortran(outflow_in)#, existing_fname = field_root)

    # np.save(f'{field_root}_br.npy', np.swapaxes(outflow_out.br, 0, 2))
    # np.save(f'{field_root}_bs.npy', np.swapaxes(outflow_out.bs, 0, 2))
    # np.save(f'{field_root}_bp.npy', np.swapaxes(outflow_out.bp, 0, 2))
    rsun_cm = 6.957e10
    surface_area = 4*np.pi*(np.exp(outflow_in.grid.rg[-1])*rsun_cm)**2
    openflux = np.sum(np.abs(outflow_out.br)[:,:,-1])*surface_area/(nphi*ns)

    return openflux

#find_synthetic_angle_distribution(2017, 30)

def compare_angles(year):

    open_flux = find_open_flux(eclipse_year, optimised = optimised, rss = rss)

    return

years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]

allpolys = []

with open(f"batch_logs/optimums.txt") as f:
    for i, line in enumerate(f):
        poly_string = line.strip()
        poly_values = [float(x) for x in poly_string[1:-1].split(",")]
        allpolys.append(poly_values)

allpolys = np.array(allpolys)
np.save(f"batch_logs/optimums.npy", allpolys)

openfluxes = []
for year in years:
    openfluxes.append(find_open_flux(year, rss = 5.0))

np.savetxt('./data/eclipse_ofluxes.txt', np.array(openfluxes), delimiter = ',')
