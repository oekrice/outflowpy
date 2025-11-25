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

def find_real_angle_distribution(eclipse_year, nbins, doplots = False):
    r"""
    Produces an array of the average field line angles for a given eclipse year.
    Also outputs a mask of the areas to be ignored as they don't contain enough data, to allow for a fair comparison with the synthetic eclipses
    """
    resolution = 512
    eclipse_image_root = './data/eclipse_images/'
    img_title = f"{eclipse_image_root}{eclipse_year}_eclipse.png"

    fieldlines = edge_detection.find_decent_lines(resolution, img_title, eclipse_year)

    # for line in fieldlines:
    #     plt.scatter(line[0], line[1])
    #
    # plt.show()
    bin_means, bin_mask = edge_detection.make_angle_histogram(fieldlines, nbins)

    rbins = np.linspace(1.0, 2.5, nbins + 1)
    thetabins = np.linspace(0.0, 2*np.pi, nbins + 1)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color = 'red')
    bin_means[bin_mask < 0.5] = np.nan

    if doplots:
        R, T = np.meshgrid(rbins, thetabins)
        R, T = -R*np.sin(T), -R*np.cos(T)
        plt.pcolormesh(R, T, bin_means.T, vmin = 0, vmax = 0.6, cmap = cmap)
        plt.gca().set_axis_off()
        plt.gca().axis('equal')
        plt.title(f'Field line deviation from radial, eclipse {eclipse_year}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('./edge_detection/dist_%d.png' % (eclipse_year), dpi = 200)
        plt.close()

    radial_distribution = np.zeros(nbins)
    #Take the average of the not-nans in each dimension
    for ri in range(nbins):
        nvalues = nbins - np.isnan(bin_means[ri,:]).sum()
        if nvalues > 0:
            radial_distribution[ri] = np.nansum(bin_means[ri,:])/nvalues
        else:
            radial_distribution[ri] = np.nan

    rcs = 0.5*(rbins[1:] + rbins[:-1])

    return radial_distribution, bin_mask

#Now look into the synthetic angle distributions

def find_eclipse_flines(eclipse_year):
    field_root = f"./data/output_{eclipse_year}"

    nrho = 60
    rss = 5.0
    ns = 60
    nphi = 120

    corona_temp = 1.5e6
    mf_constant = 5e-17
    nseeds = 100

    image_extent = 2.5
    image_resolution = 512

    obs_time = outflowpy.utils.find_eclipse_time(eclipse_year)

    # br = np.loadtxt('./data/hmi_2210_smooth.txt')
    #
    # header = outflowpy.utils.carr_cea_wcs_header(Time('2020-1-1'), br.shape)
    # input_map = sunpy.map.Map((br.T, header))
    #
    # outflow_in = outflowpy.Input(input_map, nrho, rss, mf_constant = 0.0)

    input_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = True)   #Outputs the set of data corresponding to this particular Carrington rotation.

    outflow_in = outflowpy.Input(input_map, nrho, rss, corona_temp = corona_temp, mf_constant = mf_constant)

    outflow_out = outflowpy.outflow_fortran(outflow_in)#, existing_fname = field_root)

    # np.save(f'{field_root}_br.npy', np.swapaxes(outflow_out.br, 0, 2))
    # np.save(f'{field_root}_bs.npy', np.swapaxes(outflow_out.bs, 0, 2))
    # np.save(f'{field_root}_bp.npy', np.swapaxes(outflow_out.bp, 0, 2))

    seeds = outflowpy.utils.plane_seed_sampler(outflow_out, nseeds, 0.0, rss)

    tracer = outflowpy.tracing.FastTracer()

    field_lines = tracer.trace(seeds, outflow_out, save_flag = True)

    transformed_lines = []

    for fi, fline in enumerate(field_lines):
        coords = fline.coords
        coords.representation_type = 'cartesian'

        line = np.zeros((2, len(coords)))
        line[0,:] = coords.y/const.R_sun; line[1,:] = coords.z/const.R_sun

        transformed_lines.append([line[0,:], line[1,:]])

        #For every point along the line, log the position (do angle from the top, clockwise?. Maybe just arctan2 is best) and the angle
        for i in range(1,len(line[0])-1):

            x = line[0][i]; y = line[1][i]
            angle = np.arctan2(np.abs(y), np.abs(x))
            dx = line[0][i+1] - line[0][i-1]
            dy = line[1][i+1] - line[1][i-1]
            #Make sure that the angle is always less than pi/2, as it doesn't matter which direction the line was traced.
            dangle = np.arctan2(np.abs(dy), np.abs(dx)) #This is the direction. Which could be off by pi/2, I suppose.
            #Let's establish a precedent. All ys are +ve, and all xs are +ve
            radial_difference = np.abs(dangle - angle)

    return transformed_lines

def find_synthetic_angle_distribution(eclipse_year, nbins, doplots = False):
    r"""
    Calculates the outflow field and saves out the field lines in a nice format
    """

    fieldlines = find_eclipse_flines(eclipse_year)

    # for line in fieldlines:
    #     plt.scatter(line[0], line[1])
    #
    # plt.show()
    bin_means, bin_mask = edge_detection.make_angle_histogram(fieldlines, nbins)

    bin_means[bin_mask < 0.5] = np.nan

    if doplots:
        rbins = np.linspace(1.0, 2.5, nbins + 1)
        thetabins = np.linspace(0.0, 2*np.pi, nbins + 1)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color = 'red')
        R, T = np.meshgrid(rbins, thetabins)
        R, T = -R*np.sin(T), -R*np.cos(T)
        plt.pcolormesh(R, T, bin_means.T, vmin = 0, vmax = 0.6, cmap = cmap)
        plt.gca().set_axis_off()
        plt.gca().axis('equal')
        plt.title(f'Field line deviation from radial, outflow field {eclipse_year}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('./edge_detection/synth_dist_%d.png' % (eclipse_year), dpi = 200)
        plt.close()

    return bin_means

#find_synthetic_angle_distribution(2017, 30)

def compare_angles(year):
    cmap = plt.cm.tab20

    nbins = 30
    years = [year]
    real_dists = []; synth_dists = []
    for year in years:
        real_distribution, bin_mask = find_real_angle_distribution(year, nbins)
        synthetic_means = find_synthetic_angle_distribution(year, nbins)

        synth_distribution = np.zeros(nbins)
        #Take the average of the not-nans in each dimension
        for ri in range(nbins):
            nvalues = nbins - np.isnan(bin_mask[ri,:]).sum()
            if nvalues > 0:
                synth_distribution[ri] = np.nansum(synthetic_means[ri,:])/nvalues
            else:
                synth_distribution[ri] = np.nan

        synth_dists.append(synth_distribution)
        real_dists.append(real_distribution)
    rbins = np.linspace(1.0, 2.5, 31)
    rcs = 0.5*(rbins[1:] + rbins[:-1])
    fig = plt.figure(figsize = (8,5))
    for i in range(len(real_dists)):
        plt.plot(rcs[1:], real_dists[i][1:], label = f"{years[i]} reference", linestyle = 'solid', c = cmap(i))
        plt.plot(rcs[1:], synth_dists[i][1:], label = f"{years[i]} outflow", linestyle = 'dashed', c = cmap(i))
    #plt.ylim(ymin = 0.0)
    plt.xlabel('Radius')
    plt.ylabel('Avg. deviation from radial')
    plt.legend()
    plt.savefig(f'./edge_detection/angles_{year}.png')
    plt.close()

    error = np.mean((real_dists[i][1:] - synth_dists[i][1:])**2)

    print(error)
    return error

years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]
years = [2019]

for year in years:
    compare_angles(year)
