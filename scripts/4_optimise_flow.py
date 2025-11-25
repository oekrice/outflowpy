#This script for optimising outflow speed parameters, in the same manner as the image optimisation
#Based on the one I wrote a few weeks ago but considerably neater and more scientific.

import outflowpy
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import random, time
from datetime import datetime
import cma
import sys, subprocess, time
import edge_detection
import astropy.constants as const

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

def find_eclipse_flines(eclipse_year, field_parameters):

    nrho = 90
    rss = 5.0
    ns = 90
    nphi = 180

    nseeds = 2000

    image_extent = 2.5
    image_resolution = 512

    obs_time = outflowpy.utils.find_eclipse_time(eclipse_year)

    input_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = True)   #Outputs the set of data corresponding to this particular Carrington rotation.

    outflow_in = outflowpy.Input(input_map, nrho, rss, polynomial_coeffs = field_parameters)

    print('Maximum outflow speed', np.max(outflow_in.vg))

    outflow_out = outflowpy.outflow_fortran(outflow_in) #This is where parameters can live.

    seeds = outflowpy.utils.plane_seed_sampler(outflow_out, nseeds, 0.0, rss)

    seeds = outflowpy.utils.load_sampled_seeds(outflow_out, nseeds)

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
            dangle = np.arctan2(np.abs(dy), np.abs(dx)) #This is the direction. Which could be off by pi/2, I suppose.
            radial_difference = np.abs(dangle - angle)

    return transformed_lines

def find_error_fn(coeffs, run_id, eclipse_year):

    nbins = 30

    reference_distribution, bin_mask = find_real_angle_distribution(eclipse_year, nbins, doplots = False)
    fieldlines = find_eclipse_flines(eclipse_year, field_parameters = coeffs)
    synthetic_means, _ = edge_detection.make_angle_histogram(fieldlines, nbins)
    synthetic_means[bin_mask < 0.5] = np.nan

    if False:
        rbins = np.linspace(1.0, 2.5, nbins + 1)
        thetabins = np.linspace(0.0, 2*np.pi, nbins + 1)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color = 'red')
        R, T = np.meshgrid(rbins, thetabins)
        R, T = -R*np.sin(T), -R*np.cos(T)
        plt.pcolormesh(R, T, synthetic_means.T, vmin = 0, vmax = 0.6, cmap = cmap)
        plt.gca().set_axis_off()
        plt.gca().axis('equal')
        plt.title(f'Eclipse {eclipse_year}, run {run_id}')

        plt.colorbar()
        plt.tight_layout()
        plt.savefig('./edge_detection/synth_dist_%d.png' % (run_id), dpi = 200)
        plt.close()

    synth_distribution = np.zeros(nbins)
    #Take the average of the not-nans in each dimension
    for ri in range(nbins):
        nvalues = nbins - np.isnan(bin_mask[ri,:]).sum()
        if nvalues > 0:
            synth_distribution[ri] = np.nansum(synthetic_means[ri,:])/nvalues
        else:
            synth_distribution[ri] = np.nan

    plt.plot(reference_distribution)
    plt.plot(synth_distribution)
    plt.close()

    error = np.mean((reference_distribution[1:] - synth_distribution[1:])**2)

    return error

def generate_fn(parameter_set):

    #Obtain eclipse year from system variables

    year_options = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]
    eclipse_number = int(sys.argv[1])

    eclipse_year = year_options[eclipse_number]

    print(f'Doing a run with  {eclipse_year} eclipse')
    run_id = 0
    #Do the logging and things for the error functions
    if os.path.exists("batch_logs/log_%d.txt" % eclipse_year):
        with open("batch_logs/log_%d.txt" % eclipse_year, "r") as f:
            for line in f.readlines():
                run_id += 1

    error_function = find_error_fn(parameter_set, run_id, eclipse_year)

    #Do the global scaling and stuff here
    save_line = [run_id, error_function] + parameter_set.tolist()
    with open("batch_logs/log_%d.txt" % eclipse_year, "a") as f:
        f.write(" ".join(f"{x:.3f}" for x in save_line) + "\n")

    print('Error fn', error_function)
    return error_function

if len(sys.argv) > 1:
    eclipse_number = int(sys.argv[1])
else:
    raise Exception('Specify eclipse number.')

def run_optimisation():

    year_options = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]
    eclipse_number = int(sys.argv[1])

    eclipse_year = year_options[eclipse_number]

    print('Doing optimisation run on eclipse', eclipse_year)
    if os.path.exists("batch_logs/log_%d.txt" % eclipse_year):
        os.remove("batch_logs/log_%d.txt" % eclipse_year)
    initial_parameter_set = np.array([0.0,0.0,0.0,0.0,0.0])
    es = cma.CMAEvolutionStrategy(initial_parameter_set, 0.5, {'verb_disp': 1})
    es.optimize(generate_fn)
    es.result_pretty()

run_optimisation()
# for i in range(100):
#     generate_fn(np.array([0.0,0.0,0.0,0.0,0.0]))

