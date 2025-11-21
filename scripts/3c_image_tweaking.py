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
from scipy.stats import qmc
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image
import torchvision.transforms as transforms

import os
import cma
import matplotlib

"""
Ideal parameters for given number of field lines:
10,000 : [-0.203,0.587,-1.063,0.002,-0.497,-0.831]
50,000 : [-0.135,0.764,-0.667,-0.003,-1.38,-1.359]
250,000: [-0.003,0.533,-0.165,0.009,-2.314,-2.625]
"""

def make_image(parameter_set, image_number):
    """
    Runs the outflow code, makes an image and saves it out. All parameters except for the 6 variables are hard-coded.

    Parameter set effects:
    0: Brightness of field lines due to magnetic field strength at an individual point.
    1: Brightness of field lines due to the magnetic field strength where the line meets the solar surface. Always positive
    2: Weighting based on the maximum height of the field line. Allowed to skew in either direction.
    3: Gaussian blurring factor. Always positive.
    4: Percentile clip for saturating the image. Always positive.
    5: Alters the skew with which the radial field line seeds are chosen. Sign doesn't matter as will always want to skew towards lower altitudes.
    """

    field_root = f"./data/output_{eclipse_year}"

    nrho = 60
    rss = 5.0
    ns = 180
    nphi = 360

    corona_temp = 1.5e6
    mf_constant = 5e-17
    nseeds = 50000

    image_extent = 2.5
    image_resolution = 512

    obs_time = outflowpy.utils.find_eclipse_time(eclipse_year)

    input_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = True)   #Outputs the set of data corresponding to this particular Carrington rotation.

    outflow_in = outflowpy.Input(input_map, nrho, rss, corona_temp = corona_temp, mf_constant = mf_constant)

    outflow_out = outflowpy.outflow_fortran(outflow_in, existing_fname = field_root)

    np.save(f'{field_root}_br.npy', np.swapaxes(outflow_out.br, 0, 2))
    np.save(f'{field_root}_bs.npy', np.swapaxes(outflow_out.bs, 0, 2))
    np.save(f'{field_root}_bp.npy', np.swapaxes(outflow_out.bp, 0, 2))

    seeds = outflowpy.utils.random_seed_sampler(outflow_out, nseeds, parameter_set[3], rss)

    tracer = outflowpy.tracing.FastTracer(step_size = 0.25)

    field_lines, image_matrix = tracer.trace(seeds, outflow_out, parameters = parameter_set, image_extent = image_extent, generate_image = True, save_flag = False, image_resolution = image_resolution)

    #outflowpy.plotting.plot_pyvista(outflow_out, field_lines)

    #outflowpy.plotting.make_image(image_matrix, image_extent, parameter_set, f'./img_plots/{image_number:04d}.png')

    return image_matrix


def plot_image(image_matrix, image_extent, image_parameters, image_fname, off_screen = True, hex_values = []):

    """
    Generates an image in the style of a Druckmuller eclipse picture
    """

    npixels = np.shape(image_matrix)[0]
    dpi = 100

    image_matrix = np.flip(image_matrix, 1)

    if len(hex_values) > 0:
        cmap = LinearSegmentedColormap.from_list("eclipse", hex_values)
    else:
        cmap = LinearSegmentedColormap.from_list("eclipse", ["#3b444dff", "#dadadaff"])

    fig, ax = plt.subplots(figsize = (npixels/dpi, npixels/dpi), dpi = dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    moon_face = mpimg.imread("./data/moonface_druck.png")

    xs = np.linspace(-image_extent,image_extent,np.shape(image_matrix)[0])
    ys = np.linspace(-image_extent,image_extent,np.shape(image_matrix)[1])
    ax.imshow(image_matrix.T, cmap = cmap, extent = [-image_extent,image_extent,-image_extent,image_extent],interpolation="bilinear", vmin = 0, vmax = 255)

    moon_img = ax.imshow(moon_face, extent = [-1,1,-1,1],interpolation="bilinear")
    circle = Circle((0, 0), 0.995, transform = ax.transData)
    moon_img.set_clip_path(circle)

    ax.set_xlim(-image_extent, image_extent)
    ax.set_ylim(-image_extent, image_extent)
    ax.axis("off")
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(image_fname, bbox_inches=None, pad_inches = 0, dpi = dpi)
    if not off_screen:
        plt.show()
    plt.close()

parameter_set = [-0.023,0.42,-0.746,1.084]

eclipse_years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]
#eclipse_years = [2006]

for eclipse_year in eclipse_years:
    eclipse_fname = f'./data/eclipse_images/{eclipse_year}_eclipse.png'

    if True:  #Generate the image
        image_matrix = make_image(parameter_set, 0)
        np.save(f'./data/img_data/test{eclipse_years[0]}.npy', image_matrix)

    image_matrix = np.load(f'./data/img_data/test{eclipse_years[0]}.npy')
    image_extent = 2.5

    scaled_matrix, hex_values = outflowpy.plotting.match_image(image_matrix,eclipse_fname, image_extent)

    #scaled_matrix, hex_values = scale_image(image_matrix,'./data/eclipse_images/sun.png', image_extent)
    outflowpy.plotting.plot_image(image_matrix, image_extent, parameter_set, f'./plots/raw_{eclipse_year}.png', off_screen = True)

    outflowpy.plotting.plot_image(scaled_matrix, image_extent, parameter_set, f'./plots/tweaked_{eclipse_year}.png', off_screen = True, hex_values = hex_values)
