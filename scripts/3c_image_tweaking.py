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

    field_root = "./data/output_08"
    br = np.loadtxt("./data/2008_input.txt")

    nrho = 60
    rss = 5.0

    corona_temp = 2e6
    mf_constant = 5e-17
    nseeds = 250000

    image_extent = 2.5
    image_resolution = 512

    header = outflowpy.utils.carr_cea_wcs_header(Time('2020-1-1'), br.T.shape)
    input_map = sunpy.map.Map((br, header))

    outflow_in = outflowpy.Input(input_map, nrho, rss, corona_temp = corona_temp, mf_constant = mf_constant)

    outflow_out = outflowpy.outflow_fortran(outflow_in, existing_fname = field_root)

    # np.save(f'{field_root}_br.npy', np.swapaxes(outflow_out.br, 0, 2))
    # np.save(f'{field_root}_bs.npy', np.swapaxes(outflow_out.bs, 0, 2))
    # np.save(f'{field_root}_bp.npy', np.swapaxes(outflow_out.bp, 0, 2))

    seeds = outflowpy.utils.random_seed_sampler(outflow_out, nseeds, parameter_set[5], rss)

    tracer = outflowpy.tracing.FastTracer()

    field_lines, image_matrix = tracer.trace(seeds, outflow_out, parameters = parameter_set, image_extent = image_extent, generate_image = True, save_flag = False, image_resolution = image_resolution)

    #outflowpy.plotting.plot_pyvista(outflow_out, field_lines)

    #outflowpy.plotting.make_image(image_matrix, image_extent, parameter_set, f'./img_plots/{image_number:04d}.png')

    return image_matrix

def scale_image(image_matrix, reference_fname, image_extent, crefs = np.linspace(0.0,255.0,20)):
    """
    This is an experiment to scale the pixel brightness of the generated image to match that of the target comparison one. Not sure yet how well (or at all) that will work...
    Only want to compare pixels which lie outside the picture of the moon.
    """
    #Find distribution of these pixels (can show in histogram form)
    reference_dist = np.zeros(256)
    synthetic_dist = np.zeros(256)

    image_matrix = image_matrix*255/np.max(image_matrix)
    img_ref = Image.open(reference_fname).convert("L")  #Real image in greyscale
    img_colours = Image.open(reference_fname).convert("RGB")

    reference_matrix = np.zeros((img_ref.size))
    synthetic_matrix = np.zeros((img_ref.size))
    res = np.shape(reference_matrix)[0]
    for i in range(res):
        for j in range(res):
            radius = np.sqrt((i-res//2)**2 + (j-res//2)**2)*(2*image_extent/res)
            if radius>1:
                reference_dist[img_ref.getpixel((i,j))] += 1
                synthetic_dist[int(image_matrix[i,j])] += 1
            reference_matrix[i,j] = img_ref.getpixel((i,j))
            synthetic_matrix[i,j] = int(image_matrix[i,j])
    #Now need to somehow map percentiles for these data. Hmm. Requires paper, and percentiles. I think I have a way to do it though...
    scaled_matrix = np.zeros((img_ref.size))
    raw_sums = np.cumsum(reference_dist)/np.sum(reference_dist)
    for value in range(256):
        #Find number of cells with this value OR less
        nbelow = np.sum(synthetic_dist[:value])/np.sum(synthetic_dist)
        #This is essentially the percentile of the current pixel value
        #Target is the equivalent in the reference dist.
        target = np.searchsorted(raw_sums, nbelow)
        #Find number of cells matching this exactly and scale appropriately
        scaled_matrix[synthetic_matrix == value] = int(target)

    hex_refs = []; hex_values = []
    #Obtain colour codes for the cmap
    hex_values.append((0.0, '#000000ff'))

    for c_value in crefs:
        #Get list of indices to check
        i_values, j_values = np.where((reference_matrix > c_value - 5)*(reference_matrix < c_value + 5))
        avg_colour = np.zeros(3)
        if len(i_values) > 10:
            for pix in range(len(i_values)):
                i = i_values[pix]; j = j_values[pix]
                colour = [img_colours.getpixel((i, j))[0], img_colours.getpixel((i, j))[1], img_colours.getpixel((i, j))[2]]
                avg_colour += colour
            avg_colour = avg_colour/len(i_values)
            print(c_value, avg_colour)
            hex_values.append((c_value/255.0, '#%02x%02x%02xff' % (int(avg_colour[0]), int(avg_colour[1]), int(avg_colour[2]))))
    hex_values.append((1.0, '#ffffffff'))

    return scaled_matrix, hex_values

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

parameter_set = [-0.003,0.533,-0.165,0.000,0.000,-2.625]

if False:  #Generate the image
    image_matrix = make_image(parameter_set, 0)
    np.save('./data/img_data/test1.npy', image_matrix)

image_matrix = np.load('./data/img_data/test1.npy')
image_extent = 2.5

scaled_matrix, hex_values = scale_image(image_matrix,'./data/eclipse_images/2008_eclipse.png', image_extent,)
plot_image(scaled_matrix, image_extent, parameter_set, f'./tweaked_plot.png', off_screen = False, hex_values = hex_values)
