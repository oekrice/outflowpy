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

def make_image(parameter_set, image_number):
    """
    Runs the outflow code, makes an image and saves it out. All parameters except for the 6 variables are hard-coded.

    Parameter set effects:
    0: Brightness of field lines due to magnetic field strength at an individual point.
    1: Brightness of field lines due to the magnetic field strength where the line meets the solar surface. Always positive
    2: Weighting based on the maximum height of the field line. Allowed to skew in either direction.
    3: Gaussian blurring factor. Always positive.
    4: Percentile clip for saturating the image. Always positive.
    5: Alters the skew with which the radial field line seeds are chosen. Sign doesn't matter as will always want to skew towards lower altitudes
    """
    br = np.loadtxt("./data/hmi_2210_smooth.txt")

    nrho = 60
    ns = 90
    nphi = 180
    rss = 5.0

    corona_temp = 2e6
    mf_constant = 5e-17
    nseeds = 10000

    image_extent = 2.5
    image_resolution = 512

    header = outflowpy.utils.carr_cea_wcs_header(Time('2020-1-1'), br.T.shape)
    input_map = sunpy.map.Map((br, header))

    outflow_in = outflowpy.Input(input_map, nrho, rss, corona_temp = corona_temp, mf_constant = mf_constant)
    outflow_out = outflowpy.outflow_fortran(outflow_in)

    seeds = outflowpy.utils.random_seed_sampler(outflow_out, nseeds, parameter_set[5], rss)

    tracer = outflowpy.tracing.FastTracer()

    field_lines, image_matrix = tracer.trace(seeds, outflow_out, parameters = parameter_set, image_extent = image_extent, generate_image = True, save_flag = False, image_resolution = image_resolution)

    outflowpy.plotting.make_image(image_matrix, image_extent, parameter_set, f'./img_plots/{image_number:04d}.png')

def compare_image(image_id, eclipse_year):
    resolution = 512
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')   #Image similarity thingummy. Whatever this is, it is large...
    transform = transforms.Compose([transforms.Resize((resolution, resolution), interpolation = transforms.InterpolationMode.BICUBIC),transforms.PILToTensor()])

    def convert_tensor(tensor):
        #Swap axes first, see if that works. Not sure where the 4 is from though...
        tensor = tensor[np.newaxis,:,:,:]
        tensor = 2*((tensor/(resolution - 1.)) - 0.5)
        return tensor

    #Transform this into a tensor which Torch can use
    #If want to retain the colourmap, use 'RGB' as the convert. For greyscale I'm pretty sure it's 'L'
    img_root = './data/eclipse_images/'
    sim_sum = 0; all_sims = []
    img_name = f'{eclipse_year}_eclipse.png'
    fname = img_root + img_name

    synthetic_root = './img_plots/'

    print('%s%04d.png' % (synthetic_root, image_id))
    print(fname)
    img1 = Image.open('%s%04d.png' % (synthetic_root, image_id)).convert("RGB")   #Synthetic one
    img2 = Image.open(fname).convert("RGB")  #Real one

    tensor1 = transform(img1)
    tensor2 = transform(img2)

    tensor1 = convert_tensor(tensor1)
    tensor2 = convert_tensor(tensor2)

    sim = lpips(tensor1, tensor2).item()
    print('Similarity to eclipse', eclipse_year, ':', sim)
    return sim

def generate_and_compare(parameter_set):

    #Assuming all other parameters are the same, will run the code and generate an image with the above parameter set.
    #Use code from runbatch?

    run_id = 0
    if os.path.exists("img_plots/log.txt"):
        with open("img_plots/log.txt", "r") as f:
            for line in f.readlines():
                run_id += 1

    make_image(parameter_set, run_id)
    similarity = compare_image(run_id, 2008)

    save_line = [run_id, similarity] + parameter_set.tolist()
    with open("img_plots/log.txt", "a") as f:
        f.write(" ".join(f"{x:.3f}" for x in save_line) + "\n")

    return similarity

if True:
    for i in range(6):
        parameter_set = np.zeros(100)
        parameter_set[i] = 1.0
        #parameter_set[:6] = [-0.004,-0.444,-0.298,0.699,-1.265,0.0]
        similarity = generate_and_compare(parameter_set)

def run_optimisation():
    print('Running optimisation run')
    if os.path.exists("img_plots/log.txt"):
        os.remove("img_plots/log.txt")
    initial_parameter_set = np.zeros(6)
    es = cma.CMAEvolutionStrategy(initial_parameter_set, 0.5, {'verb_disp': 1})
    es.optimize(generate_and_compare)
    es.result_pretty()

#run_optimisation()


