#This script is for the comparison of field line shapes, with the ability to generate new fields as it goes, one hopes.
#As the real data is quite messy I think taking the average values in radial and poloidal bins BUT blocking some of them out is probably a good idea.
#That should provide a nicer comparison between the nice eclipses and the nasty eclipses.
#Need to produce a new field line seed distributor too, as just in-plane isn't really up to the job
import matplotlib.pyplot as plt
import numpy as np
import outflowpy
import astropy.constants as const
from astropy.time import Time
import sunpy
import seaborn as sns
import cv2 as cv
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
from matplotlib.patches import Circle

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # LaTeX-like Computer Modern
    "font.family": "serif",
})

colors = sns.color_palette('colorblind')

#cmap = sns.color_palette("magma", as_cmap=True)
#plt.rcParams['axes.prop_cycle'] = cycler('linestyle', ['-', '--', '-.', ':'])


def generate_eclipse_image(eclipse_year, optimised = True, rss = 5.0, match_flux = False):

    nrho = 60
    rss = rss
    ns = 60
    nphi = 120

    corona_temp = 1.5e6
    mf_constant = 5e-17
    nseeds = 25000

    image_extent = 2.5
    image_resolution = 512

    obs_time = outflowpy.utils.find_eclipse_time(eclipse_year)
    #0.633,-0.006,-4.299,-2.38
    #parameter_set = [0.216,0.377,-0.323,1.567]
    #parameter_set = [0.633,-0.006,-4.299,-2.38]
    parameter_set = [-0.195,0.279,-2.986,-1.759]

    year_options = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]
    poly_values = [0.0,0.0,0.0,0.0,0.0]

    allpolys = np.load(f"./data/batch_logs/optimums.npy")

    eclipse_index = year_options.index(eclipse_year)

    input_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = True)   #Outputs the set of data corresponding to this particular Carrington rotation.

    year_index = year_options.index(eclipse_year)

    if match_flux:
        mf_constant = 5e-17
        corona_temp = 2.718*1e6
        print('Parameters', mf_constant, corona_temp)


    elif optimised:
        mf_constant = allpolys[year_index][0]*1e-17
        corona_temp = allpolys[year_index][1]*1e6
        print('Parameters', mf_constant, corona_temp)

    else:
        mf_constant = 0.0
        corona_temp = 1.5e6

    #outflow_in = outflowpy.Input(input_map, nrho, rss, polynomial_coeffs = poly_values, polynomial_type = source)
    outflow_in = outflowpy.Input(input_map, nrho, rss, mf_constant = mf_constant, corona_temp = corona_temp)

    outflow_out = outflowpy.outflow_fortran(outflow_in)#, existing_fname = field_root)

    # np.save(f'{field_root}_br.npy', np.swapaxes(outflow_out.br, 0, 2))
    # np.save(f'{field_root}_bs.npy', np.swapaxes(outflow_out.bs, 0, 2))
    # np.save(f'{field_root}_bp.npy', np.swapaxes(outflow_out.bp, 0, 2))

    #seeds = outflowpy.utils.plane_seed_sampler(outflow_out, nseeds, 0.0, rss)
    seeds = outflowpy.utils.random_seed_sampler(outflow_out, nseeds, parameter_set[3], rss)

    tracer = outflowpy.tracing.FastTracer(step_size = 0.25)

    field_lines, image_matrix = tracer.trace(seeds, outflow_out, parameters = parameter_set, image_extent = image_extent, generate_image = True, save_flag = False, image_resolution = image_resolution)

    npixels = np.shape(image_matrix)[0]
    dpi = 100

    image_matrix = np.flip(image_matrix, 1)

    eclipse_fname = f'./data/eclipse_images/{eclipse_year}_eclipse.png'

    image_matrix, hex_values = outflowpy.plotting.match_image(image_matrix, image_extent, reference_image = eclipse_fname)

    if len(hex_values) > 0:
        cmap = LinearSegmentedColormap.from_list("eclipse", hex_values)
    else:
        cmap = LinearSegmentedColormap.from_list("eclipse", ["#3b444dff", "#dadadaff"])

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

    if match_flux:
        ax.set_title('Outflow - Matched Open Flux')
    elif optimised:
        ax.set_title('Outflow - Matched Topology')
    else:
        ax.set_title('PFSS Field')

    #plt.savefig(image_fname, bbox_inches=None, pad_inches = 0, dpi = dpi)
    # if not off_screen:
    #     plt.show()
    # plt.close()

fig, axs = plt.subplots(2,2, figsize = (6.9, 6.9))

ax = axs[1,1]
generate_eclipse_image(2017, optimised = True, rss = 5.0, match_flux = True)

ax = axs[1,0]
generate_eclipse_image(2017, optimised = True, rss = 5.0)

ax = axs[0,1]
generate_eclipse_image(2017, optimised = False, rss = 5.0)

ax = axs[0,0]
moon_face = mpimg.imread('./data/eclipse_images/2017_eclipse.png')
ax.imshow(moon_face, extent = [-2.5,2.5,-2.5,2.5],interpolation="bilinear")
ax.set_xlim(-2.5,2.5)
ax.set_ylim(-2.5,2.5)
ax.axis("off")
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Reference image')

plt.tight_layout()

plt.savefig('./12_nice_image.png', dpi = 700)
plt.show()


