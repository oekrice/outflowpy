import pyvista as pv
import os, sys
import numpy as np
import astropy.constants as const
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

def match_image(image_matrix, image_extent, reference_image = None, crefs = np.linspace(0.0,255.0,20), overwrite_reference = False):
    """
    This is an experiment to scale the pixel brightness of the generated image to match that of the target comparison one. Not sure yet how well (or at all) that will work...
    Only want to compare pixels which lie outside the picture of the moon.
    Now has been edited so this should work with multiple images, but that comes with some problems
    """

    #Need to add some kind of cutoff so that the modal value is reasonable. Currently everything is getting squished right into the first few pixels for some of the plots.
    #This is skewed because the chance of a really high value is actually not that unlikely. But it does need to be dealt with nicely...
    #Perhaps base it on modal values of the targets?

    image_matrix = gaussian_filter(image_matrix, sigma = 0.5)

    target_res = image_matrix.shape[0]

    synthetic_matrix = np.zeros((target_res, target_res))
    synthetic_dist = np.zeros(256)
    image_matrix = np.log(image_matrix + 1.0)  #All these should be positive by adding the 1? That seems wise
    image_matrix = image_matrix*255.0/np.max(image_matrix)
    image_matrix = np.clip(image_matrix, 0.0, 255.0)  #To stop the interpolator complaining

    if reference_image is None:
        reference_dist = [np.load('./data/img_data/reference_distribution.npy')]

    else:
        #Find distribution of these pixels (can show in histogram form)
        reference_dist = np.zeros(256)

        if not isinstance(reference_image, list):
            reference_image = [reference_image]
        img_refs = []; img_colours = []
        for img_num in range(len(reference_image)):
            img_refs.append(Image.open(reference_image[img_num]).convert("L"))  #Real image in greyscale)
            img_colours.append(Image.open(reference_image[img_num]).convert("RGB"))

        reference_res = img_refs[0].size[0]
        reference_matrix = np.zeros((len(reference_image), reference_res, reference_res))

        for i in range(reference_res):
            for j in range(reference_res):
                radius = np.sqrt((i-reference_res//2)**2 + (j-reference_res//2)**2)*(2*image_extent/reference_res)
                if radius>1:
                    for img_num in range(len(reference_image)):
                        reference_dist[img_refs[img_num].getpixel((i,j))] += 1
                for img_num in range(len(reference_image)):
                    reference_matrix[img_num, i,j] = img_refs[img_num].getpixel((i,j))

        if overwrite_reference:
            np.save('./data/img_data/reference_distribution.npy', reference_dist)

    for i in range(target_res):
        for j in range(target_res):
            radius = np.sqrt((i-target_res//2)**2 + (j-target_res//2)**2)*(2*image_extent/target_res)
            if radius>1:
                synthetic_dist[int(image_matrix[i,j])] += 1
            synthetic_matrix[i,j] = image_matrix[i,j]

    #Now need to somehow map percentiles for these data
    xs = []; ys = []
    scaled_matrix = np.zeros((image_matrix.shape))
    raw_sums = np.cumsum(reference_dist)/np.sum(reference_dist)
    for value in range(256):
        #Find number of cells with this value OR less
        nbelow = np.sum(synthetic_dist[:value])/np.sum(synthetic_dist[:])
        #This is essentially the percentile of the current pixel value
        #Target is the equivalent in the reference dist.
        target = np.searchsorted(raw_sums, nbelow)
        #Find number of cells matching this exactly and scale appropriately
        xs.append(value); ys.append(int(target))
    ys[0] = ys[1] #Enforce the minimum value so you don't get black blobs

    f = interp1d(xs, ys, kind = 'linear')
    scaled_matrix = f(synthetic_matrix)

    #Turn all zeros into the minimum nonzero value, in case a field line never goes through here (which wouldn't be realistic)
    scaled_matrix[scaled_matrix == 0] = np.min(np.ma.masked_where(scaled_matrix == 0, scaled_matrix))
    hex_refs = []; hex_values = []

    if reference_image is not None:
        #Obtain colour codes for the cmap. This just uses the reference image -- nothing from the original
        hex_values.append((0.0, '#000000ff'))
        for c_value in crefs:
            #Get list of indices to check
            allavg_colour = np.zeros(3)
            allavg_count = 0
            for img_num in range(len(reference_image)):
                i_values, j_values = np.where((reference_matrix[img_num] > c_value - 5)*(reference_matrix[img_num]  < c_value + 5))
                avg_colour = np.zeros(3)
                if len(i_values) > 10:
                    for pix in range(len(i_values)):
                        i = i_values[pix]; j = j_values[pix]
                        colour = [img_colours[0].getpixel((i, j))[0], img_colours[0].getpixel((i, j))[1], img_colours[0].getpixel((i, j))[2]]
                        avg_colour += colour
                    avg_colour = avg_colour/len(i_values)
                    allavg_count += 1
                    #print(img_num, avg_colour)
                    allavg_colour = allavg_colour + avg_colour
            if allavg_count > 0:
                allavg_colour = allavg_colour/allavg_count
                hex_values.append((c_value/255.0, '#%02x%02x%02xff' % (int(allavg_colour[0]), int(allavg_colour[1]), int(allavg_colour[2]))))
        hex_values.append((1.0, '#ffffffff'))
        hex_values[0] = (0.0, hex_values[1][1])

        if overwrite_reference:
            np.save('./data/img_data/hex_values.npy', hex_values)
    else:
        hex_values_load = np.load('./data/img_data/hex_values.npy')
        hex_values = []
        for value in hex_values_load:
            hex_values.append((float(value[0]), str(value[1])))
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

def plot_pyvista(output, fieldlines, fname = './plots/vista.png'):
    """
    Plots calculated field lines using pyvista, along with a colourmap on the lower surface corresponding to the photospheric magnetic field.
    """

    off_screen = True
    print('Plotting in pyvista...')
    if off_screen and not os.name == 'nt':
       pv.start_xvfb()

    pvplot = pv.Plotter(off_screen=off_screen)
    pvplot.background_color = "black"

    Ps, Ss = np.meshgrid(output.grid.pg[:] + np.pi, output.grid.sg[:])
    xs = np.sin(Ps) * np.sqrt(1.0 - Ss**2)
    ys = np.cos(Ps) * np.sqrt(1.0 - Ss**2)
    zs = Ss

    surface_points = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    rectangle_coords = []; surface_scalars = []
    #Define surface faces. This isn't particularly easy due to the funny stretched coordinates, but I'll try to figure it out
    br_surface = output.bc[0][: ,: , 0]
    surf_max = np.max(np.abs(br_surface))
    for ti in range(output.grid.ns):
        for pi in range(output.grid.nphi):
            rectangle_coords.append([4, ti*(output.grid.nphi+1) + pi,ti*(output.grid.nphi+1) + pi+1,(ti+1)*(output.grid.nphi+1) + pi + 1,(ti+1)*(output.grid.nphi+1) + pi])
            surface_scalars.append(br_surface[pi,ti]/surf_max)
    rectangle_coords = np.array(rectangle_coords)

    surface = pv.PolyData(surface_points, rectangle_coords)

    sun_cmap = LinearSegmentedColormap.from_list(
    "sun", ["darkred", "orangered", "orange", "gold"], N=256)

    pvplot.add_mesh(surface, scalars=1.0-np.abs(surface_scalars), show_edges=False, cmap=sun_cmap, clim = [0.0,1.0])

    plot_open = True

    all_linepts = []
    all_lines = []
    ptcount = 0
    for li, line in enumerate(fieldlines):
        coords = line.coords
        coords.representation_type = 'cartesian'

        pts = np.zeros((len(coords), 3))
        pts[:,0] = coords.x/const.R_sun; pts[:,1] = coords.y/const.R_sun; pts[:,2] = coords.z/const.R_sun
        #Thin down the line if necessary

        if len(pts) > 2:
            #pvplot.add_mesh(pv.Spline(pts, len(pts)), color='white', line_width=1.0)
            if not plot_open and not(np.linalg.norm(pts[0]) < 1.1 and np.linalg.norm(pts[-1]) < 1.1):
                continue

            if not(np.linalg.norm(pts[0]) < 1.1 and np.linalg.norm(pts[-1]) < 1.1):
                if random.uniform(0,1) > 0.1:
                    continue

            all_linepts.append(pts)
            n = len(pts)
            all_lines.append(np.hstack([n, np.arange(ptcount, ptcount + n)]))
            ptcount += n

    if len(all_linepts) > 0:

        allpts_stack = np.vstack(all_linepts)
        all_lines = np.hstack(all_lines)

        spline_mesh = pv.PolyData()
        spline_mesh.points = allpts_stack
        spline_mesh.lines = all_lines
        if not off_screen or len(all_lines) < 1000:
            pvplot.add_mesh(spline_mesh, color='white', line_width=1.0)
        else:
            pvplot.add_mesh(spline_mesh, color='white', line_width=0.3)

    theta = np.pi/2
    r = 12.
    phi = 0.0
    pvplot.remove_scalar_bar()
    pvplot.camera.position = (r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta))
    pvplot.camera.focal_point = (0,0,0)
    #p.add_title('%d days' % t, font='times', color='white', font_size=40)
    if off_screen:
        #pass
        #pvplot.export_html('./plots/vista%05d.html' % Paras.run_id)
        #pvplot.add_text(Paras.data_time.strftime("%Y_%m_%d %H:%M"), position='lower_edge', font_size=36, color = 'white')
        pvplot.show(screenshot=fname,window_size=[2160, 2160])
    else:
        pvplot.show()

    print('Plotted seemingly sucessfullly...')

    return None


