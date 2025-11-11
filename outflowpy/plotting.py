import pyvista as pv
import os
import numpy as np
import astropy.constants as const
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
from matplotlib.patches import Circle

def make_image(image_matrix, image_extent, image_parameters, image_fname):

    size = 7
    cmap = "bone"
    fig, ax = plt.subplots(figsize = (size, size))
    moon_face = mpimg.imread("./data/moonface_druck.png")

    image_matrix = np.flip(image_matrix, 1)

    xs = np.linspace(-image_extent,image_extent,np.shape(image_matrix)[0])
    ys = np.linspace(-image_extent,image_extent,np.shape(image_matrix)[1])
    ax.imshow(image_matrix.T, cmap = cmap, extent = [-image_extent,image_extent,-image_extent,image_extent],interpolation="bilinear")

    moon_img = ax.imshow(moon_face, extent = [-1,1,-1,1],interpolation="bilinear")
    circle = Circle((0, 0), 0.995, transform = ax.transData)
    moon_img.set_clip_path(circle)

    ax.set_xlim(-image_extent, image_extent)
    ax.set_ylim(-image_extent, image_extent)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    plt.savefig(image_fname)
    plt.close()

def plot_pyvista(output, fieldlines):
    """
    Plots calculated field lines using pyvista, along with a colourmap on the lower surface corresponding to the photospheric magnetic field.
    """


    #Uses the three classes to construct a Pyvista plot replete with all the options one might require from such a thing
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
        pvplot.show(screenshot='plots/vista%05d.png' % 0,window_size=[2160, 2160])
    else:
        pvplot.show()

    print('Plotted seemingly sucessfullly...')

    return None
