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
import cv2 as cv
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # LaTeX-like Computer Modern
    "font.family": "serif",
})

cmap = sns.color_palette("magma", as_cmap=True)


def determine_monotonic_sections(rs, length_limit = 50):
    #Use this to separate out the bits of field lines which are probably those which we care about. Some may loop back on each other and things, which is undesirable.
    #This just returns the start and end of the longest 'monotonic' section
    section_ends = []
    updown = -1 #Whether it's going up or down. Doesn't really matter overall
    for i, ri in enumerate(rs[:-1]):
        if i == 0:
            section_ends.append(i)
            updown = np.sign(rs[i+1] - rs[i])
        else:
            if np.sign(rs[i+1] - rs[i]) != updown:
                section_ends.append(i)
                updown = np.sign(rs[i+1] - rs[i])
    section_ends.append(len(rs))
    #section_ends are now all the points at which the line changes direction
    #They need to be filtered for noise a bit... Lots of wiggling in some of the lines
    sections = []
    for test_section in range(len(section_ends) - 1):
        if section_ends[test_section+1] - section_ends[test_section] > length_limit:
            sections.append([section_ends[test_section], section_ends[test_section+1]])

    return sections

def find_decent_lines(resolution, image_title, year, doplots = True):
    #Let's run through a load of parameters and see what happens...'
    #Uses the edge detection algorithm to return a list of the lines which are worth having a look at
    def find_radius(x,y):
        #Given coordinates x, y, determine the radius in suns
        x_coord = 5.0*(x - resolution//2)/resolution
        y_coord = 5.0*(y - resolution//2)/resolution
        return np.sqrt(x_coord**2 + y_coord**2)

    all_lines = []
    img = cv.imread(image_title, cv.IMREAD_GRAYSCALE)
    img= cv.resize(img, (resolution , resolution), interpolation=cv.INTER_LINEAR)
    img_original = img.copy()

    smooth_factor = 1
    line_smoothing = 2 #Applies Gaussian filter to the individual lines, to make angle appraisal more accurate
    lowpass_smoothing = 5
    tests = np.linspace(50,1000,1)

    brightness_scale = 105.

    scale_factor = brightness_scale/np.mean(img)
    img = scale_factor*img
    img = gaussian_filter(img.copy(),1)
    img = np.clip(img,0,255)
    img = img.astype(np.uint8)

    t_lower = 60; t_upper = 250
    aperture_size = 5
    edges = cv.Canny(img,t_lower, t_upper, apertureSize = aperture_size, L2gradient = True)

    if doplots:
        fig, axs = plt.subplots(2,2, figsize = (6.9, 6.9))
        axs[0,0].imshow(img_original,cmap = 'gray', rasterized=True)
        axs[0,0].set_title('Pre-processed Image'), axs[0,0].set_xticks([]), axs[0,0].set_yticks([])
        # axs[1].imshow(img, cmap = 'gray')
        # axs[1].set_title('Processed Image'), axs[1].set_xticks([]), axs[1].set_yticks([])
        axs[0,1].imshow(edges, cmap = 'gray', rasterized=True)
        axs[0,1].set_title('All Detected Edges'), axs[0,1].set_xticks([]), axs[0,1].set_yticks([])

    #Look for contours?
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    qualities = []  #Invent some kind of measure of the quality of the field lines.
    #Number above a certain length in the outward direction
    eclipse_lines = []
    for ci, contour in enumerate(contours):
        if len(contour) > 1:
            xs, ys = np.array(contour)[:,:,0], np.array(contour)[:,:,1]
            rs = find_radius(xs, ys)
            #Flip y coordinate, because image
            sections = determine_monotonic_sections(rs, length_limit = 0)
            for section in sections:
                #These are the actual ones to plot, and the parameters can be optimised based upon them
                #axs[1].plot(np.arange(section[0], section[1]), rs[section[0]:section[1]])
                dr = np.abs(rs[section[1]-1] - rs[section[0]])
                # if doplots:
                #     axs[1,0].plot(xs[section[0]:section[1]], ys[section[0]:section[1]], c = 'blue', linewidth = 0.1)
                if section[1] - section[0] > 20 and np.max(rs[section[0]:section[1]]) > 1.05 and np.min(rs[section[0]:section[1]]) < 2.45:
                    line_xs = np.array([x[0] for x in xs[section[0]:section[1]]]).astype(float)
                    line_ys = np.array([y[0] for y in ys[section[0]:section[1]]]).astype(float)
                    #Apply smoothing here before saving out.
                    line_xs = gaussian_filter1d(line_xs, sigma = line_smoothing)
                    line_ys = gaussian_filter1d(line_ys, sigma = line_smoothing)
                    eclipse_lines.append([line_xs, line_ys])
                    if doplots:
                        axs[1,0].plot(eclipse_lines[-1][0], eclipse_lines[-1][1], c = 'red', linewidth = 1.0)
                qualities.append(dr)

    qualities = np.array(qualities)
    nabove = len(qualities[qualities > 0.5])
    print('Avg. line length and number of long lines', np.mean(qualities), nabove)
    if doplots:
        axs[1,0].imshow(img_original,cmap = 'gray',vmin=0, vmax = 255, rasterized = True)
        axs[1,0].axis('equal')
        axs[1,0].set_xticks([]); axs[1,0].set_yticks([])
        axs[1,0].set_axis_off()
        axs[1,0].set_title('Selected Edges')

    transformed_lines = []


    def absangle(ang1, ang2):
        #Find the minimum distance between two angles.
        raw_diff = abs(ang1 - ang2)
        if raw_diff > np.pi:
            diff = 2*np.pi - np.pi
        else:
            diff = raw_diff

        if diff < np.pi/2:
            return diff
        else:
            return np.pi - diff

    def pt_to_xy(pt):
        return 5.0*(pt[0] - resolution/2)/resolution, -5.0*(pt[1] - resolution/2)/resolution
    #Determine the radialness of these lines
    for line in eclipse_lines[:]:
        line_xs = 2*2.5*(np.array(line)[0,:] - resolution/2)/resolution
        line_ys = -2*2.5*(np.array(line)[1,:] - resolution/2)/resolution
        xs, ys, cs = [], [], []

        if True:#line_xs[0]**2 + line_ys[0]**2 < line_xs[-1]**2 + line_ys[-1]**2:
            #  #Reverse the direction, for consistency. So they always go outwards. Not really necessary but meh.
            # line[0] = line[0][::-1]
            # line[1] = line[1][::-1]
            #For every point along the line, log the position (do angle from the top, clockwise?. Maybe just arctan2 is best) and the angle
            for i in range(1,len(line[0])-1):

                x, y = pt_to_xy([line[0][i], line[1][i]])
                #x = line[0][i] - resolution/2; y = -1.0*(line[1][i] - resolution/2)
                angle = np.arctan2(y, x)
                x_up, y_up     = pt_to_xy([line[0][i+1], line[1][i+1]])
                x_down, y_down = pt_to_xy([line[0][i-1], line[1][i-1]])

                dx = x_up - x_down
                dy = y_up - y_down
                #Make sure that the angle is always less than pi/2, as it doesn't matter which direction the line was traced.


                #Calculate angle using cosine similarity
                top = np.abs(x*dx + y*dy)
                bottom = np.sqrt(x**2 + y**2)*np.sqrt(dx**2 + dy**2)

                dangle = np.arccos(top/bottom)

                #dangle = np.arctan2(dy, dx) + np.pi #This is the direction. Which could be off by pi/2, I suppose.

                #Pick the angle opposite this if necessary. They should be close
                radial_difference = dangle#absangle(angle, dangle)

                xs.append(x); ys.append(y); cs.append(radial_difference)

        if len(xs) > 0:
            # Normalize color range
            norm = Normalize(0.0, np.pi/3)

            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Create LineCollection
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(cs)            # Assign values for colormap
            lc.set_linewidth(1)
            axs[1,1].add_collection(lc)

    #axs[1,1].scatter(xs, ys, c = cs, s = 0.5, marker = ',', vmin = 0.0, vmax = np.percentile(cs, 98), cmap = cmap)

    if doplots:
        axs[1,1].set_xticks([]); axs[1,1].set_yticks([])
        #axs[1,1].scatter(xs, ys, c = cs, s = 0.1, vmin = 0, vmax = np.percentile(cs, 90))
        axs[1,1].set_title('Field line deviation from radial')
        axs[1,1].set_axis_off()
        axs[1,1].set_xlim(-2.5,2.5)
        axs[1,1].set_ylim(-2.5,2.5)
        plt.tight_layout()
        plt.savefig('7_real_images.png', dpi = 700)
        plt.show()
        plt.close()

    return transformed_lines

def find_real_angle_distribution(eclipse_year, doplots = True):
    r"""
    Produces an array of the average field line angles for a given eclipse year.
    Also outputs a mask of the areas to be ignored as they don't contain enough data, to allow for a fair comparison with the synthetic eclipses
    """
    resolution = 512
    eclipse_image_root = './data/eclipse_images/'
    img_title = f"{eclipse_image_root}{eclipse_year}_eclipse.png"

    fieldlines = find_decent_lines(resolution, img_title, eclipse_year)


years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]

find_real_angle_distribution(2017)
