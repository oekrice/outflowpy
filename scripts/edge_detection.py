#Script to start testing the edge detection hypotheses. Will first attempt with the Canny edge detection module which seems to already exist. Just on the real photos for now as I think there will be better ways of doing it with the synthetic ones.

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d

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

#Now need to look at the angles of these lines to make a histogram like the paper. This will obviously be noisy but hopefully not too terrible.
#Histograms may be hard
def make_angle_histogram(all_lines, bin_resolution = 10, num = 0, resolution = 512):
    #Histogram bin sizes (to be incorporated somewhere proper later on). Doesn't really matter as long as it's consistent.
    nbins_r = bin_resolution
    nbins_theta = bin_resolution

    rbins = np.linspace(1.0, 2.5, nbins_r + 1)
    thetabins = np.linspace(0.0, 2*np.pi, nbins_theta + 1)

    def find_radius(x,y):
        #Given coordinates x, y, determine the radius in suns
        x_coord = 5.0*(x/resolution)
        y_coord = 5.0*(y/resolution)
        return np.sqrt(x_coord**2 + y_coord**2)

    for eclipse, eclipse_lines in enumerate(all_lines):
        histogram_sum = np.zeros((nbins_r, nbins_theta))
        histogram_count = np.zeros((nbins_r, nbins_theta))
        xs, ys, cs = [], [], []
        for line in eclipse_lines:
            #For every point along the line, log the position (do angle from the top, clockwise?. Maybe just arctan2 is best) and the angle
            for i in range(1,len(line[0])-1):
                x = line[0][i] - resolution/2; y = -1.0*(line[1][i] - resolution/2)
                angle = np.arctan2(np.abs(y), np.abs(x))
                dx = line[0][i+1] - line[0][i-1]
                dy = line[1][i+1] - line[1][i-1]
                #Make sure that the angle is always less than pi/2, as it doesn't matter which direction the line was traced.
                dangle = np.arctan2(np.abs(dy), np.abs(dx)) #This is the direction. Which could be off by pi/2, I suppose.
                #Let's establish a precedent. All ys are +ve, and all xs are +ve

                radial_difference = np.abs(dangle - angle)
                #print(radial_difference)
                xs.append(x); ys.append(y); cs.append(radial_difference)

                #This bit for binning
                real_angle = np.arctan2(x, y) + np.pi
                radius = find_radius(x, y)
                if radius > 1.0 and radius < 2.5:
                    r_index = int(nbins_r*((radius - rbins[0])/(rbins[-1] - rbins[0])))
                    theta_index = int(nbins_theta*((real_angle - thetabins[0])/(thetabins[-1] - thetabins[0])))
                    histogram_count[r_index, theta_index%nbins_theta] += 1
                    histogram_sum[r_index, theta_index%nbins_theta] += radial_difference
        #Filter the histogram values with too few in them, or will get spuriousity
        min_value = 2
        zero_mask = histogram_count < min_value
        histogram_sum[zero_mask] = 0
        histogram_count[zero_mask] = 1e-6

        histogram_mean = histogram_sum/histogram_count   #This is the number to care about, not the individual values
        histogram_count[histogram_count >= min_value] = 1  #Remove this weighting

        #Radial mean etc. should be not be weighted by the number of values I think. Yes, that works.
        radial_mean = np.sum(histogram_mean, axis = 1)/np.sum(histogram_count, axis = 1)
        latitude_mean = np.sum(histogram_mean, axis = 0)/np.sum(histogram_count, axis = 0)
        fig, axs = plt.subplots(1,1, figsize = (10,8))

        axs.set_xticks([]); axs.set_yticks([])
        plt.scatter(xs, ys, c = cs, s= 0.1, vmin = 0, vmax = np.percentile(cs, 90))
        plt.colorbar()
        plt.axis('equal')
        plt.savefig('./images_data/edges/radial_%d.png' % (num), dpi = 200)
        plt.close()

        fig, axs = plt.subplots(2,1, figsize = (10,8))

        im = axs[0].pcolormesh(thetabins, rbins, histogram_mean)
        plt.colorbar(im, label = 'Deviation from radialness')
        axs[0].set_xlabel('Latitude')
        axs[0].set_ylabel('Altitude')

        axs[1].plot(0.5*(rbins[1:] + rbins[:-1]), radial_mean)
        axs[1].set_xlabel('Altitude')
        axs[1].set_ylabel('Avg. Deviation from Radial')
        axs[1].set_ylim(ymin = 0.0, ymax = 1.0)

        # axs[2].plot(0.5*(thetabins[1:] + thetabins[:-1]), latitude_mean)
        # axs[2].set_xlabel('Latitude')
        # axs[2].set_ylabel('Avg. Deviation from Radial')

        plt.suptitle(('Eclipse Number ' + str(eclipse)))
        plt.tight_layout()
        plt.savefig('./images_data/edges/hist_%d.png' % (num), dpi = 200)
        plt.close()

    return radial_mean

def find_decent_lines(counter, resolution, image_title, year):
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

    for j in range(len(tests)):
        brightness_scale = 105.

        scale_factor = brightness_scale/np.mean(img)
        img = scale_factor*img
        img = gaussian_filter(img.copy(),1)
        img = np.clip(img,0,255)
        img = img.astype(np.uint8)

        t_lower = 60; t_upper = 250
        aperture_size = 5
        edges = cv.Canny(img,t_lower, t_upper, apertureSize = aperture_size, L2gradient = True)

        fig, axs = plt.subplots(2,2, figsize = (10,10))
        axs[0,0].imshow(img_original,cmap = 'gray')
        axs[0,0].set_title('Original Image'), axs[0,0].set_xticks([]), axs[0,0].set_yticks([])
        # axs[1].imshow(img, cmap = 'gray')
        # axs[1].set_title('Processed Image'), axs[1].set_xticks([]), axs[1].set_yticks([])
        axs[0,1].imshow(edges, cmap = 'gray')
        axs[0,1].set_title('Edges'), axs[0,1].set_xticks([]), axs[0,1].set_yticks([])

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
                    axs[1,0].plot(xs[section[0]:section[1]], ys[section[0]:section[1]], c = 'blue', linewidth = 0.1)
                    if section[1] - section[0] > 20 and np.max(rs[section[0]:section[1]]) > 1.05 and np.min(rs[section[0]:section[1]]) < 2.45:
                        line_xs = np.array([x[0] for x in xs[section[0]:section[1]]]).astype(float)
                        line_ys = np.array([y[0] for y in ys[section[0]:section[1]]]).astype(float)
                        #Apply smoothing here before saving out.
                        line_xs = gaussian_filter1d(line_xs, sigma = line_smoothing)
                        line_ys = gaussian_filter1d(line_ys, sigma = line_smoothing)
                        eclipse_lines.append([line_xs, line_ys])
                        axs[1,0].plot(eclipse_lines[-1][0], eclipse_lines[-1][1], c = 'red', linewidth = 1.0)
                    qualities.append(dr)

        qualities = np.array(qualities)
        nabove = len(qualities[qualities > 0.5])
        print('Avg. line length and number of long lines', j, tests[j], np.mean(qualities), nabove)
        axs[1,0].imshow(img_original,cmap = 'gray',vmin=0, vmax = 255)
        axs[1,0].axis('equal')
        axs[1,0].set_xticks([]); axs[1,0].set_yticks([])
        axs[1,0].set_axis_off()
        axs[1,0].set_title('Selected Edges')


        #Determine the radialness of these lines
        xs, ys, cs = [], [], []
        for line in eclipse_lines:
            #For every point along the line, log the position (do angle from the top, clockwise?. Maybe just arctan2 is best) and the angle
            for i in range(1,len(line[0])-1):
                x = line[0][i] - resolution/2; y = -1.0*(line[1][i] - resolution/2)
                angle = np.arctan2(np.abs(y), np.abs(x))
                dx = line[0][i+1] - line[0][i-1]
                dy = line[1][i+1] - line[1][i-1]
                #Make sure that the angle is always less than pi/2, as it doesn't matter which direction the line was traced.
                dangle = np.arctan2(np.abs(dy), np.abs(dx)) #This is the direction. Which could be off by pi/2, I suppose.
                #Let's establish a precedent. All ys are +ve, and all xs are +ve

                radial_difference = np.abs(dangle - angle)
                #print(radial_difference)
                xs.append(x); ys.append(y); cs.append(radial_difference)

        axs[1,1].set_xticks([]); axs[1,1].set_yticks([])
        axs[1,1].scatter(xs, ys, c = cs, s = 0.1, vmin = 0, vmax = np.percentile(cs, 90))
        axs[1,1].set_title('Field line angles')
        axs[1,1].set_axis_off()

        plt.suptitle(f"{year} eclipse")
        plt.savefig('./edge_detection/%d_%d.png' % (year,j), dpi = 200)
        plt.close()

        all_lines.append(eclipse_lines)
    return all_lines

def analyse_image_edges(years, counter = 0, image_title = '2019_trim.png', nbins = 30):

    #Wrapper function for a single eclipse/generated image
    eclipse_image_root = './data/eclipse_images/'

    image_titles = []
    for year in years:
        image_titles.append(f"{eclipse_image_root}{year}_eclipse.png")

    n_imgs = len(image_titles)
    resolution = 512

    for pi, img_title in enumerate(image_titles):
        all_lines = find_decent_lines(counter, resolution, img_title, years[pi])
        #radial_mean = make_angle_histogram(all_lines, bin_resolution = nbins, num = counter, resolution = resolution)

    return radial_mean

years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]
analyse_image_edges(years)




