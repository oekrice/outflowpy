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

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # LaTeX-like Computer Modern
    "font.family": "serif",
})

colors = sns.color_palette('colorblind')

#cmap = sns.color_palette("magma", as_cmap=True)
#plt.rcParams['axes.prop_cycle'] = cycler('linestyle', ['-', '--', '-.', ':'])


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

#Histograms may be hard
def make_angle_histogram(flines, bin_resolution = 10, num = 0, resolution = 512):
    #Histogram bin sizes (to be incorporated somewhere proper later on). Doesn't really matter as long as it's consistent.

    nbins_r = bin_resolution
    nbins_theta = bin_resolution

    rbins = np.linspace(1.0, 2.5, nbins_r + 1)
    thetabins = np.linspace(0.0, 2*np.pi, nbins_theta + 1)

    def find_radius(x,y):
        #Given coordinates x, y, determine the radius in suns
        return np.sqrt(x**2 + y**2)


    histogram_sum = np.zeros((nbins_r, nbins_theta))
    histogram_count = np.zeros((nbins_r, nbins_theta))
    xs, ys, cs = [], [], []
    for line in flines:
        #For every point along the line, log the position (do angle from the top, clockwise?. Maybe just arctan2 is best) and the angle
        for i in range(1,len(line[0])-1):

            x, y = line[0][i], line[1][i]
            angle = np.arctan2(y, x)
            x_up, y_up     = line[0][i+1], line[1][i+1]
            x_down, y_down = line[0][i-1], line[1][i-1]

            dx = x_up - x_down
            dy = y_up - y_down

            #Calculate angle using cosine similarity
            top = np.abs(x*dx + y*dy)
            bottom = np.sqrt(x**2 + y**2)*np.sqrt(dx**2 + dy**2)

            dangle = np.arccos(top/bottom)

            radial_difference = dangle

            #This bit for binning
            real_angle = np.arctan2(x, y) + np.pi
            radius = find_radius(x, y)

            if radius > 1.0 and radius < 2.5:
                r_index = int(nbins_r*((radius - rbins[0])/(rbins[-1] - rbins[0])))
                theta_index = int(nbins_theta*((real_angle - thetabins[0])/(thetabins[-1] - thetabins[0])))
                histogram_count[r_index, theta_index%nbins_theta] += 1
                histogram_sum[r_index, theta_index%nbins_theta] += radial_difference

    #Filter the histogram values with too few in them, or will get spuriousity
    min_value = 10
    zero_mask = histogram_count < min_value
    histogram_sum[zero_mask] = 0
    histogram_count[zero_mask] = 1e-6

    histogram_mean = histogram_sum/histogram_count   #This is the number to care about, not the individual values
    histogram_count[histogram_count >= min_value] = 1  #Remove this weighting

    return histogram_mean, histogram_count

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
                dr = np.abs(rs[section[1]-1] - rs[section[0]])
                if section[1] - section[0] > 20 and np.max(rs[section[0]:section[1]]) > 1.05 and np.min(rs[section[0]:section[1]]) < 2.45:
                    line_xs = np.array([x[0] for x in xs[section[0]:section[1]]]).astype(float)
                    line_ys = np.array([y[0] for y in ys[section[0]:section[1]]]).astype(float)
                    #Apply smoothing here before saving out.
                    line_xs = gaussian_filter1d(line_xs, sigma = line_smoothing)
                    line_ys = gaussian_filter1d(line_ys, sigma = line_smoothing)
                    eclipse_lines.append([line_xs, line_ys])
                qualities.append(dr)

    qualities = np.array(qualities)
    nabove = len(qualities[qualities > 0.5])
    print('Avg. line length and number of long lines', np.mean(qualities), nabove)

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

            transformed_lines.append([line_xs, line_ys])

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

    return transformed_lines

def find_real_angle_distribution(eclipse_year, counter, doplots = True):
    r"""
    Produces an array of the average field line angles for a given eclipse year.
    Also outputs a mask of the areas to be ignored as they don't contain enough data, to allow for a fair comparison with the synthetic eclipses
    """
    resolution = 512
    eclipse_image_root = './data/eclipse_images/'
    img_title = f"{eclipse_image_root}{eclipse_year}_eclipse.png"

    fieldlines = find_decent_lines(resolution, img_title, eclipse_year)

    nbins = 30
    bin_means, bin_mask = make_angle_histogram(fieldlines, nbins)

    rbins = np.linspace(1.0, 2.5, nbins + 1)
    thetabins = np.linspace(0.0, 2*np.pi, nbins + 1)
    bin_means[bin_mask < 0.5] = np.nan
    rcs = 0.5*(rbins[1:] + rbins[:-1])

    radial_distribution = np.zeros(nbins)
    #Take the average of the not-nans in each dimension
    for ri in range(nbins):
        nvalues = nbins - np.isnan(bin_means[ri,:]).sum()
        if nvalues > 0:
            radial_distribution[ri] = np.nansum(bin_means[ri,:])/nvalues
        else:
            radial_distribution[ri] = np.nan

    return radial_distribution, bin_mask
#Now look into the synthetic angle distributions

def find_eclipse_flines(eclipse_year, optimised = True, rss = 5.0):

    nrho = 60
    rss = rss
    ns = 60
    nphi = 120

    corona_temp = 1.5e6
    mf_constant = 5e-17
    nseeds = 2000

    image_extent = 2.5
    image_resolution = 512

    obs_time = outflowpy.utils.find_eclipse_time(eclipse_year)

    year_options = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]
    poly_values = [0.0,0.0,0.0,0.0,0.0]

    allpolys = np.load(f"./data/batch_logs/optimums.npy")

    eclipse_index = year_options.index(eclipse_year)

    input_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = True)   #Outputs the set of data corresponding to this particular Carrington rotation.

    year_index = year_options.index(eclipse_year)
    if optimised:
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
            #Make sure that the angle is always less than pi/2, as it doesn't matter which direction the line was traced.
            dangle = np.arctan2(np.abs(dy), np.abs(dx)) #This is the direction. Which could be off by pi/2, I suppose.
            #Let's establish a precedent. All ys are +ve, and all xs are +ve
            radial_difference = np.abs(dangle - angle)

    return transformed_lines

def find_synthetic_angle_distribution(eclipse_year, nbins, doplots = False, optimised = True, rss = 5.0):
    r"""
    Calculates the outflow field and saves out the field lines in a nice format
    """

    fieldlines = find_eclipse_flines(eclipse_year, optimised = optimised, rss = rss)

    # for line in fieldlines:
    #     plt.scatter(line[0], line[1])
    #
    # plt.show()
    bin_means, bin_mask = make_angle_histogram(fieldlines, nbins)

    bin_means[bin_mask < 0.5] = np.nan

    return bin_means

#find_synthetic_angle_distribution(2017, 30)

def compare_angles(year):
    cmap = plt.cm.tab20

    nbins = 30
    real_distribution, bin_mask = find_real_angle_distribution(year, nbins)

    outflow_means = find_synthetic_angle_distribution(year, nbins, optimised = True, rss = 5.0)
    outflow_distribution = np.zeros(nbins)

    outflow_in_means = find_synthetic_angle_distribution(year, nbins, optimised = True, rss = 2.5)
    outflow_in_distribution = np.zeros(nbins)

    pfss_means = find_synthetic_angle_distribution(year, nbins, optimised = False, rss = 5.0)
    pfss_distribution = np.zeros(nbins)

    pfss_in_means = find_synthetic_angle_distribution(year, nbins, optimised = False, rss = 2.5)
    pfss_in_distribution = np.zeros(nbins)
    #Take the average of the not-nans in each dimension
    for ri in range(nbins):
        nvalues = nbins - np.isnan(bin_mask[ri,:]).sum()
        if nvalues > 0:
            outflow_distribution[ri] = np.nansum(outflow_means[ri,:])/nvalues
            outflow_in_distribution[ri] = np.nansum(outflow_in_means[ri,:])/nvalues
            pfss_distribution[ri] = np.nansum(pfss_means[ri,:])/nvalues
            pfss_in_distribution[ri] = np.nansum(pfss_in_means[ri,:])/nvalues
        else:
            outflow_distribution[ri] = np.nan
            outflow_in_distribution[ri] = np.nan
            pfss_distribution[ri] = np.nan
            pfss_in_distribution[ri] = np.nan


    rbins = np.linspace(1.0, 2.5, 31)
    rcs = 0.5*(rbins[1:] + rbins[:-1])
    fig = plt.figure(figsize = (6.9,4))
    plt.plot(rcs[1:], real_distribution[1:], label = "Reference image", c= 'black')
    plt.plot(rcs[1:], pfss_distribution[1:], label = "PFSS, $r_{ss} = 5.0R_\odot$", c = colors[3], linestyle = 'dashed')
    plt.plot(rcs[1:], pfss_in_distribution[1:], label = "PFSS, $r_{ss} = 2.5R_\odot$", c = colors[3])
    plt.plot(rcs[1:], outflow_distribution[1:], label = "Outflow, $r_{ss} = 5.0R_\odot$", c = colors[0], linestyle = 'dashed')
    plt.plot(rcs[1:], outflow_in_distribution[1:], label = "Outflow, $r_{ss} = 2.5R_\odot$", c = colors[0])

    #plt.title(f"{year} eclipse")
    #plt.ylim(ymin = 0.0)
    plt.xlim(1.0, 2.5)
    plt.xlabel('Altitude r ($R_\odot$)')
    plt.ylabel('Avg. deviation from radial direction (radians)')
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'./temp/angles_fluxmatch_{year}.png')
    if year == 2017:
        plt.savefig('11_quality_check.pdf')
    plt.close()
    return

years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]
allpolys = []
years = [2017]

with open(f"./data/batch_logs/optimums.txt") as f:
    for i, line in enumerate(f):
        poly_string = line.strip()
        poly_values = [float(x) for x in poly_string[1:-1].split(",")]
        allpolys.append(poly_values)

allpolys = np.array(allpolys)
np.save(f"./data/batch_logs/optimums.npy", allpolys)

for year in years:
    compare_angles(year)





