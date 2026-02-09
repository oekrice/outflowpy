#This script to generate a text array of the optimum values determined by the CMA-es runs.
#Because of the various absolute values these cannot be expressed explicitly (very easily, at least), so will just set up an interpolation thing
#It would be best if this ends up as the default, I think. Maybe. Perhaps if a corona temperature isn't specified?
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from scipy.optimize import root_scalar, minimize_scalar
from scipy import interpolate
from datetime import datetime, timedelta
import time
from scipy.interpolate import interp1d
import outflowpy

def toYearFraction(date):

    #Returns the year plus a fraction of the year, as a float. Very good.

    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)


    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration


    return date.year + fraction

frost_data = pd.read_csv("./data/frost_data.txt", sep='\\s+', comment = "%", header = None, on_bad_lines = 'skip')

years = frost_data[0].to_numpy()
months = frost_data[1].to_numpy()
days = frost_data[2].to_numpy()


dates = [datetime(int(years[i]), int(months[i]), int(days[i])) for i in range(len(years))]
tflux = np.array([toYearFraction(d) for d in dates])

oflux = frost_data[6]
ofluxmin = frost_data[7]
ofluxmax = frost_data[8]

tflux = tflux[oflux > 0]
ofluxmin = ofluxmin[oflux > 0]
ofluxmax = ofluxmax[oflux > 0]
oflux = oflux[oflux > 0]

#Sort out units
ofluxmin = ofluxmin*1e14*1e8
ofluxmax = ofluxmax*1e14*1e8
oflux = oflux*1e14*1e8

frost_xs = tflux
frost_ys = oflux

frost_interp = interp1d(frost_xs, frost_ys, kind = 'linear')

years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019]

eclipse_times =  []
for eclipse_year in years:
    eclipse_times.append(toYearFraction(datetime.fromisoformat(outflowpy.utils.find_eclipse_time(eclipse_year))))

frost_values = frost_interp(eclipse_times)

nx = 100
allys = np.zeros((nx))
ycount = 0

xs = np.linspace(1.0,2.5,nx)  #Basis for the x axis

model_predictions = np.zeros(len(years))




for ei, eclipse_number in enumerate(years):

    #os.system(f"scp -r vgjn10@hamilton8.dur.ac.uk:/home/vgjn10/projects/outflowpy/scripts/batch_logs/log_{eclipse_number}.txt ./batch_logs_{source}")
    os.system(f"scp -r vgjn10@hamilton8.dur.ac.uk:/home/vgjn10/projects/outflowpy/scripts/batch_logs/log_{eclipse_number}.txt ./data/batch_logs")

    #log_file = './batch_logs/log_%d.txt' % eclipse_number
    log_file = f'./data/batch_logs/log_{eclipse_number}.txt'

    if not os.path.exists(log_file):
        print('pass')
        continue

    log_info = []

    with open(log_file, "r") as f:
        for line in f.readlines():
            log_info.append(line.split(" "))
    log_info = np.array(log_info, dtype = 'float')

    #log_info = log_info[:counter+1,:]


    def implicit_fn(r_c, r, v):
        """
        This is where the implicit Parker Solar Wind function is defined.
        The algorithm should find zeros of this such that f(r, v) = 0.0
        The 'sound speed' here is set to zero as this will be scaled in the function _get_parker_wind_speed (makes the numerics more stable)
        """

        _c_s = 1.0; r_c = r_c
        if np.abs(v/_c_s) < 1e-12:
            return 1e12
        res = v**2/_c_s**2
        res -= 2*np.log(abs(v/_c_s))
        res -= 4*(np.log(abs(r/r_c)) + r_c/r)
        res += 3
        return res

    def _get_parker_wind_speed(r_c):
        """
        Given up on the meshgrid approach as it just doesn't work very well for low velocities.
        Instead doing the original options approach but with the linear prediction option if things are ambiguous
        """
        #Find initial point, assuming that the velocity is small here
        rg = xs
        rg = np.log(rg)
        min_r = -1.0; max_r = rg[-1]*2.0
        vtest_min = 1e-6
        dr = (max_r - min_r)/(2*len(rg))
        #Log two solutions
        vslows = []; vfasts = []
        r0s = []; vfinals = []
        r0 = min_r

        while r0 <= max_r:
            #Find the minimum value of the fn at this point? Would probably be more reliable for more complex functions.
            #Also could put a check in to make sure everything is the right way around?
            #Must be an inbuilt for the minimum of a function within a range?
            minimum = minimize_scalar(lambda v: implicit_fn(r_c, np.exp(r0), v))
            p0 = vtest_min; p1 = minimum.x; p2 = 10.0*minimum.x
            #If the three points have a crossing, then find the actual minimum using the standard root finding thing
            if  implicit_fn(r_c, np.exp(r0), p0)* implicit_fn(r_c, np.exp(r0), p1) < 0.0 and  implicit_fn(r_c, np.exp(r0), p1)* implicit_fn(r_c, np.exp(r0), p2) < 0.0:
                #This is valid -- find the roots
                vslow = root_scalar((lambda v: implicit_fn(r_c, np.exp(r0), v)), bracket = [p0, p1]).root
                vfast = root_scalar((lambda v: implicit_fn(r_c, np.exp(r0), v)), bracket = [p1, p2]).root
                vslows.append(vslow); vfasts.append(vfast)
                if len(vfinals) < 2:  #For the first two, it's probably safe to assume that this is the slow solution
                    vfinals.append(vslows[-1])
                    r0s.append(r0)
                else:
                    prediction = 2*vfinals[-1] - vfinals[-2]
                    diffslow = np.abs(vslows[-1] - prediction)
                    difffast = np.abs(vfasts[-1] - prediction)
                    if diffslow < difffast:
                        vfinals.append(vslows[-1])
                        r0s.append(r0)
                    else:
                        vfinals.append(vfasts[-1])
                        r0s.append(r0)
            else:
                #If r is reasonably small, it is probably zero, so add something to that effect at the start
                if r0 < np.log(2.5):
                    vfinals.append(0.0)
                    r0s.append(r0)
                else:
                    raise Exception('A sensible solution to the implicit wind speed equation could not be found')
            r0 = r0 + dr

        vfinals = np.array(vfinals); r0s = np.array(r0s)

        #Interpolate these values onto the desired grid points, then differentiate (in RHO)
        #To find the values on the extended inner cells, extend the grid cells (briefly) and do central differences
        vf = interpolate.interp1d(r0s, vfinals,bounds_error=False, fill_value='extrapolate')
        rgx = np.zeros((len(rg) + 2))
        rgx[1:-1] = rg
        rgx[0] = 2*rgx[1] - rgx[2]; rgx[-1] = 2*rgx[-2] - rgx[-3]

        vgx = vf(rgx)

        return vgx[1:-1]

    def find_speed_parker(mf_constant, corona_temp):
        #Finds the Parker coefficients
        mf_in_sensible_units = mf_constant*(6.957e10)**2 #In seconds/solar radius
        sound_speed = np.sqrt(1.38064852e-23*corona_temp/1.67262192e-27) #Sound speed in m/s
        r_c = (6.67408e-11*1.98847542e30/(2*sound_speed**2))/(6.957e8)   #Critical radius in solar radii (code units)
        c_s = mf_in_sensible_units*sound_speed/6.957e8  #Sound speed in seconds/solar radius (code units)

        rg = xs
        rg = np.log(rg)
        vg = _get_parker_wind_speed(r_c)
        vg = vg*c_s
        return vg

    best_id = 0; score = 1.
    for i in range(np.size(log_info,0)):
        if log_info[i,1] < score:
            best_id = i
            score = log_info[i,1]

    #     plt.plot(xs, ys, linewidth = 0.1, c = colors[ei])

    string = ''
    for var in range(2, np.size(log_info[0])):
        string = string + str(log_info[-1, var]) + ','
    print('Eclipse', eclipse_number, string)

    mf_constant = np.abs(log_info[-1,2])*1e-17
    corona_temp = np.abs(log_info[-1,3])*1e6

    print(mf_constant, corona_temp)
    ys = find_speed_parker(mf_constant, corona_temp)

    if np.max(ys) > 0.0:
        allys += ys
        ycount += 1

    #plt.plot(xs, ys, linewidth = 2.0, label = f'{eclipse_number}', linestyle = 'dashed')
    #print(log_info[best_id,2:])

    model_predictions[ei] = ys[-1]
    plt.scatter(frost_values[ei], ys[-1], label = years[ei])

r = np.corrcoef(frost_values, model_predictions)[0,1]

print('Correlation', r)
#plt.show()

plt.legend()
plt.savefig('nocorrelation.png')
plt.show()






