#Script to do a nice animation of the convergence of an outflow PARKER SOLUTION
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.optimize import root_scalar, minimize_scalar
from scipy import interpolate

colors = sns.color_palette('tab20')

fig = plt.figure(figsize = (9,5))
years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]

sources = ['clip', 'abs', 'raw', 'parker']

source = sources[3]

nx = 100
allys = np.zeros((nx))
ycount = 0

if os.path.exists(f"batch_logs_{source}/optimums.txt"):
    os.remove(f"batch_logs_{source}/optimums.txt")

for counter in range(0,1):
    xs = np.linspace(1.0,2.5,nx)  #Basis for the x axis

    for ei, eclipse_number in enumerate(years):

        os.system(f"scp -r vgjn10@hamilton8.dur.ac.uk:/home/vgjn10/projects/outflowpy/scripts/batch_logs/log_{eclipse_number}.txt ./batch_logs_{source}")

        #log_file = './batch_logs/log_%d.txt' % eclipse_number
        log_file = f'./batch_logs_{source}/log_{eclipse_number}.txt'

        if not os.path.exists(log_file):
            continue

        print('asd')
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

        if source == 'clip':
            ys[ys < 0.0] = 0.0
        elif source == 'abs':
            ys = np.abs(ys)
        else:
            ys = ys

        if np.max(ys) > 0.0:
            allys += ys
            ycount += 1

        plt.plot(xs, ys, linewidth = 2.0, c = colors[2*ei%20 + ei//10], label = f'{eclipse_number}', linestyle = 'dashed')
        #print(log_info[best_id,2:])

        with open(f"batch_logs_{source}/optimums.txt", mode = "a") as f:
            f.write(f"{log_info[-1, 2:].tolist()}\n")

    plt.plot(xs, allys/ycount, linewidth = 3.0, c = 'black', label = 'Mean', linestyle = 'solid')

    plt.title('Optimum outflow speeds for various eclipses')
    plt.xlabel('Radius')
    plt.ylabel('Outflow speed (code units)')
    plt.legend()
    plt.savefig('./comp.png')
    plt.show()
    plt.close()
