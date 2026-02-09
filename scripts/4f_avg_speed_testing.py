#Just a little script to compare the implicit outflow speeds to an equivalent polynomial. Should allow an 'ideal temperature' and mf constants to be estimated -- hopefully!

import numpy as np
import matplotlib.pyplot as plt
import outflowpy
from scipy.optimize import root_scalar, minimize_scalar
from scipy import interpolate
from scipy.optimize import newton

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
    rg = np.linspace(1.0,2.5,1000)
    rg = np.log(rg)
    min_r = -1.0; max_r = rg[-1]*2.0
    vtest_min = 1e-6
    dr = (max_r - min_r)/2000
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

    rg = np.linspace(1.0,2.5,1000)
    rg = np.log(rg)
    vg = _get_parker_wind_speed(r_c)
    vg = vg*c_s
    return vg

def _poly_at_pt(r, polynomial_coeffs):
    #Polynomial value at the explicit point r (don't forget the exponentials!)
    res = 0
    for i in range(len(polynomial_coeffs)):
        res = res + polynomial_coeffs[i]*np.exp(r)**i
    return res

def find_speed_poly(coeffs):
    rg = np.linspace(1.0,2.5,1000)
    rg = np.log(rg)
    vg = _poly_at_pt(rg, coeffs)
    return vg

flow_data = np.loadtxt('./data/opt_flow.txt', delimiter = ',')
rs_interp = flow_data[:,0]
vs_interp = flow_data[:,1]

rg = np.linspace(1.0,2.5,1000)

rgx = np.zeros((len(rg) + 2))
rgx[1:-1] = rg
rgx[0] = 2*rgx[1] - rgx[2]; rgx[-1] = 2*rgx[-2] - rgx[-3]

f = interpolate.interp1d(rs_interp, vs_interp, fill_value = "extrapolate")
vgx = f(rgx)


vg_ref = vgx[1:-1]

def errorfn(temp):
    vg_test = find_speed_parker(5e-17, temp*1e6)

    mf = 5e-17*np.mean(vg_ref)/np.mean(vg_test)

    vg_test = vg_test*np.mean(vg_ref)/np.mean(vg_test)

    error = np.sum((vg_test - vg_ref)**2)

    print(temp, error)
    return error

best_temp = minimize_scalar(errorfn, bounds = [1.0,5.0]).x*1e6

vg_best = find_speed_parker(5e-17, best_temp)

mf = 5e-17*np.mean(vg_ref)/np.mean(vg_best)

vg_best = vg_best*np.mean(vg_ref)/np.mean(vg_best)

print('Best parameters to fit outflow profile', mf, best_temp)

plt.figure(figsize = (10,5))
plt.title(f"Optimum parameters: mf = {mf*1e17:.2f} x 10^17, temp = {best_temp/1e6:.2f} MK")
plt.plot(np.linspace(1.0,2.5,1000), vg_ref, c= 'black')
plt.plot(np.linspace(1.0,2.5,1000), vg_best, c = 'green')
plt.show()
plt.close()




