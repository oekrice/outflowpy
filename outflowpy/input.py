import copy

import numpy as np
import sunpy.map
import matplotlib.pyplot as plt

from scipy.optimize import root_scalar, minimize_scalar
from scipy import interpolate

import outflowpy.utils
import sys
from outflowpy.grid import Grid

from skimage import measure

class Input:
    r"""
    Input to PFSS/outflow field modelling.

    Parameters
    ----------
    br : sunpy.map.GenericMap
        Boundary condition of radial magnetic field at the inner surface.
        Note that the data *must* have a cylindrical equal area projection.
    nr : int
        Number of cells in the radial direction on which to calculate the 3D solution.
    rss : float
        Radius of the source surface, in units of solar radius
    corona_temp : float 
        Temperature of the corona for the implicit solar wind solution (see paper for details)
    mf_constant : float
        Magnetofrictional constant factor 

    Notes
    -----
    The input must be on a regularly spaced grid in :math:`\phi` and
    :math:`s = \cos (\theta)`. See `outflowpy.grid` for more
    information on the coordinate system.
    """
    def __init__(self, br, nr, rss, corona_temp = 2e6, mf_constant = 5e-17, polynomial_coeffs = None, polynomial_type = 'abs'):
        if not isinstance(br, sunpy.map.GenericMap):
            raise ValueError('br must be a sunpy Map object')
        if np.any(~np.isfinite(br.data)):
            raise ValueError('At least one value in the input is NaN or '
                             'non-finite. The input must consist solely of '
                             'finite values.')

        # The below does some checks to make sure this is a valid input
        outflowpy.utils.is_cea_map(br, error=True)
        outflowpy.utils.is_full_sun_synoptic_map(br, error=True)

        self._map_in = copy.deepcopy(br)
        self.dtime = self.map.date
        self.br = self.map.data

        # Force some nice defaults, just for plotting (I believe)
        self._map_in.plot_settings['cmap'] = 'RdBu'
        lim = np.nanmax(np.abs(self._map_in.data))
        self._map_in.plot_settings['vmin'] = -lim
        self._map_in.plot_settings['vmax'] = lim

        ns = self.br.shape[0]
        nphi = self.br.shape[1]
        self._grid = Grid(ns, nphi, nr, rss)

        if polynomial_coeffs is None:

            #Assuming the solution for the isothermal corona, calculate the sound speed and critical radius etc.
            #Could import these from astopy.constants but I don't think they're going to change any time soon, so I'll assume it's fine
            mf_in_sensible_units = mf_constant*(6.957e10)**2 #In seconds/solar radius
            sound_speed = np.sqrt(1.38064852e-23*corona_temp/1.67262192e-27) #Sound speed in m/s
            self.r_c = (6.67408e-11*1.98847542e30/(2*sound_speed**2))/(6.957e8)   #Critical radius in solar radii (code units)
            self.c_s = mf_in_sensible_units*sound_speed/6.957e8  #Sound speed in seconds/solar radius (code units)

            self.vg, self.vcx, self.vdcx = self._get_parker_wind_speed()

            #Then finally multiply by the 'wind speed' constant calculated using physics. Should all be roughly order of magnitude 1/10 after this.
            self.vg = self.vg*self.c_s
            self.vcx = self.vcx*self.c_s
            self.vdcx = self.vdcx*self.c_s
        
        else:
            #Calculate the wind speed just as a combination of the given polynomial coefficcients (with self._grid.rg known already)
            def poly_at_pt(r):
                #Polynomial value at the explicit point r (don't forget the exponentials!)
                res = 0
                for i in range(len(polynomial_coeffs)):
                    res = res + polynomial_coeffs[i]*np.exp(r)**i
                return res

            rgx = np.zeros((len(self._grid.rg) + 2))
            rgx[1:-1] = self._grid.rg
            rgx[0] = 2*rgx[1] - rgx[2]; rgx[-1] = 2*rgx[-2] - rgx[-3]

            vgx = poly_at_pt(rgx)
            vcx = poly_at_pt(self._grid.rcx)

            #Make these always positive (abs does weird things so just clip to zero)
            if polynomial_type == 'abs':
                vgx = np.abs(vgx)
                vcx = np.abs(vcx)
            elif polynomial_type == 'clip':
                vgx[vgx < 0] = 0.0
                vcx[vcx < 0] = 0.0

            elif polynomial_type == 'raw':
                vgx = vgx
                vcx = vcx
            else:
                raise Exception('Polynomial type not recognised. Currently allowed types are "clip", "abs" and "raw"')

            vdcx = np.zeros(len(vcx))
            vdcx = (vgx[1:] - vgx[:-1]) / (rgx[1:] - rgx[:-1])

            self.vg = vgx[1:-1]; self.vcx = vcx; self.vdcx = vdcx

            print('poly', polynomial_coeffs, polynomial_type)

    def _parker_implicit_fn(self, r, v):
        """
        This is where the implicit Parker Solar Wind function is defined.
        The algorithm should find zeros of this such that f(r, v) = 0.0
        The 'sound speed' here is set to zero as this will be scaled in the function _get_parker_wind_speed (makes the numerics more stable)
        """
        _c_s = 1.0; r_c = self.r_c
        if np.abs(v/_c_s) < 1e-12:
            return 1e12
        res = v**2/_c_s**2
        res -= 2*np.log(abs(v/_c_s))
        res -= 4*(np.log(abs(r/r_c)) + r_c/r)
        res += 3
        return res
    
    def _test_fn(self, r):
        """
        This is a test function to make sure the logged and unlogged radial coordinates don't get mixed up at any point.
        Should approximate a quadratic with velocity 1 at the upper boundary (2.5 here).
        """
        self.c_s = 1.0
        return (r/2.5)**2

    def _get_parker_wind_speed(self, implicit_fn = None):
        """
        Given up on the meshgrid approach as it just doesn't work very well for low velocities. 
        Instead doing the original options approach but with the linear prediction option if things are ambiguous
        """
        #Find initial point, assuming that the velocity is small here
        min_r = -1.0; max_r = self._grid.rg[-1]*2.0
        vtest_min = 1e-6
        dr = (max_r - min_r)/2000
        #Log two solutions
        vslows = []; vfasts = []
        r0s = []; vfinals = []
        r0 = min_r
        if not implicit_fn:
            implicit_fn = self._parker_implicit_fn

        if implicit_fn == self._test_fn:
            print("Doing test outflow profile, which doesn't involve any root finding")
            while r0 <= max_r:
                vfinals.append(self._test_fn(np.exp(r0)))
                r0s.append(r0)
                r0 = r0 + dr

        else:
            while r0 <= max_r:
                #Find the minimum value of the fn at this point? Would probably be more reliable for more complex functions.
                #Also could put a check in to make sure everything is the right way around?
                #Must be an inbuilt for the minimum of a function within a range?
                minimum = minimize_scalar(lambda v: implicit_fn(np.exp(r0), v))
                p0 = vtest_min; p1 = minimum.x; p2 = 10.0*minimum.x
                #If the three points have a crossing, then find the actual minimum using the standard root finding thing
                if  implicit_fn(np.exp(r0), p0)* implicit_fn(np.exp(r0), p1) < 0.0 and  implicit_fn(np.exp(r0), p1)* implicit_fn(np.exp(r0), p2) < 0.0:
                    #This is valid -- find the roots
                    vslow = root_scalar((lambda v: implicit_fn(np.exp(r0), v)), bracket = [p0, p1]).root
                    vfast = root_scalar((lambda v: implicit_fn(np.exp(r0), v)), bracket = [p1, p2]).root
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
        rgx = np.zeros((len(self._grid.rg) + 2))
        rgx[1:-1] = self._grid.rg
        rgx[0] = 2*rgx[1] - rgx[2]; rgx[-1] = 2*rgx[-2] - rgx[-3]

        vgx = vf(rgx)
        vcx = vf(self._grid.rcx)
        vdcx = np.zeros(len(vcx))
        vdcx = (vgx[1:] - vgx[:-1]) / (rgx[1:] - rgx[:-1])

        return vgx[1:-1], vcx, vdcx

    #These things are meant to be viewed outside the class -- everything else is kept within.
    @property
    def map(self):
        """
        :class:`sunpy.map.GenericMap` representation of the input.
        """
        return self._map_in

    @property
    def grid(self):
        """
        `~outflowpy.grid.Grid` that the PFSS solution for this input is
        calculated on.
        """
        return self._grid
