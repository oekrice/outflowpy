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
    def __init__(self, br, nr, rss, corona_temp = 10e6, mf_constant = 1e-18):
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

        #Assuming the solution for the isothermal corona, calculate the sound speed and critical radius etc.
        mf_in_sensible_units = mf_constant*(6.957e10)**2   #In seconds/solar radius
        sound_speed = np.sqrt(1.380649e-23*corona_temp/1.67262192e-27) #Sound speed in m/s
        self.r_c = (6.6743e-11*1.989e30/(2*sound_speed**2))/(6.957e8)   #Critical radius in solar radii (code units)
        self.c_s = mf_in_sensible_units*sound_speed/6.957e8  #Sound speed in seconds/solar radius (code units)

        self.vg, self.vdg = self._get_parker_wind_speed()
        #Then finally multiply by the 'wind speed' constant calculated using physics.
        self.vg = self.vg*self.c_s
        self.vdg = self.vdg*self.c_s
        
    def _parker_implicit_fn(self, r, v):
        """
        This is where the implicit Parker Solar Wind function is defined.
        The algorithm should find zeros of this such that f(r, v) = 0.0
        The 'sound speed' here is set to zero as this will be scaled in the function _get_parker_wind_speed (makes the numerics more stable)
        """
        _c_s = 1.0; r_c = self.r_c
        res = v**2/_c_s**2
        res -= 2*np.log(abs(v/_c_s))
        res -= 4*(np.log(abs(r/r_c)) + r_c/r)
        res += 3
        return res
    
    def _get_parker_wind_speed(self):
        """
        Algorithm to find the zeros of the implicit function defined in _parker_implicit_fn.
        Originally used a lambda function and a scalar minimisation routine, but now will attempt to do the same using contours
        This will hopefully be faster and more reliable.
        """
        #Create a meshgrid in r, v and find the zero contours of it. 
        # Note that r here is actually the log values, and this will need to be taken into account in the implicit function.

        r_interp = np.linspace(0.0, 5*self._grid.rg[-1], self._grid.nr*20)
        v_interp = np.linspace(1e-6, 2.0, self._grid.nr*20)
        R, V = np.meshgrid(r_interp, v_interp)
        outflow_speed_grid = self._parker_implicit_fn(np.exp(R), V)

        # plt.pcolormesh(np.exp(R), V, outflow_speed_grid)
        cs = plt.contour(R,V,outflow_speed_grid, levels = [0])
        #Extract monotonic (in x) sections of these contours for interpolation
        all_lines = []
        for contour in cs.get_paths():
            section = [[contour.vertices[0,0], contour.vertices[0,1]]]
            ddir = np.sign(contour.vertices[1,0] - contour.vertices[0,0])   #Direction at the start of this section
            for i in range(len(contour.vertices[:,0]) - 1):
                if np.sign(contour.vertices[i+1,0] - contour.vertices[i,0]) == ddir:
                    section.append([contour.vertices[i+1,0], contour.vertices[i+1,1]])
                else:
                    all_lines.append(section)
                    section = [[contour.vertices[i+1,0], contour.vertices[i+1,1]]]
                    if i < len(contour.vertices[:,0]) - 1:
                        ddir = np.sign(contour.vertices[i+2,0] - contour.vertices[i+1,0])
                    else:
                        break

            all_lines.append(section)

        #Find the appropriate section by ensuring it is monotonically increasing. 
        #Picks the smallest value that is larger than the last
        
        r_options = np.zeros((len(self._grid.rg), len(all_lines)))
        for si, section in enumerate(all_lines[:]):
            f_interp = interpolate.interp1d(np.array(section)[:,0], np.array(section)[:,1], fill_value = 0, bounds_error = False)
            interp_section = f_interp(self._grid.rg)
            r_options[:,si] = interp_section

        #Now options have been established, run through and pick the ones which make the most sense
        vfinals = 0.0*self._grid.rg
        vfinals[0] = np.min(r_options[0,:][np.nonzero(r_options[0,:])])
        for i in range(1,len(vfinals)):
            actual_options = r_options[i,:][np.nonzero(r_options[i,:])][r_options[i,:][np.nonzero(r_options[i,:])] > vfinals[i-1]]                
            if i == 1:
                if len(actual_options) == 0:
                    vfinals[i] = vfinals[i-1]
                else:
                    vfinals[i] = np.min(actual_options)
            else:   #Predict based on the derivative and choose the closest option (if one is available)
                prediction =  (2*vfinals[i-1] - vfinals[i-2])
                if len(actual_options) == 0:
                    vfinals[i] = prediction
                else:
                    choice = np.argmin(np.abs(actual_options - prediction))
                    vfinals[i] = actual_options[choice]

        vdiffs = (vfinals[1:] - vfinals[:-1]) / (self._grid.rg[1:] -  self._grid.rg[:-1])

        return vfinals, vdiffs

    #These things are meant to be viewed outside the class -- everything else is kept within
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
