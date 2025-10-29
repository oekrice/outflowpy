import os

import astropy.constants as const
import astropy.coordinates as coord
import astropy.io
import astropy.time
import numpy as np
import sunpy.map
import sunpy.time
from astropy import units as u
from astropy.wcs import WCS

import outflowpy.map


#The below three functions check the downloaded data is cylindrical equal area projection
def is_cea_map(m, error=False):
    """
    Returns `True` if *m* is in a cylindrical equal area projeciton.

    Parameters
    ----------
    m : sunpy.map.GenericMap
    error : bool
        If `True`, raise an error if *m* is not a CEA projection.
    """
    return _check_projection(m, 'CEA', error=error)

def _get_projection(m, i):
    return m.meta[f'ctype{i}'][5:8]

def _check_projection(m, proj_code, error=False):
    for i in ('1', '2'):
        proj = _get_projection(m, i)
        if proj != proj_code:
            if error:
                raise ValueError(f'Projection type in CTYPE{i} keyword '
                                 f'must be {proj_code} (got "{proj}")')
            return False
    return True

#The below checks that this map covers the full sun
def is_full_sun_synoptic_map(m, error=False):
    """
    Returns `True` if *m* is a synoptic map spanning the solar surface.

    Parameters
    ----------
    m : sunpy.map.GenericMap
    error : bool
        If `True`, raise an error if *m* does not span the whole solar surface.
    """
    projection = _get_projection(m, 1)
    checks = {'CEA': _is_full_sun_cea,
              'CAR': _is_full_sun_car}
    if projection not in checks.keys():
        raise NotImplementedError('is_full_sun_synoptic_map is only '
                                  'implemented for '
                                  f'{[key for key in checks.keys()]} '
                                  'projections.')
    return checks[projection](m, error)

def _is_full_sun_cea(m, error=False):
    shape = m.data.shape

    dphi = m.scale.axis1
    phi = shape[1] * u.pix * dphi
    if not np.allclose(np.abs(phi), 360 * u.deg, atol=0.1 * u.deg):
        if error:
            raise ValueError('Number of points in phi direction times '
                             'CDELT1 must be close to 360 degrees. '
                             f'Instead got {dphi} x {shape[1]} = {phi}')
        return False

    dtheta = m.scale.axis2
    theta = shape[0] * u.pix * dtheta * np.pi / 2
    if not np.allclose(theta, 180 * u.deg, atol=0.1 * u.deg):
        if error:
            raise ValueError('Number of points in theta direction times '
                             'CDELT2 times pi/2 must be close to '
                             '180 degrees. '
                             f'Instead got {dtheta} x {shape[0]} * pi/2 '
                             f'= {theta}')
        return False
    return True

def _is_full_sun_car(m, error=False):
    shape = m.data.shape

    dphi = m.scale.axis1
    phi = shape[1] * u.pix * dphi
    if not np.allclose(np.abs(phi), 360 * u.deg, atol=0.1 * u.deg):
        if error:
            raise ValueError('Number of points in phi direction times '
                             'CDELT1 must be close to 360 degrees. '
                             f'Instead got {dphi} x {shape[0]} = {phi}')
        return False

    dtheta = m.scale.axis2
    theta = shape[0] * u.pix * dtheta
    if not np.allclose(theta, 180 * u.deg, atol=0.1 * u.deg):
        if error:
            raise ValueError('Number of points in theta direction times '
                             'CDELT2 must be close to 180 degrees. '
                             f'Instead got {dtheta} x {shape[0]} = {theta}')
        return False
    return True

@u.quantity_input
def carr_cea_wcs_header(dtime, shape, *, map_center_longitude=0*u.deg):
    """
    Create a Carrington WCS header for a Cylindrical Equal Area (CEA)
    projection. See [1]_ for information on how this is constructed.

    Parameters
    ----------
    dtime : datetime, None
        Datetime to associate with the map.
    shape : tuple
        Map shape. The first entry should be number of points in longitude, the
        second in latitude.
    map_center_longitude : astropy.units.Quantity
        Change the world coordinate longitude of the central image pixel to allow
        for different roll angles of the Carrington map. Default to 0 deg. Must
        be supplied with units of `astropy.units.deg`

    References
    ----------
    .. [1] W. T. Thompson, "Coordinate systems for solar image data",
       https://doi.org/10.1051/0004-6361:20054262
    """
    # If datetime is None, put in a dummy value here to make
    # make_fitswcs_header happy, then strip it out at the end
    obstime = dtime or astropy.time.Time('2000-1-1')

    frame_out = coord.SkyCoord(
        map_center_longitude, 0 * u.deg, const.R_sun, obstime=obstime,
        frame="heliographic_carrington", observer='self')
    # Construct header
    header = sunpy.map.make_fitswcs_header(
        [shape[1], shape[0]], frame_out,
        scale=[360 / shape[0],
               180 / shape[1]] * u.deg / u.pix,
        reference_pixel=[(shape[0] / 2) - 0.5,
                         (shape[1] / 2) - 0.5] * u.pix,
        projection_code="CEA")

    # Fix CDELT for lat axis
    header['CDELT2'] = (180 / np.pi) * (2 / shape[1])
    # pop out the time if it isn't supplied
    return header
