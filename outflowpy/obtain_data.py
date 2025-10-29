"""
This file contains the code to download a specific Carrington rotation. 
For now I'll use the resampling method of pfsspy, although Anthony's one is superior so I'll switch to that later.
"""
import drms
from astropy.io import fits
import numpy as np
import sunpy.map

import outflowpy
from outflowpy.utils import carr_cea_wcs_header


def download_hmi_crot(crot_number):
    r"""
    Downloads (without email etc.) the HMI data matching the rotation number above

    Parameters
    ----------
    crot_number : int
        Carrington rotation number

    Returns
    -------
    map : sunpy.map.Map
    A sunpy map object corresponding to the requested rotation number

    Notes
    -----
    Must be in the allowable range of numbers (post-2010 ish). The first acceptable one is 2098.
    Outputs a list of THREE sunpy map objects -- those for the three crots near the required dates. 
    Left comes first, then right. So the order is [crot_number, crot_number+1, crot_number-1]
    """

    if crot_number < 2098 or crot_number > 2299:
        raise Exception("This Carrington rotation does not exist in the HMI database. Need to specify a rotation in the range 2097-2298 (as of July 2025).")

    #Get fits file locations
    try:
        c = drms.Client()
        seg = c.query(('hmi.synoptic_mr_polfil_720s[%4.4i]' % crot_number), seg='Mr_polfil')
        # segr = c.query(('hmi.synoptic_mr_polfil_720s[%4.4i]' % (crot_number-1)), seg='Mr_polfil')
        # segl = c.query(('hmi.synoptic_mr_polfil_720s[%4.4i]' % (crot_number+1)), seg='Mr_polfil')
    except:
        raise Exception(f'Failed to find the data source for Carrington Rotation {crot_number}')
    
    #Download these files
    try:
        data, header = fits.getdata('http://jsoc.stanford.edu' + seg.Mr_polfil[0], header=True)
        header = carr_cea_wcs_header(None, np.shape(data.T))
        brm = sunpy.map.Map(data, header)

        # data, header = fits.getdata('http://jsoc.stanford.edu' + segl.Mr_polfil[0], header=True)
        # header = fix_hmi_metadata(header)
        # brm_l = Map(data, header)

        # data, header = fits.getdata('http://jsoc.stanford.edu' + segr.Mr_polfil[0], header=True)
        # header = fix_hmi_metadata(header)
        # brm_r = Map(data, header)
    except:
        raise Exception(f'Failed to download the required Carrington rotation {crot_number}, required urls {'http://jsoc.stanford.edu' + segr.Mr_polfil[0], 'http://jsoc.stanford.edu' + segl.Mr_polfil[0], 'http://jsoc.stanford.edu' + segr.Mr_polfil[0]}')

    return brm
