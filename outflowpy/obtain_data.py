"""
This file contains the code to download a specific Carrington rotation. 
For now I'll use the resampling method of pfsspy, although Anthony's one is superior so I'll switch to that later.
"""
import sys 

import drms
from astropy.io import fits
import numpy as np
import sunpy.map

import outflowpy
from outflowpy.utils import carr_cea_wcs_header
from scipy.interpolate import RectBivariateSpline

import matplotlib.pyplot as plt
import scipy.linalg as la

def _crudely_downscale(data, downscale_factor = 4):
    """
    Function to speed up the smoothing algorithm by downscaling the input data by a factor of 'downscale_factor' in each dimension
    Set to 4 by default.
    """
    nx = np.shape(data)[0]; ny = np.shape(data)[1]
    if nx%downscale_factor != 0 or ny%downscale_factor != 0:
        raise Exception("Attempting to downscale the imported data by a factor which doesn't work. Try a small power of 2 if that's an option, or not at all.")
    nxl = nx//downscale_factor; nyl = ny//downscale_factor
    data_downscale = data.reshape(nxl, downscale_factor, nyl, downscale_factor).mean(axis=(1,3))
    return data_downscale

def _correct_flux_multiplicative(f):
    """
    Corrects the flux balance in the map f (assumes that cells have equal area).
    """
    # Compute positive and negative fluxes:
    ipos = f > 0
    ineg = f < 0
    fluxp = np.abs(np.sum(f[ipos]))
    fluxn = np.abs(np.sum(f[ineg]))

    # Rescale both polarities to mean:
    fluxmn = 0.5*(fluxn + fluxp)
    f1 = f.copy()
    f1[ineg] *= fluxmn/fluxn
    f1[ipos] *= fluxmn/fluxp

    return f1

def _sh_smooth(raw_data, smooth, ns_target, nphi_target, cutoff=10):
    """
    Parameters
    ----------
    f : array
        Input array representing the radial magnetic field on the solar surface. Designed to be an import from HMI or equivalent
    smooth : real
        Smoothing coefficient. Set to zero to include everything, and increase to increase the amount of blurring
    cutoff : integer
        The largest value of smooth*l(l+1) that will be considered. Everything else will be set to zero.

    Returns
    -------
    f : array
        The smoothed version of the input array.

    Notes
    -------
         This implementation uses discrete eigenfunctions (in latitude) instead of Plm.

    Parameters:
    smooth -- coefficient of filter exp(-smooth * lam) [set cutoff=0 to include all eigenvalues. This is quicker for small matrices.]
    cutoff -- largest value of smooth*lam to include [so 10 means ignore blm multiplied by exp(-10)]
    """

    print('Initial Data Shape', np.shape(raw_data))
    #Initially apply a crude downscale so the smoothing can happen at a reasonable pace.
    data = _crudely_downscale(raw_data)
    print('Downscaled Data Shape', np.shape(data))

    print('Smoothing data...')
    #Establish the grid on which to do the smoothing
    ns_smooth = np.size(data, axis=0)
    np_smooth = np.size(data, axis=1)
    ds_smooth = 2.0/ns_smooth 
    dp_smooth = 2*np.pi/np_smooth 
    sc_smooth = np.linspace(-1 + 0.5*ds_smooth , 1 - 0.5*ds_smooth, ns_smooth )
    sg_smooth = np.linspace(-1, 1, ns_smooth +1) 
    pc_smooth = np.linspace(-np.pi + 0.5*dp_smooth, np.pi - 0.5*dp_smooth, np_smooth)
    # Prepare tridiagonal matrix:
    Fp = sg_smooth * 0  # Lp/Ls on p-ribs
    Fp[1:-1] = np.sqrt(1 - sg_smooth[1:-1] ** 2) / (np.arcsin(sc_smooth[1:]) - np.arcsin(sc_smooth[:-1])) * dp_smooth
    Vg = Fp / ds_smooth / dp_smooth
    Fs = ((np.arcsin(sg_smooth[1:]) - np.arcsin(sg_smooth[:-1])) / np.sqrt(1 - sc_smooth ** 2) / dp_smooth)  # Ls/Lp on s-ribs
    Uc = Fs / ds_smooth / dp_smooth
    # - create off-diagonal part of the matrix:
    off_diag = -Vg[1:ns_smooth]
    # - terms required for m-dependent part of matrix:
    mu = np.fft.fftfreq(np_smooth)
    mu = 4 * np.sin(np.pi * mu) ** 2
    diag1 = Vg[:ns_smooth] + Vg[1:ns_smooth+1]

    # FFT in phi of photospheric distribution at each latitude:
    fhat = np.fft.rfft(data, axis=1)

    # Loop over azimuthal modes (positive m):
    nm = np_smooth//2 + 1
    blm = np.zeros((ns_smooth), dtype="complex")
    fhat1 = np.zeros((ns_smooth, nm), dtype="complex")
    for m in range(nm):
        # - set diagonal terms of matrix:
        diag = diag1 + Uc[:ns_smooth] * mu[m]
        # - compute eigenvectors Q_{lm} and eigenvalues lam_{lm}:
        #   (note that matrix is symmetric tridiag so use special solver)
        if cutoff > 0 and smooth > 0:
            # - ignore contributions with eigenvalues too large to contribute after smoothing:
            lamax = cutoff/smooth
            lam, Q = la.eigh_tridiagonal(diag, off_diag, select="v", select_range=(0,lamax))
            nsm1 = len(lam) # [full length would be nsm]
        else:
            #Calculate everything...
            lam, Q = la.eigh_tridiagonal(diag, off_diag)
            nsm1 = ns_smooth
        # - find coefficients of eigenfunction expansion:
        for l in range(nsm1): 
            blm[l] = np.dot(Q[:,l], fhat[:,m])
            # - apply filter [the eigenvalues should be a numerical approx of lam = l*(l+1)]:
            blm[l] *= np.exp(-smooth*lam[l])
        # - invert the latitudinal transform:
        fhat1[:,m] = np.dot(blm[:nsm1], Q.T)

    # Invert the FFT in longitude:
    data_smooth = np.real(np.fft.irfft(fhat1, axis=1))
        
    print('Interpolating to target grid...')
    ds_target = 2.0/ns_target
    dp_target = 2*np.pi/nphi_target
    sc_target = np.linspace(-1 + 0.5*ds_target , 1 - 0.5*ds_target, ns_target )
    pc_target = np.linspace(-np.pi + 0.5*dp_target, np.pi - 0.5*dp_target, nphi_target)
    
    bri = RectBivariateSpline(sc_smooth, pc_smooth, data_smooth[:,:])
    data_target = np.zeros((ns_target, nphi_target))
    for i in range(ns_target):
        data_target[i,:] = bri(sc_target[i], pc_target).flatten()
    del(data_smooth, bri)

    data_target = _correct_flux_multiplicative(data_target)

    return data_target


def prepare_hmi_crot(crot_number, ns_target, nphi_target, smooth = 0.0):
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
        print('Attempting to download data...')
        c = drms.Client()
        seg = c.query(('hmi.synoptic_mr_polfil_720s[%4.4i]' % crot_number), seg='Mr_polfil')
        # segr = c.query(('hmi.synoptic_mr_polfil_720s[%4.4i]' % (crot_number-1)), seg='Mr_polfil')
        # segl = c.query(('hmi.synoptic_mr_polfil_720s[%4.4i]' % (crot_number+1)), seg='Mr_polfil')
    except:
        raise Exception(f'Failed to find the data source for Carrington Rotation {crot_number}')
    
    print('Data downloaded...')
    
    try:
        data, header = fits.getdata('http://jsoc.stanford.edu' + seg.Mr_polfil[0], header=True)

        #Add smoothing in here
        data = _sh_smooth(data, ns_target = ns_target, nphi_target = nphi_target, smooth = smooth)

        header = carr_cea_wcs_header(None, np.shape(data.T))
        brm = sunpy.map.Map(data, header)

        print('Data successfully downloaded, smoothed, interpolated and balanced.')
        # data, header = fits.getdata('http://jsoc.stanford.edu' + segl.Mr_polfil[0], header=True)
        # header = fix_hmi_metadata(header)
        # brm_l = Map(data, header)

        # data, header = fits.getdata('http://jsoc.stanford.edu' + segr.Mr_polfil[0], header=True)
        # header = fix_hmi_metadata(header)
        # brm_r = Map(data, header)
    except:
        raise Exception(f'Failed to download the required Carrington rotation {crot_number}, required urls {'http://jsoc.stanford.edu' + seg.Mr_polfil[0]}')


    return brm
