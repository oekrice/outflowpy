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

def sh_smooth(raw_data, smooth, ns_target, nphi_target, cutoff=10):
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

    if ns_target%2 == 1 or nphi_target == 1:
        raise Exception("Attempting to interpolate onto a grid with an odd number of cells in at least one dimension. This will likely cause errors, so don't do it.")

    if smooth > 0.0:
        if cutoff/smooth <= 1.5:
            raise Exception("Smoothing value is too high, try reducing it.")

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

    if np.sum(np.abs(data_target)) < 1e-10:
        raise Exception('Smoothing has resulted in a zero map. Try reducing the smoothing factor?')

    data_target = _correct_flux_multiplicative(data_target)

    return data_target

def _scale_mdi(mdi_input):
    #Converts each pixel of the MDI magnetogram so it matches HMI (which I'm assuming is more accurate, but that's up for debate).
    #Doesn't apply the different corrections at the poles, but they're not too different
    #Will now try to make this work more properly. The strong field behaves differently to this, and we can probably deal with that rather than ignoring it...
    #Treat regions with strenth of more than 600 Mx/cm^2 differently, as these are 'strong-field'.
    def scale_pixel(value):
        strong_value = (mdi_input - 10.2) /1.31
        weak_value = (mdi_input + 0.18) /1.4
        prop = np.clip((value - 400) / 200.0, 0.0, 1.0)

        return prop*strong_value + (1. - prop)*weak_value

    mdi_input = scale_pixel(mdi_input)

    return mdi_input

def download_hmi_mdi_crot(crot_number):
    r"""
    Downloads the raw HMI data with Carrington rotation number 'crot_number'.

    Parameters
    ----------
    crot_number : int
        Carrington rotation number

    Returns
    -------
    data : array
    Array corresponding to the magnetic field strength on the solar surface

    header: sunpy.header object
    Object containing some metadata about the downloaded data. May not be quite accurate.

    Notes
    -----
    If the specified rotation is less than 2098, the data downloaded willl be from MDI. If not, HMI.
    This information will hopefully be contained within the header so it can be 'corrected' in due course.
    """

    if crot_number < 1909 or crot_number > 2299:
        raise Exception("This Carrington rotation does not exist in the MDI/HMI database. Need a rotation in range 2097-2298 (as of July 2025).")

    if crot_number < 2098:
        mdi_flag = True
    else:
        mdi_flag = False

    c = drms.Client()
    if mdi_flag:
        seg = c.query(('mdi.synoptic_mr_polfil_96m[%4.4i]' % crot_number), seg='Br_polfil')
        data, header = fits.getdata('http://jsoc.stanford.edu' + seg.Br_polfil[0], header=True)
        data = _scale_mdi(data)   #Scale the magfield data from MDI so that it matches HMI. Using the correlation deduced by "https://link.springer.com/article/10.1007/s11207-012-9976-x"
    else:
        seg = c.query(('hmi.synoptic_mr_polfil_720s[%4.4i]' % crot_number), seg='Mr_polfil')
        data, header = fits.getdata('http://jsoc.stanford.edu' + seg.Mr_polfil[0], header=True)
        #np.savetxt(f'./tests/data/hmi_2210.txt', data)

    return data, header

def prepare_hmi_mdi_crot(crot_number, ns_target, nphi_target, smooth = 0.0):
    r"""
    Downloads (without email etc.) the HMI or MDI data matching the rotation number above

    Parameters
    ----------
    crot_number : int
        Carrington rotation number

    Returns
    -------
    data : sunpy.map.Map
    A sunpy map object corresponding to the requested rotation number

    Notes
    -----
    Must be in the allowable range of Carrington rotations
    Outputs a sunpy map object
    """
    
    data, header = download_hmi_mdi_crot(crot_number)
    print('Data downloaded...')
    #Add smoothing in here
    data = sh_smooth(data, ns_target = ns_target, nphi_target = nphi_target, smooth = smooth)

    header = carr_cea_wcs_header(None, np.shape(data.T))
    brm = sunpy.map.Map(data, header)

    print('Data successfully downloaded, smoothed, interpolated and balanced.')

    return brm
