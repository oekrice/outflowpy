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

from datetime import datetime

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

def download_hmi_mdi_crot(crot_number, source = None):
    r"""
    Downloads the raw HMI data with Carrington rotation number 'crot_number'.

    Parameters
    ----------
    crot_number : int
        Carrington rotation number
    source (optional): string
        If specified, ensures that the data comes from either 'MDI' or 'HMI'. This stops a mismatch if the maps are stitched together.
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

    if source == 'MDI':
        mdi_flag = True
    elif source == 'HMI':
        mdi_flag = False
    else:
        if crot_number < 2098:
            mdi_flag = True
        else:
            mdi_flag = False

    success = False
    while not success:
        try:
            c = drms.Client()
            if mdi_flag:
                seg = c.query(('mdi.synoptic_mr_polfil_96m[%4.4i]' % crot_number), seg='Br_polfil')
                data, header = fits.getdata('http://jsoc.stanford.edu' + seg.Br_polfil[0], header=True)
                data = _scale_mdi(data)   #Scale the magfield data from MDI so that it matches HMI. Using the correlation deduced by "https://link.springer.com/article/10.1007/s11207-012-9976-x"
            else:
                seg = c.query(('hmi.synoptic_mr_polfil_720s[%4.4i]' % crot_number), seg='Mr_polfil')
                data, header = fits.getdata('http://jsoc.stanford.edu' + seg.Mr_polfil[0], header=True)
                #np.savetxt(f'./tests/data/hmi_2210.txt', data)
            success = True
        except:
            print("Failed to find the Carrington Rotation database. This is likely due to an internet error caused by multiple threads, and so will try again shortly.")
            time.sleep(10.0)

    data = np.nan_to_num(data)

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
    #np.savetxt(f'./tests/data/mdi_2000_smooth.txt', data)

    return brm

def _find_crot_numbers(obs_time):
    r"""
    Outputs the three Carrington rotation numbers around the time of the requested observation.
    Obtains this data from the online data series, so it should stay up to date

    Parameters
    ----------
    obs_time : string
        String corresponding to the observation time. Format is YYYY-MM-DDThh:mm:ss.

    Returns
    -------
    crot_number : int
    Integer corresponding to the required Carrington rotation
    crot_fraction: float
    Fraction in time through this rotation. 0.5 would be precisely at the observation time, 0 is 13 days beforehand etc.
    """

    if datetime.fromisoformat(obs_time) < datetime.fromisoformat("2010-08-15T10:00:00"):
        mdi_flag = True
    else:
        mdi_flag = False

    success = False
    while not success:
        try:
            c = drms.Client()
            #Find the correct Carrington Rotation for this date.
            if mdi_flag:
                crot_times = c.query(('mdi.synoptic_mr_polfil_96m'), key = ["T_START","T_STOP","CAR_ROT"])
            else:
                crot_times = c.query(('hmi.synoptic_mr_polfil_720s'), key = ["T_START","T_STOP","CAR_ROT"])

            success = True
        except:
            print("Failed to find the Carrington Rotation database. This is likely due to an internet error caused by multiple threads, and so will try again shortly.")
            time.sleep(10.0)

    #"T_START" and "T_STOP" are the useful things
    start_times_raw = list(crot_times.pop("T_START"))
    for i in range(len(start_times_raw)):
        if start_times_raw[i][-6:-4] == "60":
            start_times_raw[i] = start_times_raw[i][:-6] + "00" + start_times_raw[i][-4:]
    end_times_raw = list(crot_times.pop("T_STOP"))
    for i in range(len(end_times_raw)):
        if end_times_raw[i][-6:-4] == "60":
            end_times_raw[i] = end_times_raw[i][:-6] + "00" + end_times_raw[i][-4:]

    start_times = [datetime.strptime(s.split('_TAI')[0], "%Y.%m.%d_%H:%M:%S") for s in start_times_raw]
    end_times = [datetime.strptime(s.split('_TAI')[0], "%Y.%m.%d_%H:%M:%S") for s in end_times_raw]
    crots = crot_times.pop("CAR_ROT")
    if np.max(end_times) < datetime.fromisoformat(obs_time):
        raise Exception('Failed to find a Carrington rotation corresponding to this observation time')

    time_index = np.searchsorted(end_times, datetime.fromisoformat(obs_time))
    rot = int(crots[time_index])   #This is the rotation at this time

    crot_fraction = (datetime.fromisoformat(obs_time) - start_times[time_index])/(end_times[time_index] - start_times[time_index])  #Distance through this Carrington rotation

    if rot < 1909 or rot > 2299:
        raise Exception(f"Data for this Carrington rotation ({rot}) does not exist")

    return rot, crot_fraction

def prepare_hmi_mdi_time(obs_time, ns_target, nphi_target, smooth = 0.0):
    r"""
    Downloads (without email etc.) the HMI or MDI data corresponding to the specified time
    Obtains three HMI/MDI magnetograms and stiches them together as appropriate.
    Then applies smoothing etc. as for the single crot script.

    Parameters
    ----------
    obs_time : string
        String corresponding to the observation time. Format is YYYY-MM-DDThh:mm:ss

    Returns
    -------
    data : sunpy.map.Map
    A sunpy map object corresponding to the requested time

    Notes
    -----
    Must be in the allowable range of Carrington rotations
    Outputs a sunpy map object
    """

    #Determine the rotation number and the fraction of time through this rotation
    crot_number, crot_fraction = _find_crot_numbers(obs_time)

    if crot_number < 2098:
        source = 'MDI'
    else:
        source = 'HMI'

    print(f"Obtaining data from {source}, downloading rotations, {crot_number-1, crot_number, crot_number+1}")
    #Download the respective sets of data
    brm  , header   = download_hmi_mdi_crot(crot_number  , source = source)
    brm_l, header_l = download_hmi_mdi_crot(crot_number+1, source = source)
    brm_r, header_r = download_hmi_mdi_crot(crot_number-1, source = source)

    brm_shift = 0.0*brm
    nphi = np.shape(brm_shift)[1]

    #Select the correct part of this map based on the fraction through this particular rotation
    #Shift amounts:
    #0.5 -> 0
    #0.0 -> nphi//2
    #1.0 -> -nphi//2
    brm3 = np.concatenate((brm_l, brm, brm_r), axis=1)

    ncells_shift = round(nphi*(0.5-crot_fraction))   #This tells it where to position the synoptic map
    ncells_start = nphi + ncells_shift

    brm_shift[:,:] = brm3[:, ncells_start:ncells_start + nphi]

    del(brm, brm_l, brm_r)

    data = sh_smooth(brm_shift, ns_target = ns_target, nphi_target = nphi_target, smooth = smooth)

    header = carr_cea_wcs_header(obs_time, np.shape(data.T))
    brm = sunpy.map.Map(data, header)

    print('Data successfully downloaded, smoothed, interpolated and balanced.')

    return brm



