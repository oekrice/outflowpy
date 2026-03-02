import numpy as np
import pytest
from astropy.time import Time
from sunpy.map import Map

import pathlib
import outflowpy

test_data = pathlib.Path(__file__).parent / 'data'

@pytest.fixture
def zero_map():
    # Test a completely zero input
    ns = 30
    nphi = 20
    nr = 10
    rss = 2.5
    br = np.zeros((nphi, ns))
    header = outflowpy.utils.carr_cea_wcs_header(Time('1992-12-21'), br.shape)
    input_map = Map((br.T, header))

    input = outflowpy.Input(input_map, nr, rss)
    output = outflowpy.pfss(input)
    return input, output

@pytest.fixture
def dipole_map():
    ntheta = 30
    nphi = 20

    phi = np.linspace(0, 2 * np.pi, nphi)
    theta = np.linspace(-np.pi / 2, np.pi / 2, ntheta)
    theta, phi = np.meshgrid(theta, phi)

    def dipole_Br(r, theta):
        return 2 * np.sin(theta) / r**3

    br = dipole_Br(1, theta)
    header = outflowpy.utils.carr_cea_wcs_header(Time('1992-12-21'), br.shape)
    header['bunit'] = 'nT'
    return Map((br.T, header))


@pytest.fixture
def dipole_result(dipole_map):
    nr = 10
    rss = 2.5

    input = outflowpy.Input(dipole_map, nr, rss)
    output = outflowpy.pfss(input)
    return input, output


@pytest.fixture
def gong_map():
    """
    Automatically download and unzip a sample GONG synoptic map.
    """
    return outflowpy.sample_data.get_gong_map()


@pytest.fixture
def adapt_map():
    """
    Automatically download and unzip a sample GONG synoptic map.
    """
    return outflowpy.sample_data.get_adapt_map()

@pytest.fixture
def hmi_map():
    """
    Loads the HMI data for rotation 2210 directly from 'data', where it is saved as a .txt array
    """
    data = np.loadtxt(f'{test_data}/hmi_2210.txt')
    return data

@pytest.fixture
def mdi_map():
    """
    Loads the MDI data for rotation 2000 directly from 'data', where it is saved as a .txt array
    """
    data = np.loadtxt(f'{test_data}/mdi_2000.txt')
    return data

