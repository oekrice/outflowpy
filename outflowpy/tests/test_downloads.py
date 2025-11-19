import pytest
import outflowpy
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os

from outflowpy import obtain_data

test_data = pathlib.Path(__file__).parent / 'data'

@pytest.mark.parametrize(['crot_number','use_cached'],
                         [[2000,False],
                          [2210,False],
                          [2000,True],
                          [2210,True],
                          ])
def test_data_downloads(crot_number, use_cached):
    #Test to check that data is correctly downloaded for both HMI and MDI
    data, header = obtain_data.download_hmi_mdi_crot(crot_number, use_cached = use_cached)
    #Check if the data files exist. If so, compare them. If not, don't bother.
    if crot_number < 2098:
        if os.path.isfile(f'{test_data}/mdi_2000.txt'):
            expected_data = np.loadtxt(f'{test_data}/mdi_2000.txt')
            np.testing.assert_allclose(data, expected_data, atol=1e-10)
    else:
        if os.path.isfile(f'{test_data}/hmi_2210.txt'):
            expected_data = np.loadtxt(f'{test_data}/hmi_2210.txt')
            np.testing.assert_allclose(data, expected_data, atol=1e-10)

def test_nonexistent_download():
    #Tests to check that a reasonable exception is rasied if a Carrington rotation is specified outside the allowed range
    with pytest.raises(Exception) as excinfo:
        data, header = obtain_data.download_hmi_mdi_crot(1000)
    with pytest.raises(Exception) as excinfo:
        data, header = obtain_data.download_hmi_mdi_crot(3000)
    assert "This Carrington rotation does not exist" in str(excinfo.value)

@pytest.mark.parametrize(['smooth','ns','nphi'],
                         [[1.0,90,180],
                          [10.0,40,90]
                          ])

def test_smoothing(smooth, ns, nphi):
    #Tests the smoothing algorithm on the HMI map, ensuring dimensions and flux balance are accurate.
    if os.path.isfile(f'{test_data}/hmi_2210.txt'):
        data = np.loadtxt(f'{test_data}/hmi_2210.txt')
        smooth_data = obtain_data.sh_smooth(data, smooth*5e-2/nphi, ns, nphi)

        assert np.shape(smooth_data) == (ns, nphi)
        assert np.allclose(np.sum(smooth_data), 0.0)

def test_smoothing_errors():
    if os.path.isfile(f'{test_data}/mdi_2000.txt'):
        data = np.loadtxt(f'{test_data}/mdi_2000.txt')
        with pytest.raises(Exception) as excinfo:
            data, header = obtain_data.sh_smooth(data, 1e6, 90, 180)
        assert "Smoothing value is too high" in str(excinfo.value)

        with pytest.raises(Exception) as excinfo:
            data, header = obtain_data.sh_smooth(data, 1.0, 45, 180)
        assert "Attempting to interpolate onto a grid with an odd number of cells" in str(excinfo.value)

        with pytest.raises(Exception) as excinfo:
            data, header = obtain_data.sh_smooth(data[:20,:30], 1.0, 90, 180)
        assert "Attempting to downscale the imported data" in str(excinfo.value)

@pytest.mark.parametrize(['crot_number'],
                         [[2000],
                          [2210],
                          ])
def test_prepare_script(crot_number):
    #Tests that the headers and map creation works for both HMI and MDI
    #Checks against a previously-saved smoothed file
    data_map = obtain_data.prepare_hmi_mdi_crot(crot_number, 60, 120, smooth = 5e-2/180)
    if crot_number < 2098:
        expected_data = np.loadtxt(f'{test_data}/mdi_2000_smooth.txt')
    else:
        expected_data = np.loadtxt(f'{test_data}/hmi_2210_smooth.txt')
    np.testing.assert_allclose(data_map.data, expected_data, atol=1e-10)

@pytest.mark.parametrize(['obs_time','expected_crot','expected_frac'],
                         [["2005-04-13T10:00:00",2028,0.7099823230844563],
                          ["2020-04-13T10:00:00",2229,0.588236841937736],
                          ])
def test_crot_numbers(obs_time, expected_crot, expected_frac):
    #Checks that the correct rotation numbers and fractions are obtained for a given set of times
    rot, crot_fraction = obtain_data._find_crot_numbers(obs_time)
    assert rot == expected_crot
    assert crot_fraction == expected_frac

def test_time_errors():
    #Makes sure that times outside the reasonable range give expected test_smoothing_errors
    with pytest.raises(Exception) as excinfo:
        rot, crot_fraction = obtain_data._find_crot_numbers("2030-04-13T10:00:00")
    assert 'Failed to find a Carrington rotation corresponding to this observation time' in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:
        rot, crot_fraction = obtain_data._find_crot_numbers("1900-04-13T10:00:00")
    assert "Failed to find a Carrington rotation corresponding to this observation time"  in str(excinfo.value)


