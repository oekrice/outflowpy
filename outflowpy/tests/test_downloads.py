import pytest
import outflowpy
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from outflowpy import obtain_data

test_data = pathlib.Path(__file__).parent / 'data'

@pytest.mark.parametrize('crot_number',
                         [2000,
                          2210,
                          ])
def test_data_downloads(crot_number):
    #Test to check that data is correctly downloaded for both HMI and MDI
    data, header = obtain_data.download_hmi_mdi_crot(crot_number)
    if crot_number < 2098:
        expected_data = np.loadtxt(f'{test_data}/mdi_2000.txt')
    else:
        expected_data = np.loadtxt(f'{test_data}/hmi_2210.txt')

    np.testing.assert_allclose(data, expected_data, atol=1e-13, rtol=0)

def test_nonexistent_download():
    #Tests to check that a reasonable exception is rasied if a Carrington rotation is specified outside the allowed range
    with pytest.raises(Exception) as excinfo:
        data, header = obtain_data.download_hmi_mdi_crot(1000)

    assert "This Carrington rotation does not exist" in str(excinfo.value)

@pytest.mark.parametrize(['smooth','ns','nphi'],
                         [[1.0,90,180],
                          [10.0,40,90]
                          ])
def test_smoothing(smooth, ns, nphi):
    #Tests the smoothing algorithm on the HMI map, ensuring dimensions and flux balance are accurate.
    data = np.loadtxt(f'{test_data}/hmi_2210.txt')
    smooth_data = obtain_data.sh_smooth(data, smooth*5e-2/nphi, ns, nphi)

    assert np.shape(smooth_data) == (ns, nphi)
    assert np.allclose(np.sum(smooth_data), 0.0)

def test_smoothing_errors():
    data = np.loadtxt(f'{test_data}/hmi_2210.txt')
    with pytest.raises(Exception) as excinfo:
        data, header = obtain_data.sh_smooth(data, 1e6, 90, 180)
    assert "Smoothing value is too high" in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:
        data, header = obtain_data.sh_smooth(data, 1.0, 45, 180)
    assert "Attempting to interpolate onto a grid with an odd number of cells" in str(excinfo.value)

@pytest.mark.parametrize(['crot'],
                         [[2000],
                          [2210],
                          ])
def test_prepare_script(crot):
    #Tests that the headers and map creation works for both HMI and MDI
    obtain_data.prepare_hmi_mdi_crot(crot, 90, 180, smooth = 5e-2/180)




