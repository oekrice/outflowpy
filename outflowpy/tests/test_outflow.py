#Tests for the outflow calculations, using the smoothed sample data
#Compares the python and fortran versions against the pfsspy solution and solenoidal condition  etc.
import pathlib
import numpy
import sunpy
import pytest
import numpy as np
import outflowpy
from astropy.time import Time
import matplotlib.pyplot as plt

test_data = pathlib.Path(__file__).parent / 'data'


@pytest.mark.parametrize('test_fname, nrho, rss',
                         [['dipole_smooth.txt',30,2.5],
                          ['hmi_2210_smooth.txt',60,3.5]
                          ])
def test_potential_fields(test_fname, nrho, rss):
    #Tests the three methods of calculating potential fields against each other.
    #Uses an MDI map, an HMI map and a analytic dipole solution
    def find_oflux_profile(outflow_out):
        br_out = outflow_out.br
        ofluxes = np.zeros(np.shape(br_out)[2])
        for ri in range(np.shape(br_out)[2]):
            surface_area = 4*np.pi*np.exp(outflow_in.grid.rg[ri])**2
            oflux = np.sum(np.abs(br_out)[:,:,ri])*surface_area
            ofluxes[ri] = oflux
        return ofluxes

    br = np.loadtxt(f'{test_data}/{test_fname}')

    header = outflowpy.utils.carr_cea_wcs_header(Time('2020-1-1'), br.shape)
    input_map = sunpy.map.Map((br.T, header))

    outflow_in = outflowpy.Input(input_map, nrho, rss, mf_constant = 0.0)

    #Make the three output fields
    pfss_out = outflowpy.pfss(outflow_in)
    python_out = outflowpy.outflow(outflow_in)
    fortran_out = outflowpy.outflow_fortran(outflow_in)

    pfss_profile = find_oflux_profile(pfss_out)
    python_profile = find_oflux_profile(python_out)
    fortran_profile = find_oflux_profile(fortran_out)

    np.testing.assert_allclose(fortran_out.br, python_out.br, atol=1e-10, rtol=0)   #Ensure the python and fortran match perfectly
    np.testing.assert_allclose(pfss_profile, fortran_profile, rtol = 1e-2)            #Ensure the open flux profiles of the pfss and outflows are close
    np.testing.assert_allclose(fortran_out.br, pfss_out.br, atol = 1e-2*np.max(np.abs(fortran_out.br)))   #Ensure the overall field is close between pfsspy and outflowpy

    def calculate_divergence(outflow_in, output):
        #Calculate areas for the solenoidal condition and add up
        br = np.swapaxes(output.br, 0, 2)
        bs = np.swapaxes(output.bs, 0, 2)
        bp = np.swapaxes(output.bp, 0, 2)

        r,s,p = np.meshgrid(outflow_in.grid.rg,outflow_in.grid.sg,outflow_in.grid.pg,indexing='ij')
        Sr = np.exp(2*r[:,1:,1:])*outflow_in.grid.ds*outflow_in.grid.dp
        Ss = 0.5 * (np.exp(2*r[1:,:,1:]) - np.exp(2*r[:-1,:,1:])) * np.sqrt(np.ones((outflow_in.grid.nr,outflow_in.grid.ns+1,outflow_in.grid.nphi))-s[:-1,:,1:]**2) * outflow_in.grid.dp
        Sp = 0.5 * (np.exp(2*r[1:,1:,:]) - np.exp(2*r[:-1,1:,:])) * (np.arcsin(s[1:,1:,:])-np.arcsin(s[1:,:-1,:]))

        div_field = np.zeros((outflow_in.grid.nr,outflow_in.grid.ns,outflow_in.grid.nphi))
        div_field += br[1:,:,:]*Sr[1:,:,:] - br[:-1,:,:]*Sr[:-1,:,:]
        div_field += bs[:,1:,:]*Ss[:,1:,:] - bs[:,:-1,:]*Ss[:,:-1,:]
        div_field += bp[:,:,1:]*Sp[:,:,1:] - bp[:,:,:-1]*Sp[:,:,:-1]

        return div_field

    div_pfss = calculate_divergence(outflow_in, pfss_out)
    div_python = calculate_divergence(outflow_in, python_out)
    div_fortran = calculate_divergence(outflow_in, fortran_out)

    np.testing.assert_allclose(div_pfss, 0.0*div_pfss, atol = 1e-10)
    np.testing.assert_allclose(div_python, 0.0*div_python, atol = 1e-10)
    np.testing.assert_allclose(div_fortran, 0.0*div_fortran, atol = 1e-10)




