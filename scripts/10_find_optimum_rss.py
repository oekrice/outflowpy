#Script to find optimum pfss rss heights and optimum outflow solutions for a given time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from scipy.optimize import root_scalar, minimize_scalar
from scipy import interpolate
from datetime import datetime, timedelta
import time
from scipy.interpolate import interp1d
import outflowpy
from scipy.optimize import newton

colors = sns.color_palette('tab20')
obs_time = "2017-08-21T00:00:00"


def find_frost_target(obs_time):
    #Finds the target flux from the Frost data, for this time, in Mx
    frost_data = pd.read_csv("./data/frost_data.txt", sep='\\s+', comment = "%", header = None, on_bad_lines = 'skip')
    years = frost_data[0].to_numpy()
    months = frost_data[1].to_numpy()
    days = frost_data[2].to_numpy()

    dates = np.array([datetime(int(years[i]), int(months[i]), int(days[i])) for i in range(len(years))])

    oflux = frost_data[6]
    dates = dates[oflux > 0]
    oflux = oflux[oflux > 0]
    oflux = oflux*1e14*1e8


    def toYearFraction(date):

        #Returns the year plus a fraction of the year, as a float. Very good.

        def sinceEpoch(date): # returns seconds since epoch
            return time.mktime(date.timetuple())
        s = sinceEpoch

        year = date.year
        startOfThisYear = datetime(year=year, month=1, day=1)
        startOfNextYear = datetime(year=year+1, month=1, day=1)


        yearElapsed = s(date) - s(startOfThisYear)
        yearDuration = s(startOfNextYear) - s(startOfThisYear)
        fraction = yearElapsed/yearDuration

        return date.year + fraction

    frost_years = np.array([toYearFraction(d) for d in dates])

    frost_interp = interp1d(frost_years, oflux, kind = 'linear')

    oflux_ref = frost_interp(toYearFraction(datetime.fromisoformat(obs_time)))

    return oflux_ref

def find_pfss_openflux(hmi_map, rss):
    #Just finds the pfss open flux, as it says
    nrho = 120
    outflow_in = outflowpy.Input(hmi_map, nrho, rss, mf_constant = 0.0)
    outflow_out = outflowpy.outflow_fortran(outflow_in)

    rsun_cm = 6.957e10
    surface_area = 4*np.pi*(np.exp(outflow_in.grid.rg[-1])*rsun_cm)**2

    oflux_pfss = np.sum(np.abs(outflow_out.br)[:,:,-1])*surface_area/(nphi*ns)
    surface_area = 4*np.pi*(np.exp(outflow_in.grid.rg[0])*rsun_cm)**2
    return oflux_pfss


oflux_ref = find_frost_target(obs_time)
ns = 180; nphi = 360
hmi_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = True)

print('Target reference', oflux_ref)
#oflux_pfss = find_pfss_openflux(hmi_map, 2.5)

def minimise_function_pfss(rss):
    oflux_pfss = find_pfss_openflux(hmi_map, np.exp(rss))

    print('Current iteration', np.exp(rss), (oflux_pfss - oflux_ref)/1e22)
    return (oflux_pfss - oflux_ref)/1e22


def optimise_pfss():
    #Use Newton's method to find the ideal rss for this time
    rss_ideal = newton(minimise_function_pfss, 0.5)
    print('Ideal source surface height', np.exp(rss_ideal))
#optimise_pfss()


def find_outflow_openflux(hmi_map,t0):
    #Just finds the pfss open flux, as it says
    nrho = 120
    outflow_in = outflowpy.Input(hmi_map, nrho, 5.0, mf_constant = 5e-17, corona_temp = t0*1e6)
    outflow_out = outflowpy.outflow_fortran(outflow_in)

    rsun_cm = 6.957e10
    surface_area = 4*np.pi*(np.exp(outflow_in.grid.rg[-1])*rsun_cm)**2

    oflux_pfss = np.sum(np.abs(outflow_out.br)[:,:,-1])*surface_area/(nphi*ns)
    surface_area = 4*np.pi*(np.exp(outflow_in.grid.rg[0])*rsun_cm)**2
    return oflux_pfss

def minimise_function_outflow(t0):
    oflux_outflow = find_outflow_openflux(hmi_map, t0)

    print('Current iteration', t0*1e6, (oflux_outflow - oflux_ref)/1e22)
    return (oflux_outflow - oflux_ref)/1e22


def optimise_outflow():
    #Use Newton's method to find the ideal rss for this time
    t_ideal = newton(minimise_function_outflow, 2.0)
    print('Ideal temperature', t_ideal * 1e6)

optimise_outflow()



"""
"results:
rss = 1.5898
temp = 2.6255MK
"""
