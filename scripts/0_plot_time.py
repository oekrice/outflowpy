"""
    Script to combine Anna Frost Open Flux data with my outflow results.
    Some of this written by Anthony (2023, apparently)
"""
import numpy as np
import sys
import pandas as pd
from scipy.io import netcdf
import sys
if False:#(len(sys.argv) > 1):    # to avoid needing xwindows (allows ssh remote use)
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import time
import pickle
import os
import astropy.units as u
import portalocker
import subprocess
from scipy.interpolate import interp1d
import seaborn as sns

time_cadence = 30

class LockedDataset:
    """
    This is to check if the netcdf is being looked at by another process, and won't attempt to add more data if so. Should stop things becoming corrupted.
    Is apparently called a 'context manager'. Curious.
    """
    def __init__(self, path, mode="r", lock_path=None, lock_timeout=None):
        self.path = path
        self.mode = mode
        self.lock_path = lock_path or f"{path}.lock"
        self.lock_timeout = lock_timeout
        self._lock = None
        self._ds = None
        time.sleep(1.0)   #Make it thread safe perhaps maybe

    def __enter__(self):
        # Acquire the lock before opening NetCDF
        self._lock = portalocker.Lock(
            self.lock_path,
            mode="w",
            timeout=self.lock_timeout
        )
        self._lock.acquire()  # Wait until free

        self._ds = Dataset(self.path, self.mode, format = 'NETCDF4')
        return self._ds

    def __exit__(self, exc_type, exc_val, exc_tb):
        time.sleep(1.0)   #Make it thread safe perhaps maybe
        try:
            if self._ds is not None:
                self._ds.close()
        finally:
            if self._lock is not None:
                self._lock.release()
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass

class LockedLog:
    def __init__(self, path, mode="r+", lock_timeout=None):
        self.path = path
        self.mode = mode
        self.lock_timeout = lock_timeout
        self._log = None

    def __enter__(self):
        # Open the *log file itself* and lock it directly
        self._log = open(self.path, self.mode)
        portalocker.lock(self._log, portalocker.LOCK_EX)  # Blocks until lock acquired
        return self._log

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._log.flush()
        os.fsync(self._log.fileno())  # Ensure data is written to disk
        portalocker.unlock(self._log)   #Then unlock
        self._log.close()
        self._log = None

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


# Read smoothed sunspot number:
# -----------------------------
# https://www.sidc.be/SILSO/datafiles
# ssndat = np.loadtxt('SN_ms_tot_V2.0.csv', delimiter=';', dtype=str)
# t_sunspot = np.zeros(len(ssndat))
# sunspot_num = np.zeros(len(ssndat))
# for k, s in enumerate(ssndat):
#     t_sunspot[k] = s[2]
#     sunspot_num[k] = s[3]


# Read open flux from Anna Frost:
# -------------------------------
# https://link.springer.com/article/10.1007/s11207-022-02004-6#Sec18


frost_data = pd.read_csv("./data/frost_data.txt", sep='\\s+', comment = "%", header = None, on_bad_lines = 'skip')

years = frost_data[0].to_numpy()
months = frost_data[1].to_numpy()
days = frost_data[2].to_numpy()


dates = [datetime(int(years[i]), int(months[i]), int(days[i])) for i in range(len(years))]
tflux = np.array([toYearFraction(d) for d in dates])

oflux = frost_data[6]
ofluxmin = frost_data[7]
ofluxmax = frost_data[8]

tflux = tflux[oflux > 0]
ofluxmin = ofluxmin[oflux > 0]
ofluxmax = ofluxmax[oflux > 0]
oflux = oflux[oflux > 0]

#Sort out units
ofluxmin = ofluxmin*1e14*1e8
ofluxmax = ofluxmax*1e14*1e8
oflux = oflux*1e14*1e8

data_source = 1 #Set to 0 to get from diagnostic files, set to 1 to get it directly from the batch

use_hamilton_data = False

if len(sys.argv) > 1:
    if sys.argv[1] == 'ham8':
        use_hamilton_data = True

if use_hamilton_data:
    print('Copying diagnostic data from Hamilton. Requires ssh keys and the university VPN')
    #subprocess.run( ["scp -r -i",  f"C:/Users/eleph/.ssh/id_ed25519", f"vgjn10@hamilton8.dur.ac.uk:/home/vgjn10/projects/outflow_basics/diagnostic_data", "../"], check=True)
    if data_source == 0:
        os.system("scp -r vgjn10@hamilton8.dur.ac.uk:/home/vgjn10/projects/outflow_basics/diagnostic_data ../")
    else:
        os.system("scp -r vgjn10@hamilton8.dur.ac.uk:/home/vgjn10/projects/outflow_basics/batch_logs/*.txt ../batch_logs")

#Also let's look at correlations or something for each of them. Do interpolations of frost data, respective to each one. Within a given time bound, at least.
#Then use the most correlated to determine the optimum smoothing, then hopefully can scale using more/less MF stuff. We'll see.

time_interps = tflux[tflux > 2000]  #These are the time points to attempt interpolation from, as the Frost data goes back longer. I think this is fine for now.

#Do frost 'interpolation' here
frost_xs = tflux
frost_ys = oflux

frost_interp = interp1d(frost_xs, frost_ys, kind = 'linear')

oflux_ref = frost_interp(time_interps)

#plt.plot(time_interps, oflux_ref, c = 'black')

colors = plt.get_cmap('tab10').colors
colors = sns.color_palette('dark')

batch_ids = np.arange(5)
labels = ['PFSS', 'Outflow','a','b','c']

#Put in a thing to do some interpolation to get a value which should be bang on
interp_xs = []; interp_ys = []


def determine_error_bounds(dates, ofluxes, time_cadence = 27, error_bound = 0.9):
    #Function to take the messy daily data and transform into a nice smooth function with error bounds taking account of the wiggles.
    #Time cadence is the regularity and range of the data points being tested, and error bound is the percentile range with which the min and max values will be calculated
    print(dates, ofluxes)
    tmin = np.min(dates); tmax = np.max(dates)
    regular_dates = np.linspace(tmin, tmax, int((tmax-tmin)/(time_cadence/365.25)) + 1)
    oflux_means = []; oflux_mins = []; oflux_maxs = []; regular_dates_thin = []
    for date_test in regular_dates:
        local_min = date_test - 0.5*time_cadence/365.25
        local_max = date_test + 0.5*time_cadence/365.25
        date_mask = (dates >= local_min) & (dates <= local_max)
        local_ofluxes = ofluxes[date_mask]
        if len(local_ofluxes) > 0:
            oflux_means.append(np.mean(local_ofluxes))
            oflux_mins.append(np.percentile(local_ofluxes, 100*(1.0 - error_bound)))
            oflux_maxs.append(np.percentile(local_ofluxes, 100*(error_bound)))
            regular_dates_thin.append(date_test)

    return regular_dates_thin, oflux_means, oflux_mins, oflux_maxs

fig = plt.figure(figsize = (10,5))

overallcount = 0
for plot_num, batch_id in enumerate(batch_ids):
#Read in the data
    with LockedLog("./batch_logs/0_%d.txt" % batch_id, mode = "r+") as f:
        #This does now appear to be thread-safe, so we can be a bit smarter now and it'll run even faster and not get particularly baffled.
        #Look at the runs which satisfy the requirements and pick a random one from them.
        ofluxes = []; dates = []
        start = datetime.fromisoformat("1999-01-01T00:00:00")
        for run, info in enumerate(f, start=0):
            date = (start + timedelta(days=time_cadence*run))
            try:
                parts = info.strip().split('\t')
                openflux = float(parts[3].strip('{}'))
            except:
                print("This is corrupted, try again in a second or two rather than the longer time?")
                raise Exception('Log file is corrupted. Bugger.')

            if toYearFraction(date) < 1999.25:
                continue
            #Reasons to run: Either no start time, or no end time but start time is old
            if openflux > 5e21:
                ofluxes.append(openflux)
                dates.append(toYearFraction(date))
                overallcount += 1

        ofluxes = np.array(ofluxes); dates = np.array(dates)
        dates_mean, oflux_mean, oflux_min, oflux_max = determine_error_bounds(dates, ofluxes, 27)

    plt.plot(dates, ofluxes, color = colors[plot_num], linewidth = 0.1, linestyle = 'dashed')
    plt.plot(dates_mean, oflux_mean, color = colors[plot_num], label = labels[plot_num%10], linewidth = 1.0)
    plt.plot(dates_mean, oflux_min, color = colors[plot_num], linewidth = 0.5, linestyle = 'dashed')
    plt.plot(dates_mean, oflux_max, color = colors[plot_num], linewidth = 0.5, linestyle = 'dashed')


print('Overall percentage complete: %.1f' % (100*(overallcount/(3*27*365/time_cadence))))

#Plot open flux measurements
plt.plot(tflux, oflux, c = 'black', linewidth = 1.0)
plt.plot(tflux, ofluxmin, c = 'black', linewidth = 0.25)
plt.plot(tflux, ofluxmax, c = 'black', linewidth = 0.25)

#Plot outflow field predictions

plt.legend()
plt.xlabel('Year')
plt.ylabel('Open Flux at 1AU (Mx)')
plt.ylim(ymin = 0.0, ymax = 0.2e24)
plt.savefig('./plots/0_openflux_time_fric.png')
plt.show()



