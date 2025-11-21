#Script to run a series of outflow calculations, with each series varying in time
#Key here is to allow multiple processes by logging which have been sucessful and reattempting those which have not
#The only data we care about here is the open flux, so only need to save that


import sys, random, time
import os
from datetime import datetime, timedelta
from pathlib import Path
import outflowpy
import numpy as np
import portalocker
import random

def run_batch(batch_id, mf_constant = 5e-17, corona_temp = 2e6, time_cadence = 30):

    delay_time = 1*random.uniform(0.,0.1)
    time.sleep(delay_time) #To offset processors when they start all at the same time (which is undesirable)
    
    print(f'Running batch with ID {batch_id} on this process')
    #This can exist as a wrapper for the individual run_outflow functions (to which arguments can be passed.)
    #The ID refers to a log file to check which of them have already been done, but otherwise contains nothing of use.
    #Log file will contain an ID for the run itself, a start time and an end time (somehow)
    Path("batch_logs").mkdir(parents=True, exist_ok=True)

    time_cadence = time_cadence   #How frequently to do a run (in days)
    nruns = int(9500/time_cadence)    #Number of runs in the batch
    timeout_limit = 60   #How long to wait before reattempting a run, in SECONDS

    start = datetime.fromisoformat("1999-01-01T00:00:00")
    print('Total number of runs planned:', nruns)

    #Set up batch logs, if they don't exist (to allow for multiple processes etc.)
    if not Path("batch_logs/0_%d.txt" % batch_id).exists():
        with LockedLog("batch_logs/0_%d.txt" % batch_id, mode = "w") as f:
            for run in range(nruns):
                f.write("{%08d}\t{%s}\t{%s}\t{%f}\n" % (run, "2000-01-01T00:00:00", "2000-01-01T00:00:00",0.0))
    else:
        time.sleep(1.0)   #Make sure the other process has time to write the blank ones
    #Determine which run to complete based on what has been done already
    go = True
    while go:
        now = datetime.now()
        done_count = 0
        #Determine next run to attempt
        next_id = -1
        with LockedLog("batch_logs/0_%d.txt" % batch_id, mode = "r+") as f:
            #This does now appear to be thread-safe, so we can be a bit smarter now and it'll run even faster and not get particularly baffled.
            #Look at the runs which satisfy the requirements and pick a random one from them.
            possible_runs = []; started_runs = []
            for try_run, info in enumerate(f, start=0):
                try:
                    parts = info.strip().split('\t')
                    start_time = datetime.fromisoformat(parts[1].strip('{}'))
                    end_time = datetime.fromisoformat(parts[2].strip('{}'))
                except:
                    #This is corrupted, try again in a second or two rather than the longer time?
                    print('Corrupt')
                    pass
                #Reasons to run: Either no start time, or no end time but start time is old
                if start_time > datetime.fromisoformat("2001-01-01T00:00:00"):
                    started_runs.append(try_run)
                if start_time < datetime.fromisoformat("2001-01-01T00:00:00"):
                    possible_runs.append(try_run)
                elif (now - start_time).total_seconds() > timeout_limit and end_time < datetime.fromisoformat("2001-01-01T00:00:00"):
                    possible_runs.append(try_run)
                elif end_time > datetime.fromisoformat("2001-01-01T00:00:00"):
                    done_count += 1
            if len(possible_runs) > 0:
                next_id = random.choice(possible_runs)
                print('This process is now attempting run', next_id, 'from batch', batch_id)
            print('Started', len(started_runs), 'Remaining', len(possible_runs))

        if done_count == nruns:
            print('All runs in batch completed. Carry  on.')
            go = False

        elif next_id == -1 and len(started_runs) == nruns:   #Nothing to be done now, but everything isn't yet complete
            print('Batch not complete but all processes started, waiting to check if anything else needs doing...')
            time.sleep(30.0*random.uniform(0.0,1.0))
        elif next_id == -1:
            print('File read failed. Reattempting soon')
            time.sleep(1.0*random.uniform(0.0,1.0))
        else:
            #Modify logs to indicate this has been started
            with LockedLog("batch_logs/0_%d.txt" % batch_id, mode = "r+") as f:
                lines = f.readlines()
                if len(lines) != nruns:
                    #print(len(lines), nruns)
                    continue
                    #raise Exception('Log file is corrupted or the batch number has been reused or something. Bugger.')
                parts = lines[next_id].strip().split('\t')
                if (datetime.now() - datetime.fromisoformat(parts[1].strip('{}'))).total_seconds() < timeout_limit:
                    print('Process clash detected, trying something else...')
                    #Another process has started this in the meantime, so don't bother, or you can get everything writing on top of each other.
                    continue
                parts[1] = datetime.now().strftime("{%Y-%m-%dT%H:%M:%S}")
                #Now add on open flux?
                lines[next_id] = '\t'.join(parts) + '\n'
                f.seek(0)
                f.writelines(lines)
                f.truncate()

            if True:

                #---------------------------------------------------
                #SCRIPT IS RUN HERE
                arg1 = (start + timedelta(days=next_id*time_cadence))
                obs_time = arg1.isoformat()
                nrho = 30
                ns = 90
                nphi = 180
                rss = 2.5

                print('Run parameters', obs_time, ns, nphi,1.0*5e-2/nphi, corona_temp, mf_constant)
                hmi_map = outflowpy.obtain_data.prepare_hmi_mdi_time(obs_time, ns, nphi, smooth = 1.0*5e-2/nphi, use_cached = True, cache_directory = "/extra/tmp/vgjn10/projects/outflowpy/test_location/_download_cache")   #Outputs the set of data corresponding to this particular Carrington rotation.

                outflow_in = outflowpy.Input(hmi_map, nrho, rss, corona_temp = corona_temp, mf_constant = mf_constant)
                outflow_out = outflowpy.outflow_fortran(outflow_in)

                #Calculate the open flux and save. Take care with the units
                rsun_cm = 6.957e10
                surface_area = 4*np.pi*(np.exp(outflow_in.grid.rg[-1])*rsun_cm)**2
                openflux = np.sum(np.abs(outflow_out.br)[:,:,-1])*surface_area/(nphi*ns)

                #---------------------------------------------------

                #_, openflux = run_outflow(parameters)
                if openflux > 1e21 and openflux < 1e25:

                    with LockedLog("./batch_logs/0_%d.txt" % batch_id, mode = "r+") as f:
                        lines = f.readlines()
                        parts = lines[next_id].strip().split('\t')
                        parts[2] = datetime.now().strftime("{%Y-%m-%dT%H:%M:%S}")
                        parts[3] = "{" + str(openflux) + "}"
                        lines[next_id] = '\t'.join(parts) + '\n'
                        f.seek(0)
                        f.writelines(lines)
                        f.truncate()
                    print('Run successful!')
                else:
                    print("Open flux value seems spurious. Waiting a bit and trying again.")
                    time.sleep(2.0*random.uniform(0.0,1.0)*timeout_limit)
                    pass
            else:
                print("Run failed for some reason... If this keeps happening try running 'runbatch.py $(batch_id) debug' and see what the problem is")
                time.sleep(2.0*random.uniform(0.0,1.0)*timeout_limit)
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

#Set the thing running. Can specify mf, corona temp and the time cadence here if desired.
if len(sys.argv) > 1:                                                                                                               
    batch_id = int(sys.argv[1])
else:                                                                              
    raise Exception('Specify batch number.')

mf_constants = [0.0,1e-17,5e-17,1e-16,5e-16]
corona_temps = [1e6,1.5e6,2e6,2.5e6,3e6]

run_batch(batch_id, time_cadence = 1, mf_constant = mf_constants[batch_id%10], corona_temp = corona_temps[batch_id//10])


