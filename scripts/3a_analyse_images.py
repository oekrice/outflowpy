#Script to analyse the results of a batch of image optimisations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

colors = sns.color_palette('dark')


if len(sys.argv) > 1:
    batch_id = int(sys.argv[1])
    file_root = "img_plots/%d/" % int(sys.argv[1])
else:
    file_root = "img_plots/"
    batch_id = 0

log_file = file_root + 'log.txt'

log_info = []

with open(log_file, "r") as f:
    for line in f.readlines():
        log_info.append(line.split(" "))
log_info = np.array(log_info, dtype = 'float')

best_id = 0; score = 1.
for i in range(np.size(log_info,0)):
    if log_info[i,1] < score:
        best_id = i
        score = log_info[i,1]

print('Best image so far:', best_id, log_info[best_id])
print('Best parameters:')
string = ''
for var in range(2, np.size(log_info[1])):
    string = string + str(log_info[best_id, var]) + ','

print(string)

print('Last parameters in previous run:')
string = ''
for var in range(2, np.size(log_info[1])):
    string = string + str(log_info[-1, var]) + ','
print(string)

def determine_error_bounds(values, time_cadence = 20, error_bound = 0.8):
    #Function to take the messy daily data and transform into a nice smooth function with error bounds taking account of the wiggles.
    #Time cadence is the regularity and range of the data points being tested, and error bound is the percentile range with which the min and max values will be calculated
    means = []; mins = []; maxs = []
    for i in range(len(values)):
        local_min = i - time_cadence/2
        local_max = i + time_cadence/2
        date_mask = (np.arange(len(values)) >= local_min) & (np.arange(len(values)) <= local_max)
        local_values = values[date_mask]
        means.append(np.mean(local_values))
        mins.append(np.percentile(local_values, 100*(1.0 - error_bound)))
        maxs.append(np.percentile(local_values, 100*(error_bound)))
    return means, mins, maxs

#Let's do some plots of the variables

fig, axs = plt.subplots(2,1, figsize = (7,5))
ax = axs[0]
for var_id, variable in enumerate(range(2, np.size(log_info[1]))):
    means, mins, maxs = determine_error_bounds(log_info[:,variable])
    ax.plot(means, color = colors[var_id%10], linewidth = 1.0, label = var_id)
    ax.plot(mins, color = colors[var_id%10], linewidth = 0.5, linestyle = 'dashed')
    ax.plot(maxs, color = colors[var_id%10], linewidth = 0.5, linestyle = 'dashed')

ax.set_title('Convergence of parameter values, run %d' % batch_id)
ax.set_ylabel('Parameter value')
ax.set_xlabel('Interation')
ax.legend()

#Do the same treatment for the skill score
ax = axs[1]
means, _, _ = determine_error_bounds(log_info[:,1])
ax.plot(log_info[:,1], color = 'black', linewidth = 0.5)
ax.plot(means, color = 'black', linewidth = 1.0)

plt.tight_layout()
plt.savefig(file_root + 'converge.png')
plt.show()
plt.close()







