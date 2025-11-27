#This script to generate a text array of the optimum values determined by the CMA-es runs.
#Because of the various absolute values these cannot be expressed explicitly (very easily, at least), so will just set up an interpolation thing
#It would be best if this ends up as the default, I think. Maybe. Perhaps if a corona temperature isn't specified?
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

colors = sns.color_palette('tab20')


years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]

source = 'abs'
allys = np.zeros(1000)
ycount = 0
fig = plt.figure(figsize = (9,5))
for ei, eclipse_number in enumerate(years):

    #os.system(f"scp -r vgjn10@hamilton8.dur.ac.uk:/home/vgjn10/projects/outflowpy/scripts/batch_logs/log_{eclipse_number}.txt ./batch_logs_abs")

    #log_file = './batch_logs/log_%d.txt' % eclipse_number
    log_file = f'./batch_logs_{source}/log_{eclipse_number}.txt'

    if not os.path.exists(log_file):
        continue

    log_info = []

    with open(log_file, "r") as f:
        for line in f.readlines():
            log_info.append(line.split(" "))
    log_info = np.array(log_info, dtype = 'float')

    xs = np.linspace(1.0,5.0,1000)  #Basis for the x axis

    def generate_poly(coeffs, xs):
        ys = np.zeros(len(xs))
        for i in range(len(coeffs)):
            ys += coeffs[i]*xs**i
        return ys

    best_id = 0; score = 1.
    for i in range(np.size(log_info,0)):
        if log_info[i,1] < score:
            best_id = i
            score = log_info[i,1]

    ys = generate_poly(log_info[-1,2:], xs)

    if source == 'clip':
        ys[ys < 0.0] = 0.0
    else:
        ys = np.abs(ys)

    if np.max(ys) > 0.0:
        allys += ys
        ycount += 1

    plt.plot(xs, ys, linewidth = 2.0, c = colors[2*ei%20 + ei//10], label = f'{eclipse_number}', linestyle = 'dashed')
    #print(log_info[best_id,2:])
    # os.remove(f"batch_logs_{source}/optimums.txt")
    #
    # with open(f"batch_logs_{source}/optimums.txt", mode = "a") as f:
    #     f.write(f"{log_info[-1, 2:].tolist()}\n")

data = np.zeros((len(xs), 2))
meanys = allys/ycount

data[:,0] = xs
data[:,1] = meanys

np.savetxt(f'./data/opt_flow_{source}.txt', data, delimiter = ',')
