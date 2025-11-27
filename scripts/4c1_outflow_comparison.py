#Script to do a nice animation of the convergence of an outflow polynomial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

colors = sns.color_palette('tab20')

fig = plt.figure(figsize = (9,5))
years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]

sources = ['clip', 'abs']

source = sources[0]

allys = np.zeros(1000)
ycount = 0
os.remove(f"batch_logs_{source}/optimums.txt")


for counter in range(0,1):
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

        #log_info = log_info[:counter+1,:]

        xs = np.linspace(1.0,2.5,1000)  #Basis for the x axis

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

        # for extras in range(0, np.shape(log_info)[0], 0):
        #     #Plot some extra ones as thin lines
        #     ys = generate_poly(log_info[extras,2:], xs)
        #
        #     if source == 'clip':
        #         ys[ys < 0.0] = 0.0
        #     else:
        #         ys = np.abs(ys)
        #
        #     print(extras, np.min(ys), np.max(ys))
        #     plt.plot(xs, ys, linewidth = 0.1, c = colors[ei])

        string = ''
        for var in range(2, np.size(log_info[0])):
            string = string + str(log_info[-1, var]) + ','
        print('Eclipse', eclipse_number, string)
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

        with open(f"batch_logs_{source}/optimums.txt", mode = "a") as f:
            f.write(f"{log_info[-1, 2:].tolist()}\n")

    plt.plot(xs, allys/ycount, linewidth = 3.0, c = 'black', label = 'Mean', linestyle = 'solid')

    plt.title('Optimum outflow speeds for various eclipses')
    plt.xlabel('Radius')
    plt.ylabel('Outflow speed (code units)')
    plt.legend()
    plt.savefig('./comp.png')
    plt.show()
    plt.close()
