#Script to do a nice animation of the convergence of an outflow PARKER SOLUTION
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.optimize import root_scalar, minimize_scalar
from scipy import interpolate

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # LaTeX-like Computer Modern
    "font.family": "serif",
})

colors = sns.color_palette('tab20')

fig = plt.figure(figsize = (6.9,3.5))
years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]

nx = 100
allys = np.zeros((nx))
ycount = 0

if os.path.exists(f"./batch_logs/optimums.txt"):
    os.remove(f"./batch_logs/optimums.txt")

for counter in range(0,1):
    rhos = np.linspace(np.log(1.0),np.log(2.5),nx)
    xs = np.exp(rhos)
    allskills = []
    allbestskills = []
    for ei, eclipse_number in enumerate(years[:]):

        os.system(f"scp -r vgjn10@hamilton8.dur.ac.uk:/home/vgjn10/projects/outflowpy/review_scripts/batch_logs/log_{eclipse_number}.txt ./batch_logs")

        log_file = f'./batch_logs/log_{eclipse_number}.txt'

        if not os.path.exists(log_file):
            continue

        log_info = []

        with open(log_file, "r") as f:
            for line in f.readlines():
                log_info.append(line.split(" "))
        log_info = np.array(log_info, dtype = 'float')

        #log_info = log_info[:counter+1,:]
        skills = log_info[:,1]
        bestskills = 0.0*skills

        best_id = 0; score = 1.
        for i in range(np.size(log_info,0)):
            if log_info[i,1] < score:
                best_id = i
                score = log_info[i,1]
                bestskills[i] = log_info[i,1]
            else:
                bestskills[i] = score
        #     plt.plot(xs, ys, linewidth = 0.1, c = colors[ei])

        string = ''
        for var in range(2, np.size(log_info[0])):
            string = string + str(log_info[-1, var]) + ','
        print('Eclipse', eclipse_number, string)

        polynomial_coeffs = log_info[best_id ,2:]
        #polynomial_coeffs = log_info[-1 ,2:]


        allbestskills.append(bestskills)
        allskills.append(skills)

        def poly_at_pt(r):
            #Polynomial value at the explicit point r (don't forget the exponentials!)
            res = 0
            for i in range(len(polynomial_coeffs)):
                res = res + polynomial_coeffs[i]*r**i
            return res

        raw_poly = poly_at_pt(xs) + 1e-6

        ys = (raw_poly*np.exp(raw_poly))/(np.exp(raw_poly)-1)


        ymax_ind = np.argmax(ys)
        if ymax_ind != len(ys) - 1:
            ys[ymax_ind:] = ys[ymax_ind]

        if np.max(ys) > 0.0:
            allys += ys
            ycount += 1

        plt.plot(xs, ys, linewidth = 2.0, c = colors[2*ei%20 + ei//10], label = f'{eclipse_number}', linestyle = 'dashed')
        #print(log_info[best_id,2:])

        with open(f"./batch_logs/optimums.txt", mode = "a") as f:
            f.write(f"{polynomial_coeffs.tolist()}\n")
        with open(f"./paper_plots/batch_logs/optimums.txt", mode = "a") as f:
            f.write(f"{polynomial_coeffs.tolist()}\n")

    plt.plot(xs, allys/ycount, linewidth = 3.0, c = 'black', label = 'Mean', linestyle = 'solid')

    plt.xlabel('Altitude $r$ ($R_\odot$)')
    plt.ylabel('Dimensionless outflow speed $V_(r)$')
    plt.legend(ncols = 3)
    plt.tight_layout()
    plt.savefig('./10_optimums.pdf')
    plt.show()
    plt.close()

fig = plt.figure()
for si, skills in enumerate(allskills):
    plt.plot(skills, c = colors[2*si%20 + si//10], linewidth = 0.2)
    plt.plot(allbestskills[si], label = years[si], c = colors[2*si%20 + si//10])
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('./temp/convergence.png')
plt.show()
