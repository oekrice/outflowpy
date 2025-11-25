#Script to do a nice animation of the convergence of an outflow polynomial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette('dark')

fig = plt.figure(figsize = (7,5))
years = [2006,2008,2009,2010,2012,2013,2015,2016,2017,2019,2023,2024]
years = [2006]

for ei, eclipse_number in enumerate(years):

    log_file = './batch_logs/log_%d.txt' % eclipse_number

    log_info = []

    with open(log_file, "r") as f:
        for line in f.readlines():
            log_info.append(line.split(" "))
    log_info = np.array(log_info, dtype = 'float')

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

    for extras in range(0, np.shape(log_info)[0], 50):
        #Plot some extra ones as thin lines
        ys = np.abs(generate_poly(log_info[extras,2:], xs))
        plt.plot(xs, ys, linewidth = 0.1, c = colors[ei])

    string = ''
    for var in range(2, np.size(log_info[1])):
        string = string + str(log_info[best_id, var]) + ','
    print('Eclipse', eclipse_number, string)
    ys = generate_poly(log_info[best_id,2:], xs)

    plt.plot(xs, ys, linewidth = 2.0, c = colors[ei], label = f'Eclipse {eclipse_number}', linestyle = 'dashed')
    #print(log_info[best_id,2:])

plt.ylim(0,10)
plt.title('Optimum outflow functions for various eclipses')
plt.legend()
plt.savefig('allcomp.png')
plt.show()
plt.close()
