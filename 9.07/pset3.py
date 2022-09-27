from statistics import variance
import numpy as np
import matplotlib.pyplot as plt

# continuous_data = np.arange(0,4,0.01)
# pdf = lambda x: x * np.exp(-x**2/2)
# cdf = lambda x: 1- np.exp(-x**2/2) 
# data_pdf = [pdf(x) for x in continuous_data] 
# data_cdf = [cdf(x) for x in continuous_data]

# plt.plot(continuous_data,data_pdf,color="blue", label="PDF")
# plt.plot(continuous_data,data_cdf,color="green", label="CDF")
# plt.legend()
# plt.show()

# import scipy.stats as stats
# mu,variance = 0,1
# sigma = np.sqrt(variance)
# x = np.linspace(mu-3*sigma,mu+3*sigma,100)
# plt.plot(x,stats.norm.pdf(x,mu,sigma))
# plt.show()

import random
k_vals = []
for iter in range(200):
    k = 0
    for iter2 in range(40):
        rng = random.uniform(0,1)
        if rng < 0.6:
            continue
        elif rng >= 0.6:
            k+=1
    k_vals.append(k)

plt.hist(k_vals,bins=40)
plt.show()
