from statistics import variance
import numpy as np
import matplotlib.pyplot as plt
import scipy
# * Question 2
# continuous_data = np.arange(0,4,0.01)
# pdf = lambda x: x * np.exp(-x**2/2)
# cdf = lambda x: 1- np.exp(-x**2/2) 
# data_pdf = [pdf(x) for x in continuous_data] 
# data_cdf = [cdf(x) for x in continuous_data]

# plt.plot(continuous_data,data_pdf,color="blue", label="PDF")
# plt.plot(continuous_data,data_cdf,color="green", label="CDF")
# plt.legend()
# plt.show()

# * Question 3
# import scipy.stats as stats
# mu,variance = 0,1
# sigma = np.sqrt(variance)
# x = np.linspace(mu-3*sigma,mu+3*sigma,100)
# plt.plot(x,stats.norm.pdf(x,mu,sigma))
# plt.show()

# * Question 4
# import random
# k_vals = []
# for iter in range(200):
#     k = 0
#     for iter2 in range(40):
#         rng = random.uniform(0,1)
#         if rng < 0.6:
#             continue
#         elif rng >= 0.6:
#             k+=1
#     k_vals.append(k)

# plt.hist(k_vals,bins=40)
# plt.show()

meg_data = scipy.io.loadmat("meg_data.mat")["back_average"][0][-500:]
minimum = np.min(meg_data)
percentile_25 = np.percentile(meg_data,25)
median = np.median(meg_data)
percentile_75 = np.percentile(meg_data,75)
maximum = np.max(meg_data) 

# print(f"Minimum is {minimum}")
# print(f"25th percentile is {percentile_25}")
# print(f"Median is {median}")
# print(f"75th percentile is {percentile_75}")
# print(f"Maximum is {maximum}")

# plt.boxplot(meg_data,0,'rs',0)
# plt.xlabel("MEG Value")
# plt.show()

def sample_mean(x):
    n = len(x)
    sum_x = np.sum(x)
    return sum_x/n

def sample_stdev(x,sample_mean):
    n = len(x)
    arr = np.array(x)
    sum_squares = np.sum((arr-sample_mean)**2) 
    divide_n = sum_squares/n
    return np.sqrt(divide_n)
    

sample_mean_val = sample_mean(meg_data)
sample_stdev_val = sample_stdev(meg_data,sample_mean_val)
n = len(meg_data)

from scipy.stats import norm
meg_data.sort()
icdf = lambda x: norm.ppf(x,loc=sample_mean_val,scale=sample_stdev_val)
cdf = lambda x: norm.cdf(x,loc=sample_mean_val, scale = sample_stdev_val)
eta = [icdf(r/(n+1)) for r in range(1,501)]
plt.scatter(meg_data,eta,marker="+")
plt.plot(meg_data,meg_data,color="red")
plt.xlabel("MEG Data")
plt.ylabel("Theoretical Data")
plt.show()