import random
import numpy as np


random.seed(907)

data_source = "CALCIUM_RATE.txt"

data = []
with open(data_source,'r') as f:
    for line in f:
        val1,val2 = line.split("\t")
        data.append((int(val1),int(val2)))

# print(data)

B = 100
x_data = np.array([x[1] for x in data])
y_data = np.array([y[0] for y in data])
n = len(x_data)

variance = np.var(y_data)
print("variance",variance)

e_k = np.array([random.gauss(0, variance) for _ in range(n)])
# print("e_k",e_k)

# * From 14.12 and 14.13, we can find hat_Beta_0 (which is just hat alpha) and hat_Beta_1 (which is just beta), and we found e_k above
# * 14.15 tells us how to find hat_variance

sample_mean_x = np.mean(x_data) # * x_bar from 14.12
sample_mean_y = np.mean(y_data) # * y_bar from 14.12

print("x bar",sample_mean_x)
print("y_bar",sample_mean_y)

def calc_hat_Beta_1(x_data,y_data, sample_mean_x, sample_mean_y):

    numerator = np.sum((x_data-sample_mean_x)*(y_data-sample_mean_y))
    # numerator = sum([((x-sample_mean_x)*(y-sample_mean_y)) for x,y in data])

    # denominator = sum([(x[0]-sample_mean_x)**2 for x in data])
    denominator = np.sum((x_data-sample_mean_x)**2)

    return numerator/denominator

hat_Beta_1 = calc_hat_Beta_1(x_data,y_data, sample_mean_x, sample_mean_y)
print("hat_beta_1",hat_Beta_1)

def calc_hat_Beta_0(beta_1,sample_mean_x,sample_mean_y):
    return sample_mean_y - (beta_1*sample_mean_x)

hat_Beta_0 = calc_hat_Beta_0(hat_Beta_1, sample_mean_x, sample_mean_y)
print("hat_beta_0",hat_Beta_0)

def calc_hat_var(x_data,y_data,beta_0,beta_1):
    numerator = np.sum((y_data-beta_0-beta_1*x_data)**2)

    return numerator/n

hat_var = calc_hat_var(x_data, y_data, hat_Beta_0, hat_Beta_1)
print("hat_var",hat_var)

hat_e_k = y_data - sample_mean_y
# print("hat_e_k",hat_e_k)

F_n = np.sort(hat_e_k)# / n
# print("F_n",F_n)

from numpy.random import default_rng

rng = default_rng()

def bootstrap(F_n,hat_Beta_0,hat_Beta_1,x_data,B):

    hat_Beta_0_stars = []
    hat_Beta_1_stars = []

    for _ in range(B):
        e_star = []

        for i in range(n): # * Step i
            e_star.append(rng.choice(F_n))
        e_star = np.array(e_star)
        # print("e_star",e_star)

        y_star = hat_Beta_0+hat_Beta_1*x_data+e_star # * step ii
        # print("y_star",y_star)

        hat_Beta_1_star = calc_hat_Beta_1(x_data, y_star, np.mean(x_data), np.mean(y_star)) # * step iii
        hat_Beta_0_star = calc_hat_Beta_0(hat_Beta_1_star, sample_mean_x, sample_mean_y)

        hat_Beta_0_stars.append(hat_Beta_0_star)
        hat_Beta_1_stars.append(hat_Beta_1_star)
    
    return np.array(hat_Beta_0_stars),np.array(hat_Beta_1_stars)

hB0star, hB1star = bootstrap(F_n, hat_Beta_0, hat_Beta_1, x_data, B) # * Part A
# print(hB0star,hB1star)
import math
def standard_error(B,b_stars):
    numerator = np.sum((b_stars-np.mean(b_stars))**2)
    denominator = B-1 

    return math.sqrt(numerator/denominator)

se_b_0 = standard_error(B, hB0star)
print("standard error of b0_star",se_b_0) # * Part A

se_b_1 = standard_error(B, hB1star)
print("standard error of b1_star",se_b_1) # * Part A

ci_method_1 = {
    "b0":(np.mean(hB0star)-se_b_0,np.mean(hB0star)+se_b_0),
    "b1":(np.mean(hB1star)-se_b_1,np.mean(hB1star)+se_b_1),
    } 

print("CI by standard error",ci_method_1)

ci_method_2 = {
    "b0":(np.percentile(hB0star,2.5),np.percentile(hB0star,97.5)),
    "b1":(np.percentile(hB1star,2.5),np.percentile(hB1star,97.5)),
    } 

print("CI by histogram",ci_method_2)

import matplotlib.pyplot as plt

# plt.hist(hB0star,bins="auto")
# plt.show()
# plt.hist(hB1star,bins="auto")
# plt.show()


# * The estimates in problem 8 were 1670 for alpha (beta_0) and -3.24 for beta (beta_1)
# * The standard errors were 29 and 0.47
# * The confidence intervals were [1612,1728] and [-4.2,-2.2]

# * The estimates here are 1670.31 and -3.245
# * The standard errors are 30 and 0.639
# * The CIs by standard error are [1638.32,1698.89] and [-3.8491,-2.56917]
# * The CIs by histogram are [1611.789,1727.636] and [-4.4564,-2.0084]