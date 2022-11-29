# * Question 1
import math
import matplotlib.pyplot as plt
import random

random.seed(100)

def f(x,t): return t*x*math.exp(-(t/2)*x**2)
def F(x,t): return 1-math.exp(-(t/2)*x**2)
def F_inv(x,t): return math.sqrt(-2*math.log(1-x)/t)

n = 350 
tau = 0.25

samples = []
eta = [F_inv(r/(n+1),tau) for r in range(1,n+1)] # calculate eta
for i in range(n):
    samples.append(F_inv(random.uniform(0,1),tau)) # build our random sample

samples.sort()

plt.scatter(samples,eta,marker="+")
plt.plot(samples,samples,color="red")
plt.xlabel("Samples")
plt.ylabel("Theoretical Data")
plt.show()

sample_mean = sum(samples)/len(samples)
tau_mm = math.pi/(2*sample_mean**2) # = 0.248 
print(f"tau_mm={tau_mm}")

bootstraps = []
for i in range(1,1001):
    xs = [F_inv(random.uniform(0,1),tau_mm) for _ in range(350)]
    tmp_mean = sum(xs)/len(xs)
    bootstraps.append(math.pi/(2*tmp_mean**2))

import numpy as np
ci = np.percentile(bootstraps,97.5),np.percentile(bootstraps,2.5)
print(f"CI={ci}")
plt.hist(bootstraps,20)
plt.xlabel("Freq")
plt.ylabel("Bootstrap Samples")
plt.show()

tau_ml = 1/(1/(2*n)*sum([x**2 for x in samples]))
print(f"tau_ml={tau_ml}")

alpha,beta = 2,0.1
y = 0.5 * sum([x**2 for x in samples])

E = (n+alpha)/(beta + y)
variance = (n+alpha)/(beta + y)**2 
mode = (n+alpha-1)/(beta+y)

print(f"E = {E}")
print(f"variance = {variance}")
print(f"mode={mode}")

sigma_MAP  = math.sqrt((n+alpha-1)/(beta+y)**2)
credibility = mode - (2*sigma_MAP), mode + (2*sigma_MAP)
print(f"credibility={credibility}")






