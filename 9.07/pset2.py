import matplotlib.pyplot as plt
import numpy as np
def p(x):
    c = 29
    return x/c

x_vals = [2,2,3,3,5,2,2,2,2,1,1,4]
pmf = [p(x) for x in x_vals]
cdf = np.cumsum(pmf)

# plt.plot(cdf,color='green', linestyle='solid')
# plt.title("CDF")
# plt.xticks(np.arange(len(cdf)),np.arange(1,len(cdf)+1))
# plt.ylabel("CDF(X)")
# plt.xlabel("X")
# plt.show()

out = np.sum([x*p(x) for x in x_vals])
print(out)