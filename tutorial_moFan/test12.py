import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()

mu,sigma=-1,1
xs=np.linspace(-5,5,1000)
plt.plot(xs, norm.pdf(xs,loc=mu,scale=sigma))
plt.show

# import time
# plt.ion() 	#interative mode is on
# x = np.linspace(0, 50, 1000)
# plt.figure(1) # figure 1
# plt.plot(x, np.sin(x))
# plt.draw()
# time.sleep(5)
# plt.close(1)
# plt.figure(2) # figure 2
# plt.plot(x, np.cos(x))
# plt.draw()
# time.sleep(5)
# print 'it is ok'