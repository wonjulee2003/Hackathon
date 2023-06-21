import numpy as np
import time
import matplotlib.pyplot as plt

file="/Users/eddiedeleu/Desktop/MagNet Project/Python/epochTime.npy"
times = np.load(file)

times = times[10:]
xs = [x for x in range(len(times))]

plt.plot(xs, times)
plt.show()