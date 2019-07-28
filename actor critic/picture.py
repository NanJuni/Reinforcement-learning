import matplotlib.pyplot as plt
import numpy as np
AC_001 = np.loadtxt("AC_001.txt")
AC_01 = np.loadtxt("AC_01.txt")
ACjoin_001 = np.loadtxt("ACjoin_001.txt")
ACjoin_01 = np.loadtxt("ACjoin_01.txt")
x = np.arange(0, len(AC_01), 1)
plt.plot(x, AC_01, label="AC learningrate = 0.01")
plt.plot(x, AC_001, label="AC rate = 0.001")
plt.plot(x, ACjoin_01, label="ACjoin rate = 0.01")
plt.plot(x, ACjoin_001, label="ACjoin rate = 0.001")
plt.legend()
plt.savefig("performance.jpg")
plt.show()