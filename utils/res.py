import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

plt.rcParams['text.usetex'] = True

# 10 k squares
x1 = [1, 1.2, 1.9, 2.9, 4]
x2 = [1.4, 1.75, 1.8, 2, 4.4]
x3 = [8, 7, 5, 4, 1]
time = [11.5, 10.8, 8.3, 6.26, 1.41]
acc = [93.84,	93.09,	91.34,	90.63,	39.16]
timeq = [25.58,	23.03,	17.92,	13.38,	4.12]
accq = [95.95,	95.38,	95.19,	94.63,	38.15]
"""
# 1k squares
x1 = [1, 1.1, 1.4, 2.1, 2.4, 2.5, 2.6, 2.9, 3.3]
x2 = [2.2,	2.5,	2.55,	2.65,	3,	3.1,	3.4,	3.65,	4.95]
x3 = [8,	7,	6,	5,	5,	5,	5,	4,	1]
time = [10.3,	9.3,	8.46,	6.9,	6.2,	6.1,	5.94,	5.38,	1.64]
acc = [97.97,	97.76,	97.07,	96.62,	96.57,	96.48,	96.46,	95.84,	39.2]
timeq = [18.7,	18.5,	15.49,	13.3,	12.2,	12.4, 12.1,	10.4, 4.88]
accq = [96.61,	96.7,	96.37,	96.59,	96.69,	96.48,	95.2,	93.8, 38.17]
# plot acc accq
# plot a step function

"""
plt.step(x1, x2, where='post', linewidth=2)
# name x and y axis
plt.xlabel(r'$w_{N_{cl}}$', fontsize=14)
plt.ylabel(r'$h_n$', fontsize=14)
# show the plot
# put grid
plt.grid()
plt.show()

"""dd
plt.plot(x2, acc, label='acc')
plt.plot(x2, accq, label='accq')
plt.xlabel(r'$h_n$')
plt.ylabel('Accuracy')
# set ylim
plt.ylim(80, 100)
# put names
plt.legend("Accuracy of NN")
plt.show()
# plot acc and accq as functions of time and timeq
# make line thicke
"""
"""
plt.plot(time, acc, label='Rtrees Accuracy', linewidth=2)
plt.plot(timeq, accq, label='Quadtrees Accuracy', linewidth=2)
plt.xlabel(r'$Time(sec)$', fontsize=14)
plt.ylabel('$Accuracy\ \%$', fontsize=14)
# set ylim
plt.ylim(70, 100)
# put grid
plt.grid()
plt.legend()
plt.show()
# save image
# plt.savefig('Figure_10k_acc_time.png', dpi=300)
"""

"""

fig, ax1 = plt.subplots()

color1 = 'tab:red'
color2 = 'tab:blue'
color3 = 'tab:green'

ax1.set_xlabel(r'$h_n$', fontsize=14)
ax1.plot(x2, time, color=color1)
ax1.plot(x2, timeq, color=color2)
ax1.tick_params(axis='y')
# set ax1 y limits
ax1.set_ylim(0, 30)
ax1.set_xlim(2, 5)
ax1.set_ylabel(r'$Time\ (s)$', fontsize=14)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color4 = 'tab:orange'
color5 = 'tab:purple'

# we already handled the x-label with ax1
ax2.set_ylabel(r'$number\ of\ clusters$', fontsize=14)
# make line thicker
ax2.plot(x2, x3, color=color4, linewidth=2)
ax2.set_xlim(1, 5)
ax2.set_ylim(0, 12)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

ax1.legend(['Rtree duration', 'Quadtree duration'], loc='upper left')
ax2.legend(['number of clusters'], loc='upper right')
# put grid on ax1 and ax2
ax1.grid()
plt.show()

"""
"""

y = []
x = []
for i in range(1, 9):
    for j in range(90, 96):
        x.append(i)
        y.append(1/((1-j/100) + j/100/i))

plt.plot(x, y)

plt.xlabel(r'$Number of clusters$', fontsize=14)
plt.ylabel(r'$Speedup$', fontsize=14)
plt.grid(True)
plt.show()

"""
