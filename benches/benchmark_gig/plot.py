#! /usr/bin/python3

import matplotlib.pyplot as plt
import sys

points = []
for line in sys.stdin:
    point = float(line.split()[1])
    points.append(point)

plt.plot(points)
plt.show()
