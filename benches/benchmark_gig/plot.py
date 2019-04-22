#! /usr/bin/python3

import matplotlib.pyplot as plt
import sys

points = []
for line in sys.stdin:
    parts = line.split()
    if len(parts) > 1:
        point = float(parts[1])
        points.append(point)

plt.plot(points)
plt.show()
