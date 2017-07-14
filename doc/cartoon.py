#!/usr/bin/env python

import numpy as np

import matplotlib

matplotlib.use('Agg')  # Required for headless operation.
from matplotlib import pyplot as plt  # noqa: E402 isort:skip

# Vertices.
Ys = np.array([
    [-2.0, 0.6],
    [-1.5, 0.0],
    [-0.5, 0.3],
    [+0.5, 0.3],
    [+1.5, 0.0],
    [+2.0, 0.6],
])

# Edges.
E = np.array([[0, 2], [1, 2], [2, 3], [3, 4], [3, 5]], dtype=np.int32)
Ex = Ys[E[:, :], 0].T
Ey = Ys[E[:, :], 1].T

dy = 0.4
color = '#4488aa'
fig = plt.figure(figsize=(4, 2), frameon=False)
plt.plot(Ex, Ey, lw=3, color=color)
for v in range(Ys.shape[0]):
    plt.arrow(
        Ys[v, 0],
        Ys[v, 1],
        dx=0,
        dy=-0.65 * dy,
        color=color,
        head_width=0.05,
        lw=2)
plt.plot(
    Ys[:, 0],
    Ys[:, 1],
    lw=0,
    color=color,
    marker='o',
    markeredgewidth=0,
    markersize=16)
plt.plot(
    Ys[:, 0],
    Ys[:, 1] - dy,
    lw=0,
    color=color,
    marker='o',
    markeredgewidth=0,
    markersize=8)

plt.axis('equal')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.tight_layout(pad=0)

plt.savefig('cartoon.png', transparent=True)
