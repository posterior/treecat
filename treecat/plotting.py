from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from treecat.structure import find_center_of_tree
from treecat.structure import make_tree
from treecat.structure import print_tree


def order_features_v1(server):
    """Sort features using print_tree."""
    edges = server.estimate_tree
    names = server.feature_names
    order = np.empty(len(names), np.int32)
    for v, name in enumerate(print_tree(edges, names).split()):
        order[names.index(name)] = v
    return order


def order_features_v2(server):
    """Sort features using greedy bin packing."""
    edges = server.estimate_tree
    grid = make_tree(edges)
    root = find_center_of_tree(grid)

    E = len(edges)
    V = E + 1
    neighbors = [set() for _ in range(V)]
    for v1, v2 in edges:
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    orders = [None for v in range(V)]
    seen = [False] * V
    stack = [root]
    seen[root] = True
    while stack:
        v = stack[-1]
        done = True
        for v2 in neighbors[v]:
            if not seen[v2]:
                stack.append(v2)
                seen[v2] = True
                done = False
                break
        if not done:
            continue
        lhs = []
        rhs = []
        parts = [(v2, orders[v2]) for v2 in neighbors[v]
                 if orders[v2] is not None]
        parts.sort(key=lambda x: len(x[1]))
        for v2, part in parts:
            if len(lhs) < len(rhs):
                if part.index(v2) < len(part) / 2:
                    part.reverse()
                lhs += part
            else:
                if part.index(v2) > len(part) / 2:
                    part.reverse()
                rhs += part
        orders[v] = lhs + [v] + rhs
        stack.pop()

    result = [None] * V
    for v1, v2 in enumerate(orders[root]):
        result[v2] = v1
    return result


def plot_circular(server, color='#4488aa'):
    """Plot a tree stucture with features arranged around a circle.

    Args:
      server: A DataServer instance.
      color: A matplotlib color spec.

    Requires:
      matplotlib.
    """
    from matplotlib import pyplot
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    # Extract ordered parameters to draw.
    order = order_features_v2(server)
    feature_names = np.array(server.feature_names)[order]
    feature_density = server.feature_density()[order]
    observed_perplexity = server.observed_perplexity()[order]
    latent_perplexity = server.latent_perplexity()[order]
    # latent_correlation = server.latent_correlation()[order, :][:, order]
    edges = np.array(
        [[order[v1], order[v2]] for v1, v2 in server.estimate_tree],
        dtype=np.int32)
    alphas = 0.25 + 0.75 * feature_density

    V = len(feature_names)
    angle = np.array([2 * np.pi * (v / V - 0.25001) for v in range(V)])
    X = np.cos(angle)
    Y = np.sin(angle)
    R_text = 1.06
    R_obs = 1.03
    R_lat = 1.0
    adjust = np.pi / V

    # Plot labels.
    for v, name in enumerate(feature_names):
        x = X[v]
        y = Y[v]
        rot = angle[v] * 360 / (2 * np.pi)
        props = {}
        # Work around matplotlib being too smart.
        if x > 0:
            props['ha'] = 'left'
            x -= adjust
        else:
            rot -= 180
            props['ha'] = 'right'
            x += adjust
        if y > 0:
            props['va'] = 'bottom'
            y -= adjust
        else:
            props['va'] = 'top'
            y += adjust
        pyplot.text(
            R_text * x,
            R_text * y,
            name,
            props,
            rotation=rot,
            fontsize=8,
            alpha=alphas[v])

    # Plot observed-latent edges.
    s = 2 * observed_perplexity
    pyplot.scatter(R_obs * X, R_obs * Y, s, lw=0, color=color)
    s = 2 * latent_perplexity
    pyplot.scatter(R_lat * X, R_lat * Y, s, lw=0, color=color)
    pyplot.plot(
        np.stack([R_obs * X, R_lat * X]),
        np.stack([R_obs * Y, R_lat * Y]),
        color=color,
        lw=0.75)

    # Plot arcs between features.
    # Adapted from https://matplotlib.org/users/path_tutorial.html
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    for v1, v2 in edges:
        xy = np.array([[X[v1], Y[v1]], [0, 0], [0, 0], [X[v2], Y[v2]]])
        dist = 0.5 * ((X[v1] - X[v2])**2 + (Y[v1] - Y[v2])**2)**0.5
        xy[1] = xy[0] * (1 - 4 / 3 * dist + 2 / 3 * dist**2)
        xy[2] = xy[3] * (1 - 4 / 3 * dist + 2 / 3 * dist**2)
        path = Path(xy, codes)
        patch = PathPatch(path, facecolor='none', edgecolor=color, lw=1)
        pyplot.gca().add_patch(patch)
