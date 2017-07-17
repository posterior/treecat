from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from treecat.structure import print_tree


def order_features(server):
    edges = server.estimate_tree
    names = server.feature_names
    order = np.empty(len(names), np.int32)
    for v, name in enumerate(print_tree(edges, names).split()):
        order[names.index(name)] = v
    return order


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
    order = order_features(server)
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
        dist = ((xy[-1][0] - xy[0][0])**2 + (xy[-1][1] - xy[0][1])**2)**0.5
        xy[1] = xy[0] * (1 - (dist / 3)**0.8)
        xy[2] = xy[3] * (1 - (dist / 3)**0.8)
        path = Path(xy, codes)
        patch = PathPatch(path, facecolor='none', edgecolor=color, lw=1)
        pyplot.gca().add_patch(patch)
