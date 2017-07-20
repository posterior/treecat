from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import re
from collections import Counter

import numpy as np
import scipy.linalg

from treecat.structure import order_vertices

SVGPAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svgpan.js')


def layout_tree(correlation):
    """Layout tree for visualization with e.g. matplotlib.

    Args:
      correlation: A [V, V]-shaped numpy array of latent correlations.

    Returns:
      A [V, 3]-shaped numpy array of spectral positions of vertices.
    """
    assert len(correlation.shape) == 2
    assert correlation.shape[0] == correlation.shape[1]
    assert correlation.dtype == np.float32

    laplacian = -correlation
    np.fill_diagonal(laplacian, 0)
    np.fill_diagonal(laplacian, -laplacian.sum(axis=0))
    evals, evects = scipy.linalg.eigh(laplacian, eigvals=[1, 2, 3])
    assert np.all(evals > 0)
    assert evects.shape[1] == 3
    return evects


def nx_plot_tree(server, node_size=200, **options):
    """Visualize the tree using the networkx package.

    This plots to the current matplotlib figure.

    Args:
      server: A DataServer instance.
      options: Options passed to networkx.draw().
    """
    import networkx as nx
    edges = server.estimate_tree
    perplexity = server.latent_perplexity()
    feature_names = server.feature_names

    V = 1 + len(edges)
    G = nx.Graph()
    G.add_nodes_from(range(V))
    G.add_edges_from(edges)
    H = nx.relabel_nodes(G, dict(enumerate(feature_names)))
    node_size = node_size * perplexity / perplexity.max()

    options.setdefault('alpha', 0.2)
    options.setdefault('font_size', 8)
    nx.draw(H, with_labels=True, node_size=node_size, **options)


def contract_positions(XY, edges, stepsize):
    """Perturb vertex positions by an L1-minimizing attractive force.

    This is used to slightly adjust vertex positions to provide a visual
    hint to their grouping.

    Args:
      XY: A [V, 2]-shaped numpy array of the current positions.
      edges: An [E, 2]-shaped numpy array of edges as (vertex,vertex) pairs.

    """
    E = edges.shape[0]
    V = E + 1
    assert edges.shape == (E, 2)
    assert XY.shape == (V, 2)
    old = XY
    new = old.copy()
    heads = edges[:, 0]
    tails = edges[:, 1]
    diff = old[heads] - old[tails]
    distances = (diff**2).sum(axis=1)**0.5
    spacing = distances.min()
    assert spacing > 0
    diff /= distances[:, np.newaxis]
    diff *= spacing
    new[tails] += stepsize * diff
    new[heads] -= stepsize * diff
    return new


def plot_chord(begin, end, spacing, color, alpha=None):
    """Plots a circular chord from begin to end.

    This assumes that the outer circle is centered at (0,0).

    Args:
      begin: A [2]-shaped numpy array.
      end: A [2]-shaped numpy array.
      spacing: A float, extra spacing around the edge of the circle.
      color: A matplotlib color spec.
      apha: A float or None.
    """
    # Adapted from https://matplotlib.org/users/path_tutorial.html
    from matplotlib import pyplot
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    xy = np.array([begin, begin, end, end])
    dist = ((begin - end)**2).sum()**0.5
    xy[[1, 2], :] *= 1 - 2 / 3 * dist + 1 / 6 * dist**2 - spacing
    path = Path(xy, codes)
    patch = PathPatch(
        path, facecolor='none', edgecolor=color, lw=1, alpha=alpha)
    pyplot.gca().add_patch(patch)


def plot_circular(server,
                  tree_samples=0,
                  fontsize=8,
                  color='#4488aa',
                  contract=0.08):
    """Plot a tree stucture with features arranged around a circle.

    Args:
      server: A DataServer instance.
      tree_samples: Number of trees to sample, in addition to the mode.
      fontsize: The font size for labels, in points.
      color: A matplotlib color spec for edge colors.
      contract: Contract vertex positions by this amount to visually hint
        their grouping.

    Requires:
      matplotlib.
    """
    from matplotlib import pyplot

    # Extract ordered parameters to draw.
    edges = server.estimate_tree
    order, order_inv = order_vertices(edges)
    edges = np.array(
        [[order[v1], order[v2]] for v1, v2 in edges], dtype=np.int32)
    feature_names = np.array(server.feature_names)[order_inv]
    feature_density = server.feature_density()[order_inv]
    observed_perplexity = server.observed_perplexity()[order_inv]
    latent_perplexity = server.latent_perplexity()[order_inv]
    alphas = 0.25 + 0.75 * feature_density

    V = len(feature_names)
    angle = np.array([2 * np.pi * ((v + 0.5) / V + 0.25) for v in range(V)])
    XY = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    if contract:
        XY = contract_positions(XY, edges, stepsize=contract)
    X = XY[:, 0]
    Y = XY[:, 1]
    R_text = 1.06
    R_obs = 1.03
    R_lat = 1.0
    adjust = np.pi / V
    spacing = 1 / V

    # Plot feature labels.
    for v, name in enumerate(feature_names):
        x, y = XY[v]
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
            fontsize=fontsize,
            alpha=alphas[v])

    # Plot observed-latent edges.
    s = 2 * observed_perplexity
    pyplot.scatter(R_obs * X, R_obs * Y, s, lw=0, color=color)
    s = 2 * latent_perplexity
    pyplot.scatter(R_lat * X, R_lat * Y, s, lw=0, color=color)
    pyplot.plot(
        np.stack([R_obs * X, X]),
        np.stack([R_obs * Y, Y]),
        color=color,
        lw=0.75)

    # Plot maximum a posteriori latent-latent edges.
    for v1, v2 in edges:
        plot_chord(XY[v1], XY[v2], spacing, color)

    # Plot monte carlo sampled latent-latent edges.
    edge_counts = Counter()
    for sample in server.sample_tree(tree_samples):
        edge_counts.update(sample)
    for (v1, v2), count in edge_counts.items():
        v1 = order[v1]
        v2 = order[v2]
        plot_chord(XY[v1], XY[v2], spacing, color, alpha=count / tree_samples)


def add_panning_to_svg(source, destin=None):
    """Add pan and zoom to an svg file by embedding SVGPan in the file.

    Args:
      source: Path to the input file.
      destin: Path to the output file. Defaults to source.
    """
    if destin is None:
        destin = source
    with io.open(source) as f:
        source_lines = list(f)
    destin_lines = []
    add = destin_lines.append
    for line in source_lines:
        if re.search('SVGPan library', line):
            raise ValueError('{} already supports panning'.format(source))
        if line.startswith('<svg '):
            add('<svg height="100%" width="100%" version="1.1"'
                ' xmlns="http://www.w3.org/2000/svg"'
                ' xmlns:xlink="http://www.w3.org/1999/xlink">\n')
            add('<script type="text/ecmascript"><![CDATA[\n')
            add(open(SVGPAN).read())
            add(']]></script>\n')
            add('<g id="viewport" transform="scale(1,1) translate(0,0)">\n')
        elif line.startswith('</svg>'):
            add('</g>\n')
            add(line)
        else:
            add(line)
    with io.open(destin, 'w') as f:
        for line in destin_lines:
            f.write(line)
