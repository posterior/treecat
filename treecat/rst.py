'''Algorithms for sampling random spanning trees of weighted dense graphs.'''

from __future__ import absolute_import, division, print_function

import numpy as np


def sample_tree_exact_naive(num_vertices,
                            grid,
                            edge_prob,
                            seed=0,
                            use_pinv=True):
    '''Sample a random spanning tree of a weighted complete graph.

    Args:
      num_vertices: Number of vertices = V.
      grid: A 3 x K array as returned by make_complete_graph().
      edge_prob: A length-K array of nonnormalized edge probabilities.
      seed: Seed for random number generation.

    Returns:
      A list of (vertex, vertex) pairs.
    '''
    np.random.seed(seed)
    V = num_vertices
    E = V - 1
    K = V * (V - 1) // 2
    infinity = 1e12
    assert len(edge_prob) == K
    assert grid.shape == (3, K)

    # Construct laplacian matrix.
    L = np.zeros([V, V], dtype=np.float32)
    L[grid[1, :], grid[2, :]] = -edge_prob
    L[grid[2, :], grid[1, :]] = -edge_prob
    L[np.diag_indices_from(L)] = -np.sum(L, axis=1)

    # Process edges in heuristic order of decreasing edge weight.
    result = []
    schedule = list(range(K))
    schedule.sort(key=lambda k: -edge_prob[k])
    for k in schedule:
        _, u, v = grid[:, k]

        # Compute effective resistance.
        uv = np.zeros([V])
        uv[u] = 1.0
        uv[v] = -1.0
        if use_pinv:
            pinvL = np.linalg.pinv(L, rcond=1e-6)
            # FIXME this fails:
            # if __debug__:
            #     total = np.dot(pinvL[grid[1, :], grid[2, :]], edge_prob)
            #     assert abs(total - (V - 1)) < 1e-4, total
            Reff = edge_prob[k] * np.dot(uv, np.dot(pinvL, uv))
        else:
            Reff = edge_prob[k] * np.dot(uv, np.linalg.solve(L, uv))
        assert 0.0 < Reff and Reff < 1.00001, Reff

        # Decide whether to include this edge in the tree.
        if np.random.uniform() < Reff:
            # Add edge to tree.
            result.append(tuple(sorted((u, v))))
            L[u, v] -= infinity
            L[v, u] -= infinity
            L[u, u] += infinity
            L[v, v] += infinity
        else:
            # Discard edge.
            p = L[u, v]
            L[u, v] = 0
            L[v, u] = 0
            L[u, u] += p
            L[v, v] += p
        if len(result) == E:
            break

    result.sort()
    return result
