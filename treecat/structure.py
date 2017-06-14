from __future__ import absolute_import, division, print_function

from collections import deque

import numpy as np


def make_complete_graph(num_vertices):
    '''Constructs a complete graph.

    Args:
      num_vertices: Number of vertices.

    Returns: A tuple with elements:
      V: Number of vertices.
      E: Number of edges.
      grid: a 3 x E grid of (edge, vertex, vertex) triples.
    '''
    V = num_vertices
    E = V * (V - 1) // 2
    grid = np.zeros([3, E], np.int32)
    e = 0
    for v1 in range(V):
        for v2 in range(v1 + 1, V):
            grid[:, e] = [e, v1, v2]
            e += 1
    return V, E, grid


def make_tree(edges):
    '''Constructs a tree graph from a set of (vertex,vertex) pairs.

    Args:
      edges: A list or set of unordered (vertex, vertex) pairs.

    Returns: A tuple with elements:
      V: Number of vertices.
      E: Number of edges.
      grid: a 3 x E grid of (edge, vertex, vertex) triples.
    '''
    assert all(isinstance(edge, tuple) for edge in edges)
    edges = [tuple(sorted(edge)) for edge in edges]
    edges.sort()
    E = len(edges)
    V = 1 + E
    grid = np.zeros([3, E], np.int32)
    for e, (v1, v2) in enumerate(edges):
        grid[:, e] = [e, v1, v2]
    return V, E, grid


def find_center_of_tree(grid):
    '''Finds a maximally central vertex in a tree graph.

    Args:
        grid: A tree graph as returned by make_tree().

    Returns:
        Vertex id of a maximally central vertex.
    '''
    E = grid.shape[1]
    V = 1 + E
    neighbors = [set() for _ in range(V)]
    for e, v1, v2 in grid.T:
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    queue = deque()
    for v in reversed(range(V)):
        if len(neighbors[v]) <= 1:
            queue.append(v)
    while queue:
        v = queue.popleft()
        for v2 in sorted(neighbors[v], reverse=True):
            neighbors[v2].remove(v)
            if len(neighbors[v2]) == 1:
                queue.append(v2)
    return v


def make_propagation_schedule(grid, root=None):
    '''Makes an efficient schedule for message passing on a tree.

    Args:
      grid: A tree graph as returned by make_tree().
      root: Optional root vertex, defaults to find_center_of_tree(grid).

    Returns:
      A list of (vertex, parent, children) tuples, where
        vertex: A vertex id.
        parent: Either this vertex's parent node, or None at the root.
        children: List of neighbors deeper in the tree.
        outbound: List of neighbors shallower in the tree (at most one).
    '''
    if root is None:
        root = find_center_of_tree(grid)
    E = grid.shape[1]
    V = 1 + E
    neighbors = [set() for _ in range(V)]
    for e, v1, v2 in grid.T:
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    schedule = []
    queue = deque()
    queue.append((root, None))
    while queue:
        v, parent = queue.popleft()
        schedule.append((v, parent, []))
        for v2 in sorted(neighbors[v]):
            if v2 != parent:
                queue.append((v2, v))
    for v, parent, children in schedule:
        for v2 in neighbors[v]:
            if v2 != parent:
                children.append(v2)
    return schedule


def sample_tree(grid, edge_prob, edges, seed=0):
    '''Sample a random spanning tree of a weighted complete graph using MCMC.

    Args:
      grid: A 3 x E array as returned by make_complete_graph().
      edge_prob: A length-E array of nonnormalized edge probabilities.
      edges: A list of initial edges in the form of (vertex,vertex) pairs.
      seed: Seed for random number generation.

    Returns:
      A list of (vertex, vertex) pairs.
    '''
    np.random.seed(seed)
    E = len(edges)
    V = 1 + E
    neighbors = [set() for _ in range(V)]
    for v1, v2 in edges:
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    components = np.zeros([V], dtype=np.bool_)

    # Remove and resample each initial edge.
    for v1, v2 in edges:
        # Remove this edge.
        neighbors[v1].remove(v2)
        neighbors[v2].remove(v1)
        stack = [v1]
        while stack:
            v1 = stack.pop()
            components[v1] = True
            for v2 in neighbors[v1]:
                if not components[v2]:
                    stack.append(v2)

        # Sample a new edge between the two components.
        valid_edges = np.where(
            components[grid[1, :]] != components[grid[2, :]])[0]
        valid_probs = edge_prob[valid_edges]
        valid_probs /= valid_probs.sum()
        k = np.random.choice(valid_edges, p=valid_probs)
        _, v1, v2 = grid[:, k]
        assert components[v1] != components[v2]
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
        components[:] = False

    edges = [(u1, u2) for u1 in range(V) for u2 in neighbors[u1] if u1 < u2]
    assert len(edges) == E
    edges.sort()
    return edges
