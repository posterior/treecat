from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections import deque

import numpy as np

from treecat.util import PROFILE_COUNTERS
from treecat.util import PROFILE_HISTOGRAMS
from treecat.util import profile_timed

logger = logging.getLogger(__name__)


class TreeStructure(object):
    '''Topological data representing a tree on features.'''

    def __init__(self, num_vertices):
        logger.debug('TreeStructure with %d vertices', num_vertices)
        self._num_vertices = num_vertices
        self._num_edges = num_vertices - 1
        self.set_edges([(v, v + 1) for v in range(num_vertices - 1)])
        self._complete_grid = None  # Lazily constructed.

    def __eq__(self, other):
        return (self._num_vertices == other._num_vertices and
                (self._tree_grid == other._tree_grid).all())

    def set_edges(self, edges):
        '''Sets the edges of this tree.

        Args:
          edges: A list of (vertex, vertex) pairs.
        '''
        assert len(edges) == self._num_edges
        self._tree_grid = make_tree(edges)
        self._tree_edges = {}
        for e, v1, v2 in self._tree_grid.T:
            self._tree_edges[v1, v2] = e
            self._tree_edges[v2, v1] = e

    @property
    def num_vertices(self):
        return self._num_vertices

    @property
    def num_edges(self):
        return self._num_edges

    @property
    def tree_grid(self):
        '''Array of (edge, vertex, vertex) triples defining the tree graph.'''
        return self._tree_grid

    @property
    def complete_grid(self):
        '''Array of (edge,vertex,vertex) triples defining a complete graph.'''
        if self._complete_grid is None:
            self._complete_grid = make_complete_graph(self._num_vertices)
        return self._complete_grid

    def find_edge(self, v1, v2):
        '''Find the edge index e of an unsorted pair of vertices (v1, v2).'''
        return self._tree_edges[v1, v2]


def make_complete_graph(num_vertices):
    '''Constructs a complete graph.

    The inverse pairing function is: k = v1 + v2 * (v2 - 1) // 2

    Args:
      num_vertices: Number of vertices.

    Returns: A tuple with elements:
      V: Number of vertices.
      K: Number of edges.
      grid: a 3 x K grid of (edge, vertex, vertex) triples.
    '''
    V = num_vertices
    K = V * (V - 1) // 2
    grid = np.zeros([3, K], np.int32)
    k = 0
    for v2 in range(V):
        for v1 in range(v2):
            grid[:, k] = [k, v1, v2]
            k += 1
    return grid


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
    grid = np.zeros([3, E], np.int32)
    for e, (v1, v2) in enumerate(edges):
        grid[:, e] = [e, v1, v2]
    return grid


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


@profile_timed
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


class MutableTree(object):
    '''MCMC tree for random spanning trees.'''

    __slots__ = ['VEK', 'grid', 'neighbors', 'components', 'num_components']

    def __init__(self, grid, edges):
        '''Build a mutable spanning tree.

        Args:
          grid: A 3 x K array as returned by make_complete_graph().
          edges: A list of E edges in the form of (vertex,vertex) pairs.
        '''
        E = len(edges)
        V = 1 + E
        K = V * (V - 1) // 2
        assert grid.shape == (3, K)
        self.VEK = (V, E, K)
        self.grid = grid
        self.neighbors = [set() for _ in range(V)]
        for v1, v2 in edges:
            self.neighbors[v1].add(v2)
            self.neighbors[v2].add(v1)
        self.components = np.zeros([V], dtype=np.bool_)
        self.num_components = 1

    def remove_edge(self, v1, v2):
        '''Remove edge (v1, v2) and update neighbors and components.'''
        assert self.num_components == 1
        neighbors = self.neighbors
        components = self.components
        neighbors[v1].remove(v2)
        neighbors[v2].remove(v1)
        stack = [v1]
        while stack:
            v1 = stack.pop()
            components[v1] = True
            for v2 in neighbors[v1]:
                if not components[v2]:
                    stack.append(v2)
        self.num_components = 2

    def add_edge(self, v1, v2):
        '''Remove edge (v1, v2) and update neighbors and components.'''
        assert self.components[v1] != self.components[v2]
        assert self.num_components == 2
        self.neighbors[v1].add(v2)
        self.neighbors[v2].add(v1)
        self.components[:] = False
        self.num_components = 1

    def move_edge(self, v1, v2, v3, v4):
        '''Move edge (v1, v2) to (v3, v4) and update neighbors.'''
        assert self.num_components == 1
        if (v1, v2) == (v3, v4) or (v1, v2) == (v4, v3):
            return
        self.neighbors[v1].remove(v2)
        self.neighbors[v2].remove(v1)
        self.neighbors[v3].add(v4)
        self.neighbors[v4].add(v3)

    def find_tour(self, tour):
        '''Backtrack to find a tour [v1, v2, ..., v1] between two vertices.

        Args:
          tour: A partial tour [v1, v2, ..., vk] where the first edge (v1, v2)
            is missing and all other edges exist. This is modified in-place.
            This assumes that there is no edge directly between v1 and v2.

        Returns:
          A completed tour on success or None on failure.
          On success, the input argument was mutated to be complete.
          On failure, the input argument was reset to its state in entry.
        '''
        for v in self.neighbors[tour[-1]]:
            if v == tour[-2]:
                continue  # No U-turn.
            tour.append(v)
            if v == tour[0] or self.find_tour(tour) is not None:
                return tour  # Done.
            tour.pop()


@profile_timed
def sample_tree(grid, edge_prob, edges, seed=0, steps=1):
    '''Sample a random spanning tree of a weighted complete graph using MCMC.

    Args:
      grid: A 3 x K array as returned by make_complete_graph().
      edge_prob: A length-K array of nonnormalized edge probabilities.
      edges: A list of E initial edges in the form of (vertex,vertex) pairs.
      seed: Seed for random number generation.
      steps: Number of MCMC steps to take.

    Returns:
      A list of (vertex, vertex) pairs.
    '''
    logger.debug('sample_tree sampling a random spanning tree')
    np.random.seed(seed)
    tree = MutableTree(grid, edges)
    V, E, K = tree.VEK

    counter_calls = PROFILE_COUNTERS['sample_tree_calls']
    counter_steps = PROFILE_COUNTERS['sample_tree_steps']
    counter_add_propose = PROFILE_COUNTERS['sample_tree_add_propose']
    counter_add_accept = PROFILE_COUNTERS['sample_tree_add_accept']
    counter_add_choices = PROFILE_HISTOGRAMS['sample_tree_add_choices']
    counter_remove_propose = PROFILE_COUNTERS['sample_tree_remove_propose']
    counter_remove_accept = PROFILE_COUNTERS['sample_tree_remove_accept']
    counter_remove_choices = PROFILE_HISTOGRAMS['sample_tree_remove_choices']

    counter_calls += 1
    counter_steps += steps * E
    for step in range(steps * E):
        k12, v1, v2 = tree.grid[:, np.random.randint(K)]
        if v2 in tree.neighbors[v1]:
            # Remove the edge and add a random edge between two components.
            logger.debug('sample_tree step %d: try to remove edge', step)
            tree.remove_edge(v1, v2)
            valid_edges = np.where(tree.components[tree.grid[1, :]] !=
                                   tree.components[tree.grid[2, :]])[0]
            valid_probs = edge_prob[valid_edges]  # Pick an edge to add.
            valid_probs /= valid_probs.sum()
            k34 = np.random.choice(valid_edges, p=valid_probs)
            k34, v3, v4 = tree.grid[:, k34]
            tree.add_edge(v3, v4)
            counter_add_propose += 1
            counter_add_accept += (k12 != k34)
            counter_add_choices[len(valid_edges)] += 1
        else:
            # Steal a random edge from the path between these two vertices.
            logger.debug('sample_tree step %d: try to add edge', step)
            tour = tree.find_tour([v1, v2])
            valid_edges = np.zeros([len(tour) - 1], dtype=np.int32)
            for i, pair in enumerate(zip(tour, tour[1:])):
                v3, v4 = sorted(pair)
                k34 = v3 + v4 * (v4 - 1) // 2
                assert all(grid[:, k34] == (k34, v3, v4))
                valid_edges[i] = k34
            # TODO Apply exp() here rather than in tf, to improve stability.
            valid_probs = 1.0 / edge_prob[valid_edges]  # Pick an edge to omit.
            valid_probs /= valid_probs.sum()
            k34 = np.random.choice(valid_edges, p=valid_probs)
            k34, v3, v4 = tree.grid[:, k34]
            tree.move_edge(v3, v4, v1, v2)
            counter_remove_propose += 1
            counter_remove_accept += (k12 != k34)
            counter_remove_choices[len(valid_edges)] += 1

    edges = [(u1, u2) for u1 in range(V) for u2 in tree.neighbors[u1]
             if u1 < u2]
    assert len(edges) == E
    edges.sort()
    return edges
