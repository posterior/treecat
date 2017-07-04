# Tree-Cat

[![Build Status](https://travis-ci.org/posterior/treecat.svg?branch=master)](https://travis-ci.org/posterior/treecat)
[![Latest Version](https://badge.fury.io/py/tree-cat.svg)](https://pypi.python.org/pypi/tree-cat)
[![DOI](https://zenodo.org/badge/93913649.svg)](https://zenodo.org/badge/latestdoi/93913649)

A Bayesian latent tree model of multivariate multinomial data.

## Intended Use

Tree-Cat is appropriate for analyzing medium-sized tabular data with
categorical and ordinal values, possibly with missing observations.

| | Tree-Cat supports |
| --- | --- |
| **Feature Types** | categorical, ordinal, binomial, multinomial |
| **# Rows (n)** | 1000-100K |
| **# Features (p)** | 10-1000 |
| **# Cells (n &times; p)** | <1M |
| **# Categories** | 2-10ish |
| **Max Ordinal** | 10ish |
| **Missing obervations?** | yes |
| **Repeated observations?** | yes |
| **Sparse data?** | no, use something else |
| **Unsupervised** | yes |
| **Semisupervised** | yes |
| **Supervised** | no, use something else |

## Installing

First install `numba`. Then

```sh
$ pip install tree-cat
```

## Quick Start

1.  Create two csv files: a `schema.csv` and a `data.csv`.

2.  Import the csv files into treecat's internal format.

    ```python
    from treecat.format import import_data

    import_data('schema.csv', 'data.csv', 'dataset.pkl.gz')
    ```

3.  Train an ensemble model on your dataset.

    ```python
    from treecat.format import pickle_load, pickle_dump
    from treecat.config import train_ensemble
    from treecat.training import train_ensemble
    from treecat.config import make_default_config

    dataset = pickle_load('dataset.pkl.gz')
    config = make_default_config()
    ensemble = train_ensemble(dataset['ragged_index'],
                              dataset['data'], config)
    pickle_dump(ensemble, 'ensemble.plk.gz')
    ```

4.  Load your trained model into a server

    ```python
    from treecat.serving import serve_ensemble

    server = serve_ensemble('ensemble.plk.gz')
    ```

5.  Run queries against the server, e.g. compute marginals
    ```python
    server.sample(100, np.ones(V)).mean(axis=1)
    ```
    or compute a latent correlation matrix
    ```python
    print(server.correlation())
    ```

## The Model

Let `V` be a set of vertices (one vertex per feature).<br />
Let `C[v]` be the dimension of the `v`th feature.<br />
Let `N` be the number of datapoints.<br />
Let `K[n,v]` be the number of observations of feature `v` in row `n`
(e.g. 1 for a categorical variable, 0 for missing data, or
`k` for an ordinal value with minimum 0 and maximum `k`).

Tree-Cat is the following generative model:
```bugs
E ~ UniformSpanningTree(V)    # An undirected tree.
for v in V:
    Pv[v] ~ Dirichlet(size = [M], alpha = 1/2)
for (u,v) in E:
    Pe[u,v] ~ Dirichlet(size = [M,M], alpha = 1/(2M))
    assume Pv[u] == sum(Pe[u,v], axis=1)
    assume Pv[v] == sum(Pe[u,v], axis=0)
for v in V:
    for i in 1:M:
        Q[v,i] ~ Dirichlet(size = [C[v]])
for n in 1:N:
    for v in V:
        X[n,v] ~ Categorical(Pv[v])
    for (u,v) in E:
        (X[n,u],X[n,v]) ~ Categorical(Pe[u,v])
    for v in V:
        Z[n,v] ~ Multinomial(Q[v,X[n,v]], count = K[n,v])
```
where we've avoided adding an arbitrary root to the tree, and instead presented
the model as a manifold with overlapping variables and constraints.

## The Inference Algorithm

This package implements fully Bayesian MCMC inference using subsample-annealed
Gibbs sampling. There are two pieces of latent state that are sampled:

- Latent classes for each row for each vertex.
  These are sampled by single-site Gibbs sampling with a linear
  subsample-annealing schedule.

- The latent tree structure is sampled by randomly removing an edge
  and replacing it. Since removing an edge splits the graph into two
  connected components, the only replacement locations that are feasible
  are those that re-connect the graph.

The single-site Gibbs sampler uses dynamic programming to simultaneously sample
the complete latent assignment vector for each row. A dynamic programming
program is created each time the tree structure changes. This program is
interpreted by various virtual machines for different purposes (training the
model, sampling from the posterior, computing log probability of the posterior).
The virtual machine for training is jit-compiled using numba.

## License

Copyright (c) 2017 Fritz Obermeyer. <br />
Tree-Cat is licensed under the [Apache 2.0 License](/LICENSE).
