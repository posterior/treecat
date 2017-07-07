![A Bayesian latent tree model](doc/cartoon.png)

# TreeCat

[![Build Status](https://travis-ci.org/posterior/treecat.svg?branch=master)](https://travis-ci.org/posterior/treecat)
[![Latest Version](https://badge.fury.io/py/pytreecat.svg)](https://pypi.python.org/pypi/pytreecat)
[![DOI](https://zenodo.org/badge/93913649.svg)](https://zenodo.org/badge/latestdoi/93913649)

## Intended Use

TreeCat is an inference engine intended to power higher-level machine learning tools.
TreeCat is appropriate for analyzing medium-sized tabular data with
categorical and ordinal values, possibly with missing observations.

|                        | TreeCat supports       |
| ---------------------- | ---------------------- |
| Feature Types          | categorical, ordinal   |
| # Rows (n)             | 1000-100K              |
| # Features (p)         | 10-1000                |
| # Cells (n &times; p)  | 10K-10M                |
| # Categories           | 2-10ish                |
| Max Ordinal            | 10ish                  |
| Missing obervations?   | yes                    |
| Repeated observations? | yes                    |
| Sparse data?           | no, use something else |
| Unsupervised           | yes                    |
| Semisupervised         | yes                    |
| Supervised             | no, use something else |

## Installing

First install `numba` (conda make this easy). Then

```sh
$ pip install pytreecat
```

## Quick Start

1.  Create two csv files: a `schema.csv` and a `data.csv`.
    The `schema.csv` specifies the column types in `data.csv`, for example
    
    | name   | type        |
    | ------ | ----------- |
    | genre  | categorical |
    | decade | categorical |
    | rating | ordinal     |
    
    The `data.csv` file should have column headings matching the schema,
    but it can have extra columns that will be ignored (e.g. `title`).
    
    | title     | genre    | decade | rating |
    | --------- | -------- | ------ | ------ |
    | vertigo   | thriller | 1950s  | 5      |
    | up        | family   | 2000s  | 3      |
    | desk set  | comedy   | 1950s  | 4      |
    | santapaws | family   | 2010s  | 1      |
    | chinatown | mystery  | 1970s  | 4      |

2.  Import your csv files into treecat's internal format. We'll call our dataset `dataset.pkl.gz`.

    ```python
    from treecat.format import import_data

    import_data('schema.csv', 'data.csv', 'dataset.pkl.gz')
    ```

3.  Train an ensemble model on your dataset.
    This typically takes ~15minutes for a 1M cell dataset.

    ```python
    from treecat.config import make_default_config
    from treecat.format import pickle_load, pickle_dump
    from treecat.training import train_ensemble

    dataset = pickle_load('dataset.pkl.gz')
    config = make_default_config()
    ensemble = train_ensemble(dataset['ragged_index'],
                              dataset['data'], config)
    # ...wait for a while...
    pickle_dump(ensemble, 'ensemble.plk.gz')
    ```

4.  Load your trained model into a server

    ```python
    from treecat.serving import serve_ensemble

    server = serve_ensemble('ensemble.plk.gz')
    ```

5.  Run queries against the server.
    For example we can compute marginals
    ```python
    server.sample(100, np.ones(V)).mean(axis=0)
    ```
    or compute a latent correlation matrix
    ```python
    print(server.correlation())
    ```
    
## The Server Interface

TreeCat's
[server](https://github.com/fritzo/treecat/blob/master/treecat/serving.py)
interface currently supports the two basic Bayesian operations:

- `server.sample(N, counts, data=None)`
  draws N samples from the joint posterior distribution, optionally conditioned on `data`.
  
- `server.logprob(data)` computes posterior log probability of data.

TreeCat's internal data representation is multinomial, and thus supports missing and repeated measurements, and even data adding. For example to compute conditional probability of data `A` given data `B`, we can simply compute

```py
cond = server.logprob(A + B) - server.logprob(B)
```

## The Model

Let `V` be a set of vertices (one vertex per feature).<br />
Let `C[v]` be the dimension of the `v`th feature.<br />
Let `N` be the number of datapoints.<br />
Let `K[n,v]` be the number of observations of feature `v` in row `n`
(e.g. 1 for a categorical variable, 0 for missing data, or
`k` for an ordinal value with minimum 0 and maximum `k`).

TreeCat is the following generative model:
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
TreeCat is licensed under the [Apache 2.0 License](/LICENSE).
