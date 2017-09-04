![](https://cdn.rawgit.com/posterior/treecat/master/doc/logo.png)

# TreeCat

[![Docs](https://readthedocs.org/projects/treecat/badge/?version=latest)](http://treecat.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/posterior/treecat.svg?branch=master)](https://travis-ci.org/posterior/treecat)
[![Latest Version](https://badge.fury.io/py/pytreecat.svg)](https://pypi.python.org/pypi/pytreecat)
[![DOI](https://zenodo.org/badge/93913649.svg)](https://zenodo.org/badge/latestdoi/93913649)

## Intended Use

TreeCat is an inference engine for machine learning and Bayesian inference.
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

If you already have [Numba](http://numba.pydata.org) installed,
you should be able to simply

```sh
pip install pytreecat
```

If you're new to Numba, we recommend installing it using
[miniconda](https://conda.io/miniconda.html) or
[Anaconda](https://www.continuum.io/downloads).

If you want to install TreeCat for development,
then clone the source code and create a new conda env
```sh
git clone git@github.com:posterior/treecat
cd treecat
conda env create -f environment.3.yml
source activate treecat3
pip install -e .
```

## Quick Start

1.  Format your data as a [`data.csv`](treecat/testdata/tiny_data.csv)
    file with a header row.
    It's fine to include extra columns that won't be used.

    Contents of [`data.csv`](treecat/testdata/tiny_data.csv):

    | title     | genre    | decade | rating |
    | --------- | -------- | ------ | ------ |
    | vertigo   | thriller | 1950s  | 5      |
    | up        | family   | 2000s  | 3      |
    | desk set  | comedy   | 1950s  | 4      |
    | santapaws | family   | 2010s  |        |
    | ...       | ...      | ...    | ...    |

2.  Generate two schema files
    [`types.csv`](treecat/testdata/tiny_types.csv) and
    [`values.csv`](treecat/testdata/tiny_values.csv)
    using TreeCat's `guess-schema` command:

    ```sh
    $ treecat guess-schema data.csv types.csv values.csv
    ```

    Contents of [`types.csv`](treecat/testdata/tiny_types.csv):

    | name   | type        | total | unique | singletons |
    | ------ | ----------- | ----- | ------ | ---------- |
    | title  |             |    11 |     11 |         11 |
    | genre  | categorical |    11 |      7 |          4 |
    | decade | categorical |    11 |      6 |          3 |
    | rating | ordinal     |    10 |      5 |          2 |

    Contents of [`values.csv`](treecat/testdata/tiny_values.csv):

    | name   | value    | count |
    | ------ | -------- | ----- |
    | genre  | drama    |     3 |
    | genre  | family   |     2 |
    | genre  | fantasy  |     2 |
    | decade | 1950s    |     3 |
    | ...    | ...      |   ... |
    
    You can manually fix any incorrectly guessed feature types,
    or add/remove feature values.
    TreeCat ignores any feature with an empty type field.

3.  Import your csv files into treecat's internal format.
    We'll call our dataset `dataset.pkz` (a gzipped pickle file).

    ```sh
    $ treecat import-data data.csv types.csv values.csv '' dataset.pkz
    ```

    (the empty argument '' is an optional structural prior that we ignore).

4.  Train an ensemble model on your dataset.
    This typically takes ~15minutes for a 1M cell dataset.

    ```sh
    $ treecat train dataset.pkz model.pkz
    ```

5.  Load your trained model into a server

    ```python
    from treecat.serving import serve_model
    server = serve_model('dataset.pkz', 'model.pkz')
    ```

6.  Run queries against the server.
    For example we can compute expecations
    ```python
    samples = server.sample(100, evidence={'genre': 'drama'})
    print(np.mean([s['rating'] for s in samples]))
    ```
    or explore feature structure through the latent correlation matrix
    ```python
    print(server.latent_correlation())
    ```

## Tuning Hyperparameters

TreeCat requires tuning of two parameters:
`learning_init_epochs` (like the number of iterations) and
`model_num_clusters` (the number of latent classes above each feature).
The easiest way to tune these is to do grid search using the `treecat.validate` module
with a csv file of example parameters.

Contents of [`tuning.csv`](treecat/testdata/tuning.csv):

| model_num_clusters | learning_init_epochs |
| ------------------ | -------------------- |
|                  2 |                    2 |
|                  2 |                    3 |
|                  4 |                    2 |
|                ... |                  ... |

```sh
# This reads parameters from tuning.csv and dumps results to tuning.pkz
$ treecat.validate tune-csv dataset.pkz tuning.csv tuning.pkz
```

The `tune-csv` command prints its results, but if you want to seem them later, you can

```sh
$ treecat.format cat tuning.pkz
```

## The Server Interface

TreeCat's
[server](https://github.com/posterior/treecat/blob/master/treecat/serving.py)
interface supports primitives for Bayesian inference and
tools to inspect latent structure:

- `server.sample(N, evidence=None)`
  draws `N` samples from the joint posterior distribution over observable data,
  optionally conditioned on `evidence`.
  
- `server.logprob(rows, evidence=None)`
  computes posterior log probability of `data`,
  optionally conditioned on `evidence`.

- `server.median(evidence)`
  computes L1-loss-minimizing estimates, conditioned on `evidence`.

- `server.observed_perplexity()`
  computes the [perplexity](https://en.wikipedia.org/wiki/Perplexity)
  (a soft measure of cardinality) of each observed feature.

- `server.latent_perplexity()`
  computes the perplexity of the latent class behind each observed feature.

- `server.latent_correlation()`
  computes the latent-latent correlation between each pair of latent variables.

- `server.estimate_tree()`
  computes a maximum a posteriori estimate of the latent tree structure.

- `server.sample_tree(N)`
  draws `N` samples from posterior distribution over the latent tree structures.

## The Model

TreeCat's generative model is closest to Zhang and Poon's Latent Tree Analysis [1],
with the notable difference that TreeCat fixes exactly one latent node per observed node.
TreeCat is historically a descendent of Mansinghka et al.'s CrossCat, a model in which latent nodes ("views" or "kinds") are completely independent.
TreeCat addresses the same kind of high-dimensional categorical distribution
that Dunson and Xing's mixture-of-product-multinomial models [3] addresses.
While TreeCat currently supports only categorical and ordinal feature types,
it is straight-forward to generalize to other feature types with conjugate
priors such as real (normal-inverse-chi-squared), integer (gamma-Poisson), and
angular (von-Mises).
This generalization places it in the class of models high-dimensional heterogeneous data with Valera et al. [4].

Let `V` be a set of vertices (one vertex per feature).<br />
Let `C[v]` be the dimension of the `v`th feature.<br />
Let `N` be the number of datapoints.<br />
Let `K[n,v]` be the number of observations of feature `v` in row `n`
(e.g. 1 for a categorical variable, 0 for missing data, or
`k` for an ordinal value with minimum 0 and maximum `k`).

TreeCat is the following generative model:
```python
E ~ UniformSpanningTree(V)    # An undirected tree.
for v in V:
    Pv[v] ~ Dirichlet(size = [M], alpha = 1/2)
for (u,v) in E:
    Pe[u,v] ~ Dirichlet(size = [M,M], alpha = 1/(2*M))
    assume(Pv[u] == sum(Pe[u,v], axis = 1))
    assume(Pv[v] == sum(Pe[u,v], axis = 0))
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
collapsed Gibbs sampling. There are two pieces of latent state that are sampled:

- Latent class assignments for each row for each vertex (feature).
  These are sampled by single-site collapsed Gibbs sampler with a linear
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

## References

1. Nevin L. Zhang, Leonard K. M. Poon (2016) <br />
   [Latent Tree Analysis](https://arxiv.org/pdf/1610.00085.pdf)
2. Vikash Mansinghka, Patrick Shafto, Eric Jonas, Cap Petschulat, Max Gasner, Joshua B. Tenenbaum (2015) <br />
   [CrossCat: A Fully Bayesian Nonparametric Method for Analyzing Heterogeneous, High Dimensional Data](https://arxiv.org/pdf/1512.01272)
3. David B. Dunson, Chuanhua Xing (2012) <br />
   [Nonparametric Bayes Modeling of Multivariate Categorical Data](https://dx.doi.org/10.1198%2Fjasa.2009.tm08439)
4. Isabel Valera, Melanie F Pradier, Zoubin Ghahramani (2017) <br />
   [General Latent Feature Modeling for Data Exploration Tasks](https://arxiv.org/pdf/1707.08352).

## License

Copyright (c) 2017 Fritz Obermeyer. <br />
TreeCat is licensed under the [Apache 2.0 License](/LICENSE).
