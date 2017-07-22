# How to contribute

TreeCat is designed to be a fun research platform both for Bayesian algorithms
for multivariate multinomial inference, and for practical data analysis.
We welcome issues and bug reports
[on GitHub](https://github.com/posterior/treecat/issues/new).
Before you submit a pull request, please set up an environment and test as
described below. 

## Setting up a conda env

To install TreeCat for development,
clone and install in a fresh conda env using
[miniconda](https://conda.io/miniconda.html) or
[Anaconda](https://www.continuum.io/downloads):

```sh
git clone git@github.com:posterior/treecat
cd treecat
conda env create -f environment.3.yml    # Or environment.2.yml for Python 2.7.
source activate treecat3
pip install -e .
```

Some changes are difficult to get working in both Python 2 and 3.
We recommend setting up a pair of conda envs when testing unicode support as in
the `treecat.format` module.

## Testing

We use standard tools for testing, linting, and formatting during development.
You can run these easily using the Makefile

```sh
make lint     # Check for syntax errors using flake8.
make format   # Automatically reformat code using isort and yapf.
make test     # Run tests with py.test.
```

Please make sure you `make format` and `make test`
before submitting a pull request.

## Coding Conventions

We use the following variable name conventions for often-used variables:

- `N` = number of rows in a dataset. This is the `n` of statisticians.
- `V` = number of features = number of vertices in a latent tree.
  This is the `p` of statisticians.
- `E` = `V - 1` = number of edges in a latent tree.
- `K` = `V * (V - 1) // 2` = number of edges in the complete undirected graph
  on `V` vertices.
- `M` = `config['model_num_clusters']` = number of latent classes.
- `R` = number of columns in any ragged array
  in TreeCat's internal format of multinomial data.
  This is the sum of the number of categories of all observed features.
- `ragged_index` is a `V+1`-long numpy array of offsets into ragged arrays.
- `config` is a global configuration dict,
  as defined in [`config.py`](treecat/config.py).
- `probs` is any array of competing probabilities,
  often non-normalized and interpreted as odds.
- `logits` is any array of competing log-probabilities,
  often non-normalized and interpreted as log-odds.
- `data` is either an `[N, V]`-shaped array of data in external dense format
  or an `[N, R]`-shaped array of data in internal ragged format.
  This is "data" in the sense of statisticians.
