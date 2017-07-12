# How to contribute

TreeCat is designed to be a fun research platform both for Bayesian algorithms
for multivariate multinomial inference, and for practical data analysis.

## Setting up a conda env

To install treecat for development, you should be able to

```sh
git clone git@github.com:posterior/treecat
cd treecat
conda env create -f environment.yml
pip install -e .
```

## Testing

We use tools for testing, linting, and formatting during development.
You can run these easily using the Makefile

```sh
make lint     # Check for syntax errors using flake8.
make format   # Automatically reformat code using isort and yapf.
make test     # Run tests with py.test.
```

Please make sure you `make format` and `make test`
before submitting a pull request.
