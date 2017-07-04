# TreeCat

A Bayesian latent tree model of multivariate multinomial data.

## Intended Use

You might want to use TreeCat if:
you have medium-sized tabular data with categorical and ordinal values,
possibly with missing observations.

| what you need | what TreeCat supports |
| --- | --- |
| **Feature Types** | categorical, ordinal, binomial, multinomial |
| **# Rows (n)** | 1000-100K |
| **# Features (p)** | 20-1000 |
| **# Cells (n &times; p)** | <10M |
| **# Categories** | 2-10ish |
| **# Max Ordinal** | 10ish |
| **Missing obervations?** | yes |
| **Repeated observations?** | yes |
| **Sparse data?** | there are better methods |
| **Unsupervised** | yes |
| **Semisupervised** | yes |
| **Supervised** | there are better methods |

## The Model

Let `V` be a set of vertices (one vertex per feature).<br />
Let `C[v]` be the dimension of the `v`th feature.<br />
Let `N` be the number of datapoints.<br />
Let `K[n,v]` be the number of observations of feature `v` in row `n`
(e.g. 1 for a categorical variable, 0 for missing data, or
`k` for an ordinal value with minimum 0 and maximum `k`).

TreeCat is the following generative model:
```bugs
E ~ UniformSpanningTree(V)
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

## Inference

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
schedule is recomputed each time the tree structure changes. This schedule is
interpreted by various virtual machines for different purposes (training the
model, sampling from the posterior, computing log probability of the posterior).
The virtual machine for training is jit-compiled using numba.

## License

Copyright (c) 2017 Fritz Obermeyer. <br />
TreeCat is licensed under the [Apache 2.0 License](/LICENSE).
