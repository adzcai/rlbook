---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Supervised learning

## Introduction

This section will cover the details of implementing the `fit` function above:
That is, how to use a dataset of labelled samples $(x_1, y_1), \dots, (x_N, y_N)$ to find a function $f$ that minimizes the empirical risk.
This requires two ingredients:

1. A **function class** $\mathcal{F}$ to search over
2. A **fitting method** for minimizing the empirical risk over this class

The two main function classes we will cover are **linear models** and **neural networks**.
Both of these function classes are *parameterized* by some parameters $\theta$,
and the fitting method will search over these parameters to minimize the empirical risk:

:::{prf:definition} Parameterized empirical risk minimization
:label: parameterized_empirical_risk_minimization

Given a dataset of samples $(x_1, y_1), \dots, (x_N, y_N)$ and a class of functions $\mathcal{F}$ parameterized by $\theta$,
we to find a parameter (vector) $\hat \theta$ that minimizes the empirical risk:

$$
\hat \theta = \arg\min_{\theta} \frac{1}{N} \sum_{i=1}^N (y_i - f_\theta(x_i))^2
$$
:::

The most common fitting method for parameterized models is **gradient descent**.

:::{prf:definition} Gradient descent
:label: gd_def

Letting $L(\theta) \in \mathbb{R}$ denote the empirical risk in terms of the parameters,
the gradient descent algorithm updates the parameters according to the rule

$$
\theta^{t+1} = \theta^t - \eta \nabla_\theta L(\theta^t)
$$

where $\eta > 0$ is the **learning rate**.
:::

```{code-cell}
:tags: [hide-input]

from jaxtyping import Float, Array
from collections.abc import Callable
```

```{code-cell}
Params = Float[Array, " D"]


def gradient_descent(
    loss: Callable[[Params], float],
    θ_init: Params,
    η: float,
    epochs: int,
):
    """
    Run gradient descent to minimize the given loss function
    (expressed in terms of the parameters).
    """
    θ = θ_init
    for _ in range(epochs):
        θ = θ - η * grad(loss)(θ)
    return θ
```

## Linear regression

In linear regression, we assume that the function $f$ is linear in the parameters:

$$
\mathcal{F} = \{ x \mapsto \theta^\top x \mid \theta \in \mathbb{R}^D \}
$$

This function class is extremely simple and only contains linear functions.
To expand its expressivity, we can _transform_ the input $x$ using some feature function $\phi$,
i.e. $\widetilde x = \phi(x)$, and then fit a linear model in the transformed space instead.

```{code-cell}
def fit_linear(X: Float[Array, "N D"], y: Float[Array, " N"], φ=lambda x: x):
    """Fit a linear model to the given dataset using ordinary least squares."""
    X = vmap(φ)(X)
    θ = np.linalg.lstsq(X, y, rcond=None)[0]
    return lambda x: np.dot(φ(x), θ)
```

## Neural networks

In neural networks, we assume that the function $f$ is a composition of linear functions (represented by matrices $W_i$) and non-linear activation functions (denoted by $\sigma$):

$$
\mathcal{F} = \{ x \mapsto \sigma(W_L \sigma(W_{L-1} \dots \sigma(W_1 x + b_1) \dots + b_{L-1}) + b_L) \}
$$

where $W_i \in \mathbb{R}^{D_{i+1} \times D_i}$ and $b_i \in \mathbb{R}^{D_{i+1}}$ are the parameters of the $i$-th layer, and $\sigma$ is the activation function.

This function class is much more expressive and contains many more parameters.
This makes it more susceptible to overfitting on smaller datasets,
but also allows it to represent more complex functions.
In practice, however, neural networks exhibit interesting phenomena during training,
and are often able to generalize well even with many parameters.

Another reason for their popularity is the efficient **backpropagation** algorithm for computing the gradient of the empirical risk with respect to the parameters.
Essentially, the hierarchical structure of the neural network,
i.e. computing the output of the network as a composition of functions,
allows us to use the chain rule to compute the gradient of the output with respect to the parameters of each layer.

{cite}`nielsen_neural_2015` provides a comprehensive introduction to neural networks and backpropagation.
