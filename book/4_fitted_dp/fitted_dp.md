---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(fitted_dp)=
# Fitted dynamic programming algorithms

```{contents}
:local:
```

This closely follows [these lecture slides](https://shamulent.github.io/RL_2023/Lectures/fitteddp_annotated.pdf).

We borrow these definitions from the {ref}`mdps` chapter:

```{code-cell} ipython3
from typing import NamedTuple, Callable, Optional
from jaxtyping import Float, Int, Array
import jax.numpy as np
from jax import grad
from functools import partial

class Transition(NamedTuple):
    s: int
    a: int
    r: float

type Trajectory = list[Transition]

def get_num_actions(trajectories: list[Trajectory]) -> int:
    """Get the number of actions in the dataset. Assumes actions range from 0 to A-1."""
    return max(max(t.a for t in τ) for τ in trajectories) + 1

type State = Float[Array, "..."]  # arbitrary shape

# assume finite `A` actions and f outputs an array of Q-values
type QFunction = Callable[[State, int], Float[Array, "A"]]

def Q_zero(A: int) -> QFunction:
    """A Q-function that always returns zero."""
    return lambda s, a: np.zeros(A)

# a deterministic time-dependent policy
type Policy = Callable[[State, int], int]

def q_to_greedy(Q: QFunction) -> Policy:
    return lambda s, h: np.argmax(Q(s, h))
```

## Introduction

The {ref}`mdps` chapter discussed the case of **finite** MDPs, where the state and action spaces $\mathcal{S}$ and $\mathcal{A}$ were finite.
This allowed us to compute the value function for a given policy exactly.
In this chapter, we consider the case of **large** or **continuous** state spaces, where the state space is too large to be enumerated.
In this case, we need to *approximate* the value function and Q-function using methods from **supervised learning**.

## Fitted value iteration

So far, how have we computed the optimal value function in MDPs with finite state spaces?

In a {ref}`finite-horizon MDP <finite_horizon_mdps>`, we can use {prf:ref}`a DP algorithm <pi_star_dp>`, working backwards from the end of the time horizon, to compute the optimal value function exactly.

In a {ref}`infinite-horizon MDP <infinite_horizon_mdps>`, we can use the {ref}`value iteration <value_iteration>` algorithm, which iterates the Bellman optimality operator {eq}`bellman_optimality_operator` to approximately compute the optimal value function.

Our existing approaches represent the value function, and the MDP itself,
in matrix notation. But what happens if the state space is extremely large, or even infinite (e.g. real-valued)?
Then computing a weighted sum over all possible next states, which is required to compute the Bellman operator,
becomes intractable.

Instead, we will need to use *function approximation* methods to represent the value function in an
alternative way. To approximate the value function and Q-function, we will turn to techniques from *supervised learning*,
another branch of machine learning that focuses on learning patterns from labelled data.

In particular, suppose we have a dataset of $N$ trajectories $\tau_1, \dots, \tau_N \sim \rho_{\pi}$ from some policy $\pi$.
Let us indicate the trajectory index in the superscript, so that

$$
\tau_i = \{ s_0^i, a_0^i, r_0^i, s_1^i, a_1^i, r_1^i, \dots, s_{\hor-1}^i, a_{\hor-1}^i, r_{\hor-1}^i \}.
$$

We want to _learn_ the optimal value function from this dataset.
Recall that the {prf:ref}`Bellman consistency equations for the optimal policy <bellman_consistency_optimal>`
don't depend on an actual policy:

$$
Q_\hi^\star(s, a) = r(s, a) + \E_{s' \sim P(s, a)} [\max_{a'} Q_{\hi+1}^\star(s', a')]
$$

Our goal is to use the dataset to find a Q-function that satisfies these equations.

Let's reframe the problem to make the connection to supervised learning more clear.
We can think of the arguments to the Q-function -- i.e. the current state, action, and timestep $\hi$ --
as the inputs $x$, and the r.h.s. of the above equation as the label $f(x)$. Note that the r.h.s. can also be expressed as a **conditional expectation**:

$$
f(x) = \E [y \mid x] \quad \text{where} \quad y = r(s_\hi, a_\hi) + \max_{a'} Q^\star_{\hi + 1}(s', a').
$$

This is precisely the kind of supervised learning problem we can solve using a dataset of labelled samples.
In particular, we'll use the following fact:

:::{prf:theorem} Conditional expectation minimizes mean squared error
:label: conditional_expectation_minimizes_mse

$$
\arg\min_{f} \E[(y - f(x))^2] = (x \mapsto \E[y \mid x])
$$
:::

::::{prf:proof}
We can decompose the mean squared error as

$$
\begin{aligned}
\E[(y - f(x))^2] &= \E[ (y - \E[y \mid x] + \E[y \mid x] - f(x))^2 ] \\
&= \E[ (y - \E[y \mid x])^2 ] + \E[ (\E[y \mid x] - f(x))^2 ] + 2 \E[ (y - \E[y \mid x])(\E[y \mid x] - f(x)) ] \\
\end{aligned}
$$

:::{attention}
Use the law of iterated expectations to show that the last term is zero.
:::

So the first term is the irreducible error, and the second term is the error due to the approximation,
which is minimized at $0$ when $f(x) = \E[y \mid x]$.
::::

Thus a natural approach for approximating the conditional expectation
is to draw $N$ samples $(x_i, y_i)$ from the joint distribution of $x$ and $y$,
and use the _sample average_ $\sum_{i=1}^N (y_i - f(x_i))^2 / N$ to approximate the mean squared error.
Then we use a _fitting method_ to find a function that minimizes this objective
and thus approximates the conditional expectation.
This approach is called **empirical risk minimization**.

:::{prf:definition} Empirical risk minimization
:label: empirical_risk_minimization

Given a dataset of samples $(x_1, y_1), \dots, (x_N, y_N)$, empirical risk minimization seeks to find a function $f$ (from some class of functions $\mathcal{F}$) that minimizes the empirical risk:

$$
\hat f = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N (y_i - f(x_i))^2
$$

We will cover the details of the minimization process in {ref}`the next section <supervised_learning>`.
:::

:::{attention}
Why is it important that we constrain our search to a class of functions $\mathcal{F}$?

Hint: Consider the function $f(x) = \sum_{i=1}^N y_i \mathbb{1}_{\{ x = x_i \}}$. What is the empirical risk of this function? Would you consider it a good approximation of the conditional expectation?
:::

Our above dataset would give us $N \cdot \hor$ samples in the dataset:

$$
x_{i \hi} = (s_\hi^i, a_\hi^i, \hi) \qquad y_{i \hi} = r(s_\hi^i, a_\hi^i) + \max_{a'} Q^\star_{\hi + 1}(s_{\hi + 1}^i, a')
$$

```{code-cell} ipython3
def transform_data(
    trajectories: list[Trajectory],
    compute_label: Callable[[State, int, float], float],
):
    """
    Transform the dataset of trajectories into a dataset for supervised learning.
    """
    X = []
    y = []
    for τ in trajectories:
        for h in range(len(τ)-1):
            s, a, r = τ[h]
            label = compute_label(s, a, r)
            X.append((s, a, h))
            y.append(label)
        # Add the last state of this trajectory to the dataset
        X.append((τ[-1].s, τ[-1].a, len(τ)-1))
        y.append(τ[-1].r)
    return X, y

def compute_label_optimal(f: QFunction, s: State, a: int, r: float):
    return r + f(s, h+1).max()

def compute_label_from_policy(f: QFunction, π: Policy, s: State, a: int, r: float):
    return r + f(s, h+1)[π(s, h+1)]
```

Then we can use empirical risk minimization to find a function $\hat f$ that approximates the optimal Q-function.

```{code-cell} ipython3
def fit(X: Float[Array, "N D"], y: Float[Array, "N"]) -> QFunction:
    # Use your favorite empirical risk minimization algorithm here
    # We will see some examples in the next section
    pass
```

But notice that the definition of $y_{i \hi}$ depends on the Q-function itself!
How might we address this?
Recall that we faced the same issue when trying to use the Bellman consistency equations in the infinite-horizon case {eq}`bellman_consistency_infinite`.
There, we used *fixed point iteration* to compute the desired value function.
We can apply the same strategy here, using the $\hat f$ from the previous iteration to compute the $y_{i \hi}$,
and then using this to get the next iterate.

```{code-cell} ipython3
def fitted_q_iteration(trajectories: list[Trajectory], epochs: int, Q_init: Optional[QFunction] = None) -> QFunction:
    """
    Run fitted Q-function iteration using the given dataset.
    Returns an estimate of the optimal Q-function.
    """
    Q_hat = Q_init or Q_zero(get_num_actions(trajectories))
    for _ in range(epochs):
        X, y = transform_data(trajectories, partial(compute_label_optimal, Q_hat))
        Q_hat = fit(X, y)
    return Q_hat

def fitted_evaluation(trajectories: list[Trajectory], π: Policy, epochs: int, Q_init: Optional[QFunction] = None) -> QFunction:
    """
    Run fitted policy evaluation using the given dataset.
    Returns an estimate of the Q-function of the given policy.
    """
    Q_hat = Q_init or Q_zero(get_num_actions(trajectories))
    for _ in range(epochs):
        X, y = transform_data(trajectories, partial(compute_label_from_policy, Q_hat, π))
        Q_hat = fit(X, y)
    return Q_hat

def fitted_policy_iteration(trajectories: list[Trajectory], epochs: int, π_init: Optional[Policy] = None):
    """Run fitted policy iteration using the given dataset."""
    π = π_init or (lambda s, h: 0)  # constant zero policy
    for _ in range(epochs):
        Q_hat = fitted_evaluation(trajectories, π)
        π = q_to_greedy(Q_hat)
    return π
```

(supervised_learning)=
## Supervised learning

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

:::{prf:algorithm} Gradient descent
Letting $L(\theta) \in \mathbb{R}$ denote the empirical risk in terms of the parameters,
the gradient descent algorithm updates the parameters according to the rule

$$
\theta^{t+1} = \theta^t - \eta \nabla_\theta L(\theta^t)
$$

where $\eta > 0$ is the **learning rate**.
:::

```{code-cell} ipython3
type Params = Float[Array, "D"]

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

### Linear regression

In linear regression, we assume that the function $f$ is linear in the parameters:

$$
\mathcal{F} = \{ x \mapsto \theta^\top x \mid \theta \in \mathbb{R}^D \}
$$

```{code-cell} ipython3
def fit_linear(X: Float[Array, "N D"], y: Float[Array, "N"]):
    """Fit a linear model to the given dataset using ordinary least squares."""
    θ = np.linalg.lstsq(X, y, rcond=None)[0]
    return lambda x: np.dot(x, θ)
```

This function class is extremely simple and only contains linear functions.
To expand its expressivity, we can _transform_ the input $x$ using some feature function $\phi$,
i.e. $\widetilde x = \phi(x)$, and then fit a linear model in the transformed space instead.

### Neural networks

In neural networks, we assume that the function $f$ is a composition of linear functions (represented by matrices $W_i$) and non-linear activation functions (denoted by $\sigma$):

$$
\mathcal{F} = \{ x \mapsto \sigma(W_L \sigma(W_{L-1} \dots \sigma(W_1 x + b_1) \dots + b_{L-1}) + b_L) \}
$$

where $W_i \in \mathbb{R}^{D_{i+1} \times D_i}$ and $b_i \in \mathbb{R}^{D_{i+1}}$ are the parameters of the $i$-th layer, and $\sigma$ is the activation function.

