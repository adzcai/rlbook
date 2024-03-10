(sec@fitted_dp)=
# Fitted dynamic programming algorithms

```{contents}
:local:
```

This closely follows [these lecture slides](https://shamulent.github.io/RL_2023/Lectures/fitteddp_annotated.pdf).

## Introduction

The {ref}`sec@mdps` chapter discussed the case of **finite** MDPs, where the state and action spaces $\mathcal{S}$ and $\mathcal{A}$ were finite.
This allowed us to compute the value function for a given policy exactly.
In this chapter, we consider the case of **large** or **continuous** state spaces, where the state space is too large to be enumerated.
In this case, we need to *approximate* the value function and Q-function.

## Fitted value iteration

So far, how have we computed the optimal value function?

In finite-horizon MDPs {ref}`sec@finite_horizon_mdps`, we used {ref}`a dynamic programming algorithm <opt_dynamic_programming>`, working backwards from the end of the time horizon, to compute the value function exactly.

Recall the {ref}`sec@value_iteration` algorithm: We initialize some vector $v \in \mathbb{R}^{|mathcal{S}|}$ and iterate the {eq}`Bellman optimality operator <bellman_optimality_operator>` until convergence.


