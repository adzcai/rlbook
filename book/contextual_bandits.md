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

(contextual_bandits)=
# Contextual bandits

In the above MAB environment, the reward distributions of the arms
remain constant. However, in many real-world settings, we might receive
additional information that affects these distributions. For example, in
the online advertising case where each arm corresponds to an ad we could
show the user, we might receive information about the user's preferences
that changes how likely they are to click on a given ad. We can model
such environments using **contextual bandits**.

:::{prf:definition} Contextual bandit
:label: contextual_bandit

At each timestep $t$, a new *context*
$x_t$ is drawn from some distribution $\nu_{\text{x}}$. The learner gets
to observe the context, and choose an action $a_t$ according to some
context-dependent policy $\pi_t(x_t)$. Then, the learner observes the
reward from the chosen arm $r_t \sim \nu^{a_t}(x_t)$. The reward
distribution also depends on the context.
:::

+++

Assuming our context is *discrete*, we can just perform the same
algorithms, treating each context-arm pair as its own arm. This gives us
an enlarged MAB of $K |\mathcal{X}|$ arms.

:::{attention}
Write down the UCB algorithm for this enlarged MAB. That is, write an
expression for $\pi_t(x_t) = \argmax_a \dots$.
:::

Recall that running UCB for $T$ timesteps on an MAB with $K$ arms
achieves a regret bound of $\tilde{O}(\sqrt{TK})$. So in this problem,
we would achieve regret $\tilde{O}(\sqrt{TK|\mathcal{X}|})$ in the
contextual MAB, which has a polynomial dependence on $|\mathcal{X}|$.
But in a situation where we have large, or even infinitely many
contexts, e.g. in the case where our context is a continuous value, this
becomes intractable.

Note that this "enlarged MAB" treats the different contexts as entirely
unrelated to each other, while in practice, often contexts are *related*
to each other in some way: for example, we might want to advertise
similar products to users with similar preferences. How can we
incorporate this structure into our solution?

+++

(lin_ucb)=
## Linear contextual bandits

We want to model the *mean reward* of arm $k$ as a function of the
context, i.e. $\mu^k(x)$. One simple model is the *linear* one:
$\mu^k(x) = x^\top \theta^k$, where $x \in \mathcal{X} = \mathbb{R}^d$ and
$\theta^k \in \mathbb{R}^d$ describes a *feature direction* for arm $k$. Recall
that **supervised learning** gives us a way to estimate a conditional
expectation from samples: We learn a *least squares* estimator from the
timesteps where arm $k$ was selected:
$$\hat \theta_t^k = \argmin_{\theta \in \mathbb{R}^d} \sum_{\{ i \in [t] : a_i = k \}} (r_i - x_i^\top \theta)^2.$$
This has the closed-form solution known as the *ordinary least squares*
(OLS) estimator:

:::{math}
:label: ols_bandit

\begin{aligned}
    \hat \theta_t^k          & = (A_t^k)^{-1} \sum_{\{ i \in [t] : a_i = k \}} x_i r_i \\
    \text{where} \quad A_t^k & = \sum_{\{ i \in [t] : a_i = k \}} x_i x_i^\top.
\end{aligned}
:::

We can now apply the UCB algorithm in this environment in order to
balance *exploration* of new arms and *exploitation* of arms that we
believe to have high reward. But how should we construct the upper
confidence bound? Previously, we treated the pulls of an arm as i.i.d.
samples and used Hoeffding's inequality to bound the distance of the
sample mean, our estimator, from the true mean. However, now our
estimator is not a sample mean, but rather the OLS estimator above {eq}`ols_bandit`. Instead, we'll use **Chebyshev's
inequality** to construct an upper confidence bound.

:::{prf:theorem} Chebyshev's inequality
:label: chebyshev

For a random variable $Y$ such that
$\E Y = 0$ and $\E Y^2 = \sigma^2$,
$$|Y| \le \beta \sigma \quad \text{with probability} \ge 1 - \frac{1}{\beta^2}$$
:::

Since the OLS estimator is known to be unbiased (try proving this
yourself), we can apply Chebyshev's inequality to
$x_t^\top (\hat \theta_t^k - \theta^k)$:

$$\begin{aligned}
    x_t^\top \theta^k \le x_t^\top \hat \theta_t^k + \beta \sqrt{x_t^\top (A_t^k)^{-1} x_t} \quad \text{with probability} \ge 1 - \frac{1}{\beta^2}
\end{aligned}$$

:::{attention}
We haven't explained why $x_t^\top (A_t^k)^{-1} x_t$ is the correct
expression for the variance of $x_t^\top \hat \theta_t^k$. This result
follows from some algebra on the definition of the OLS estimator {eq}`ols_bandit`.
:::

The first term is exactly our predicted reward $\hat \mu^k_t(x_t)$. To
interpret the second term, note that
$$x_t^\top (A_t^k)^{-1} x_t = \frac{1}{N_t^k} x_t^\top (\Sigma_t^k)^{-1} x_t,$$
where
$$\Sigma_t^k = \frac{1}{N_t^k} \sum_{\{ i \in [t] : a_i = k \}} x_i x_i^\top$$
is the empirical covariance matrix of the contexts (assuming that the
context has mean zero). That is, the learner is encouraged to choose
arms when $x_t$ is *not aligned* with the data seen so far, or if arm
$k$ has not been explored much and so $N_t^k$ is small.

We can now substitute these quantities into UCB to get the **LinUCB**
algorithm:

```{code-cell}
class LinUCBPseudocode(Agent):
    def __init__(
        self, K: int, T: int, D: int, lam: float, get_c: Callable[[int], float]
    ):
        super().__init__(K, T)
        self.lam = lam
        self.get_c = get_c
        self.contexts = [None for _ in range(K)]
        self.A = np.repeat(lam * np.eye(D)[...], K)
        self.targets = np.zeros(K, D)
        self.w = np.zeros(K, D)

    def choose_arm(self, context: Float[Array, " D"]):
        c = self.get_c(self.count)
        scores = self.w @ context + c * np.sqrt(
            context.T @ np.linalg.solve(self.A, context)
        )
        return random_argmax(scores)

    def update_history(self, context: Float[Array, " D"], arm: int, reward: int):
        self.A[arm] += np.outer(context, context)
        self.targets[arm] += context * reward
        self.w[arm] = np.linalg.solve(self.A[arm], self.targets[arm])
```

:::{attention}
Note that the matrix $A_t^k$ above might not be invertible. When does this occur? One way to address this is to include a $\lambda I$ regularization term to ensure that $A_t^k$ is invertible. This is equivalent to solving a *ridge regression* problem instead of the unregularized least squares problem. Implement this solution. TODO SOLUTION CURRENTLY SHOWN
:::

+++

$c_t$ is similar to the $\log (2t/\delta')$ term of UCB: It controls the
width of the confidence interval. Here, we treat it as a tunable
parameter, though in a theoretical analysis, it would depend on $A_t^k$
and the probability $\delta$ with which the bound holds.

Using similar tools for UCB, we can also prove an $\tilde{O}(\sqrt{T})$
regret bound. The full details of the analysis can be found in Section 3 of {cite}`agarwal_reinforcement_2022`.
