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

(bandits)=
# Multi-Armed Bandits

```{code-cell}
:tags: [hide-input]

from jaxtyping import Float, Array
import numpy as np

# from bokeh.plotting import figure, show, output_notebook
import latexify
from abc import ABC, abstractmethod  # "Abstract Base Class"
from typing import Callable, Union
import matplotlib.pyplot as plt

import solutions.bandits as solutions

np.random.seed(184)

# output_notebook()  # set up bokeh

plt.style.use("fivethirtyeight")


def random_argmax(ary: Array) -> int:
    max_idx = np.flatnonzero(ary == ary.max())
    return np.random.choice(max_idx).item()


latex = latexify.algorithmic(
    prefixes={"mab"},
    identifiers={"arm": "a_t", "reward": "r", "means": "mu"},
    use_math_symbols=True,
    escape_underscores=False,
)
```

The **multi-armed bandits** (MAB) setting is a simple setting for studying the basic challenges of RL. In this setting, an agent repeatedly chooses from a fixed set of actions, called **arms**, each of which has an associated reward distribution. The agent’s goal is to maximize the total reward it receives over some time period.

| States | Actions | Rewards                             |
| :----: | :-----: | :---------------------------------: |
| None   | Finite  | $\mathcal{A} \to \triangle([0, 1])$ |

In particular, we’ll spend a lot of time discussing the **Exploration-Exploitation Tradeoff**: should the agent choose new actions to learn more about the environment, or should it choose actions that it already knows to be good?

::::{prf:example} Online advertising
:label: advertising

Let’s suppose you, the agent, are an advertising company. You have $K$ different ads that you can show to users; For concreteness, let’s suppose there’s just a single user. You receive $1$ reward if the user clicks the ad, and $0$ otherwise. Thus, the unknown *reward distribution* associated to each ad is a Bernoulli distribution defined by the probability that the user clicks on the ad. Your goal is to maximize the total number of clicks by the user.
::::

::::{prf:example} Clinical trials
:label: clinical_trials

Suppose you’re a pharmaceutical company, and you’re testing a new drug. You have $K$ different dosages of the drug that you can administer to patients. You receive $1$ reward if the patient recovers, and $0$ otherwise. Thus, the unknown *reward distribution* associated to each dosage is a Bernoulli distribution defined by the probability that the patient recovers. Your goal is to maximize the total number of patients that recover.
::::

In this chapter, we will introduce the multi-armed bandits setting, and discuss some of the challenges that arise when trying to solve problems in this setting. We will also introduce some of the key concepts that we will use throughout the book, such as regret and exploration-exploitation tradeoffs.

+++

## Introduction

::::{prf:remark} Namesake
:label: multi-armed

The name “multi-armed bandits” comes from slot machines in casinos, which are often called “one-armed bandits” since they have one arm (the lever) and take money from the player.
::::

Let $K$ denote the number of arms. We’ll label them $0, \dots, K-1$ and use *superscripts* to indicate the arm index; since we seldom need to raise a number to a power, this won’t cause much confusion. In this chapter, we’ll consider the **Bernoulli bandit** setting from the examples above, where arm $k$ either returns reward $1$ with probability $\mu^k$ or $0$ otherwise. The agent gets to pull an arm $T$ times in total. We can formalize the Bernoulli bandit in the following Python code:

```{code-cell}
class MAB:
    """
    The Bernoulli multi-armed bandit environment.

    :param means: the means (success probabilities) of the reward distributions for each arm
    :param T: the time horizon
    """

    def __init__(self, means: Float[Array, " K"], T: int):
        assert all(0 <= p <= 1 for p in means)
        self.means = means
        self.T = T
        self.K = self.means.size
        self.best_arm = random_argmax(self.means)

    def pull(self, k: int) -> int:
        """Pull the `k`-th arm and sample from its (Bernoulli) reward distribution."""
        reward = np.random.rand() < self.means[k].item()
        return +reward
```

```{code-cell}
mab = MAB(means=np.array([0.1, 0.8, 0.4]), T=100)
```

In pseudocode, the agent’s interaction with the MAB environment can be
described by the following process:

```{code-cell}
@latex
def mab_loop(mab: MAB, agent: "Agent") -> int:
    for t in range(mab.T):
        arm = agent.choose_arm()  # in 0, ..., K-1
        reward = mab.pull(arm)
        agent.update_history(arm, reward)


mab_loop
```

The `Agent` class stores the pull history and uses it to decide which arm to pull next. Since we are working with Bernoulli bandits, we can summarize the pull history concisely in a $\mathbb{N}^{K \times 2}$ array.

```{code-cell}
class Agent(ABC):
    def __init__(self, K: int, T: int):
        """The MAB agent that decides how to choose an arm given the past history."""
        self.K = K
        self.T = T
        self.rewards = []  # for plotting
        self.choices = []
        self.history = np.zeros((K, 2), dtype=int)

    @abstractmethod
    def choose_arm(self) -> int:
        """Choose an arm of the MAB. Algorithm-specific."""
        ...

    @property
    def count(self) -> int:
        """The number of pulls made."""
        return len(self.rewards)

    def update_history(self, arm: int, reward: int):
        self.rewards.append(reward)
        self.choices.append(arm)
        self.history[arm, reward] += 1
```

What’s the *optimal* strategy for the agent, i.e. the one that achieves
the highest expected reward? Convince yourself that the agent should try
to always pull the arm with the highest expected reward:

$$\mu^\star := \max_{k \in [K]} \mu^k.$$

The goal, then, can be rephrased as to minimize the **regret**, defined
below:

::::{prf:definition} Regret
:label: regret

The agent’s **regret** after $T$ timesteps is defined as

$$
\text{Regret}_T := \sum_{t=0}^{T-1} \mu^\star - \mu^{a_t}.
$$
::::

```{code-cell}
def regret_per_step(mab: MAB, agent: Agent):
    """Get the difference from the average reward of the optimal arm. The sum of these is the regret."""
    return [mab.means[mab.best_arm] - mab.means[arm] for arm in agent.choices]
```

Note that this depends on the *true means* of the pulled arms, *not* the actual
observed rewards.
We typically think of this as a random variable where
the randomness comes from the agent’s strategy (i.e. the sequence of
actions $a_0, \dots, a_{T-1}$).

Throughout the chapter, we will try to upper bound the regret of various
algorithms in two different senses:

1.  Upper bound the *expected regret,* i.e. show
    $\E[\text{Regret}_T] \le M_T$.

2.  Find a *high-probability* upper bound on the regret, i.e. show
    $\P(\text{Regret}_T \le M_{T, \delta}) \ge 1-\delta$.

Note that these two different approaches say very different things about the regret. The first approach says that the *average* regret is at most $M_T$. However, the agent might still achieve higher regret on many runs. The second approach says that, *with high probability*, the agent will achieve regret at most $M_{T, \delta}$. However, it doesn’t say anything about the regret in the remaining $\delta$ fraction of runs, which might be arbitrarily high.

We’d like to achieve **sublinear regret** in expectation, i.e. $\E[\text{Regret}_T] = o(T)$. That is, as we learn more about the environment, we’d like to be able to exploit that knowledge to take the optimal arm as often as possible.

The rest of the chapter comprises a series of increasingly sophisticated
MAB algorithms.

```{code-cell}
:tags: [hide-input]

def plot_strategy(mab: MAB, agent: Agent):
    plt.figure(figsize=(10, 6))

    # plot reward and cumulative regret
    plt.plot(np.arange(mab.T), np.cumsum(agent.rewards), label="reward")
    cum_regret = np.cumsum(regret_per_step(mab, agent))
    plt.plot(np.arange(mab.T), cum_regret, label="cumulative regret")

    # draw colored circles for arm choices
    colors = ["red", "green", "blue"]
    color_array = [colors[k] for k in agent.choices]
    plt.scatter(np.arange(mab.T), np.zeros(mab.T), c=color_array, label="arm")

    # labels and title
    plt.xlabel("timestep")
    plt.legend()
    plt.title(f"{agent.__class__.__name__} reward and regret")
    plt.show()
```

## Pure exploration (random guessing)

A trivial strategy is to always choose arms at random (i.e. "pure
exploration").

```{code-cell}
:label: pure_exploration

class PureExploration(Agent):
    def choose_arm(self):
        """Choose an arm uniformly at random."""
        return solutions.pure_exploration_choose_arm(self)
```

Note that

$$
\E_{a_t \sim \text{Unif}([K])}[\mu^{a_t}] = \bar \mu = \frac{1}{K} \sum_{k=1}^K \mu^k
$$

so the expected regret is simply

$$
\begin{aligned}
    \E[\text{Regret}_T] &= \sum_{t=0}^{T-1} \E[\mu^\star - \mu^{a_t}] \\
    &= T (\mu^\star - \bar \mu) > 0.
\end{aligned}
$$

This scales as $\Theta(T)$, i.e. *linear* in the number of timesteps $T$. There’s no learning here: the agent doesn’t use any information about the environment to improve its strategy. You can see that the distribution over its arm choices always appears "(uniformly) random".

```{code-cell}
agent = PureExploration(mab.K, mab.T)
mab_loop(mab, agent)
plot_strategy(mab, agent)
```

## Pure greedy

How might we improve on pure exploration? Instead, we could try each arm
once, and then commit to the one with the highest observed reward. We’ll
call this the **pure greedy** strategy.

```{code-cell}
:label: pure_greedy

class PureGreedy(Agent):
    def choose_arm(self):
        """Choose the arm with the highest observed reward on its first pull."""
        return solutions.pure_greedy_choose_arm(self)
```

Note we’ve used superscripts $r^k$ during the exploration phase to
indicate that we observe exactly one reward for each arm. Then we use
subscripts $r_t$ during the exploitation phase to indicate that we
observe a sequence of rewards from the chosen greedy arm $\hat k$.

How does the expected regret of this strategy compare to that of pure
exploration? We’ll do a more general analysis in the following section.
Now, for intuition, suppose there’s just $K=2$ arms, with Bernoulli
reward distributions with means $\mu^0 > \mu^1$.

Let’s let $r^0$ be the random reward from the first arm and $r^1$ be the
random reward from the second. If $r^0 > r^1$, then we achieve zero
regret. Otherwise, we achieve regret $T(\mu^0 - \mu^1)$. Thus, the
expected regret is simply:

$$
\begin{aligned}
    \E[\text{Regret}_T] &= \P(r^0 < r^1) \cdot T(\mu^0 - \mu^1) + c \\
    &= (1 - \mu^0) \mu^1 \cdot T(\mu^0 - \mu^1) + c
\end{aligned}
$$

Which is still $\Theta(T)$, the same as pure exploration!

```{code-cell}
agent = PureGreedy(mab.K, mab.T)
mab_loop(mab, agent)
plot_strategy(mab, agent)
```

The cumulative regret is a straight line because the regret only depends on the arms chosen and not the actual reward observed. In fact, if the greedy algorithm happens to get lucky on the first set of pulls, it may act entirely optimally for that episode! But its _average_ regret is what measures its effectiveness.

+++

(etc)=
## Explore-then-commit

We can improve the pure greedy algorithm as follows: let’s reduce the variance of the reward estimates by pulling each arm $N_{\text{explore}}> 1$ times before committing. This is called the **explore-then-commit** strategy. Note that the “pure greedy” strategy above is just the special case where
$N_{\text{explore}}= 1$.

```{code-cell}
class ExploreThenCommit(Agent):
    def __init__(self, K: int, T: int, N_explore: int):
        super().__init__(K, T)
        self.N_explore = N_explore

    def choose_arm(self):
        return solutions.etc_choose_arm(self)
```

```{code-cell}
agent = ExploreThenCommit(mab.K, mab.T, mab.T // 15)
mab_loop(mab, agent)
plot_strategy(mab, agent)
```

Notice that now, the graphs are much more consistent, and the algorithm finds the true optimal arm and sticks with it much more frequently. We would expect ETC to then have a better (i.e. lower) average regret. Can we prove this?

+++

(etc-regret-analysis)=
### ETC regret analysis

Let’s analyze the expected regret of the explore-then-commit strategy by splitting it up
into the exploration and exploitation phases.

#### Exploration phase.

This phase takes $N_{\text{explore}}K$ timesteps. Since at each step we
incur at most $1$ regret, the total regret is at most
$N_{\text{explore}}K$.

#### Exploitation phase.

This will take a bit more effort. We’ll prove that for any total time $T$, we can choose $N_{\text{explore}}$ such that with arbitrarily high probability, the regret is sublinear.

Let $\hat k$ denote the arm chosen after the exploration phase. We know the regret from the
exploitation phase is

$$T_{\text{exploit}} (\mu^\star - \mu^{\hat k}) \qquad \text{where} \qquad T_{\text{exploit}} := T - N_{\text{explore}}K.$$

So we’d like to bound $\mu^\star - \mu^{\hat k} = o(1)$ (as a function
of $T$) in order to achieve sublinear regret. How can we do this?

Let’s define $\Delta^k = \hat \mu^k - \mu^k$ to denote how far the mean
estimate for arm $k$ is from the true mean. How can we bound this
quantity? We’ll use the following useful inequality for i.i.d. bounded
random variables:

:::{prf:theorem} Hoeffding’s inequality
:label: hoeffding

Let $X_0, \dots, X_{n-1}$ be i.i.d. random variables with
$X_i \in [0, 1]$ almost surely for each $i \in [n]$. Then for any
$\delta > 0$,

$$\P\left( \left| \frac{1}{n} \sum_{i=1}^n (X_i - \E[X_i]) \right| > \sqrt{\frac{\ln(2/\delta)}{2n}} \right) \le \delta.$$
:::

The proof of this inequality is beyond the scope of this book. See {cite}`vershynin_high-dimensional_2018` Chapter 2.2.

We can apply this directly to the rewards for a given arm $k$, since the rewards from that arm are i.i.d.:

:::{math}
:label: "hoeffding-etc"

\P\left(|\Delta^k | > \sqrt{\frac{\ln(2/\delta)}{2N_{\text{explore}}}} \right) \le \delta.
:::

But note that we can’t apply this to arm $\hat k$ directly since
$\hat k$ is itself a random variable. Instead, we need to “uniform-ize”
this bound across *all* the arms, i.e. bound the error across all the
arms simultaneously, so that the resulting bound will apply *no matter
what* $\hat k$ “crystallizes” to.

The **union bound** provides a simple way to do this:

:::{prf:theorem} Union bound
:label: union_bound

Consider a set of events $A_0, \dots, A_{n-1}$. Then

$$\P(\exists i \in [n]. A_i) \le \sum_{i=0}^{n-1} \P(A_i).$$

In
particular, if $\P(A_i) \ge 1 - \delta$ for each $i \in [n]$, we have

$$\P(\forall i \in [n]. A_i) \ge 1 - n \delta.$$
:::

**Exercise:** Prove the second statement above.

Applying the union bound across the arms for the l.h.s. event of {eq}`hoeffding-etc`, we have

$$
\begin{aligned}
    \P\left( \forall k \in [K], |\Delta^k | \le \sqrt{\frac{\ln(2/\delta)}{2N_{\text{explore}}}} \right) &\ge 1-K\delta
\end{aligned}
$$

Then to apply this bound to $\hat k$ in particular, we
can apply the useful trick of “adding zero”:

$$
\begin{aligned}
    \mu^{k^\star} - \mu^{\hat k} &= \mu^{k^\star} - \mu^{\hat k} + (\hat \mu^{k^\star} - \hat \mu^{k^\star}) + (\hat \mu^{\hat k} - \hat \mu^{\hat k}) \\
    &= \Delta^{\hat k} - \Delta^{k^*} + \underbrace{(\hat \mu^{k^\star} - \hat \mu^{\hat k})}_{\le 0 \text{ by definition of } \hat k} \\
    &\le 2 \sqrt{\frac{\ln(2K/\delta')}{2N_{\text{explore}}}} \text{ with probability at least } 1-\delta'
\end{aligned}
$$

where we’ve set $\delta' = K\delta$. Putting this all
together, we’ve shown that, with probability $1 - \delta'$,

$$\text{Regret}_T \le N_{\text{explore}}K + T_{\text{exploit}} \cdot \sqrt{\frac{2\ln(2K/\delta')}{N_{\text{explore}}}}.$$

Note that it suffices for $N_{\text{explore}}$ to be on the order of
$\sqrt{T}$ to achieve sublinear regret. In particular, we can find the
optimal $N_{\text{explore}}$ by setting the derivative of the r.h.s. to
zero:

$$
\begin{aligned}
    0 &= K - T_{\text{exploit}} \cdot \frac{1}{2} \sqrt{\frac{2\ln(2K/\delta')}{N_{\text{explore}}^3}} \\
    N_{\text{explore}}&= \left( T_{\text{exploit}} \cdot \frac{\sqrt{\ln(2K/\delta')/2}}{K} \right)^{2/3}
\end{aligned}
$$

Plugging this into the expression for the regret, we
have (still with probability $1-\delta'$)

$$
\begin{aligned}
    \text{Regret}_T &\le 3 T^{2/3} \sqrt[3]{K \ln(2K/\delta') / 2} \\
    &= \tilde{O}(T^{2/3} K^{1/3}).
\end{aligned}
$$

The ETC algorithm is rather “abrupt” in that it switches from
exploration to exploitation after a fixed number of timesteps. In
practice, it’s often better to use a more gradual transition, which
brings us to the *epsilon-greedy* algorithm.

+++

## Epsilon-greedy

Instead of doing all of the exploration and then all of the exploitation
separately – which additionally requires knowing the time horizon
beforehand – we can instead interleave exploration and exploitation by,
at each timestep, choosing a random action with some probability. We
call this the **epsilon-greedy** algorithm.

```{code-cell}
class EpsilonGreedy(Agent):
    def __init__(self, K: int, T: int, get_epsilon: Callable[[int], float]):
        super().__init__(K, T)
        self.get_epsilon = get_epsilon

    def choose_arm(self):
        return solutions.epsilon_greedy_choose_arm(self)
```

```{code-cell}
agent = EpsilonGreedy(mab.K, mab.T, lambda t: 0.1)
mab_loop(mab, agent)
plot_strategy(mab, agent)
```

Note that we let $\epsilon$ vary over time. In particular, we might want to gradually *decrease* $\epsilon$ as we learn more about the reward distributions and no longer need to spend time exploring.

:::{attention}
What is the expected regret of the algorithm if we set $\epsilon$ to be a constant?
:::

It turns out that setting $\epsilon_t = \sqrt[3]{K \ln(t)/t}$ also achieves a regret of $\tilde O(t^{2/3} K^{1/3})$ (ignoring the logarithmic factors). (We will not prove this here.) TODO ADD PROOF CITATION

In ETC, we had to set $N_{\text{explore}}$ based on the total number of timesteps $T$. But the epsilon-greedy algorithm actually handles the exploration *automatically*: the regret rate holds for *any* $t$, and doesn’t depend on the final horizon $T$.

But the way these algorithms explore is rather naive: we’ve been exploring *uniformly* across all the arms. But what if we could be smarter about it, and explore *more* for arms that we’re less certain about?

+++

(ucb)=
## Upper Confidence Bound (UCB)

To quantify how *certain* we are about the mean of each arm, we’ll
compute *confidence intervals* for our estimators, and then choose the
arm with the highest *upper confidence bound*. This operates on the
principle of **the benefit of the doubt (i.e. optimism in the face of
uncertainty)**: we’ll choose the arm that we’re most optimistic about.

In particular, for each arm $k$ at time $t$, we’d like to compute some
upper confidence bound $M^k_t$ such that $\hat \mu^k_t \le M^k_t$ with
high probability, and then choose $a_t := \arg \max_{k \in [K]} M^k_t$.
But how should we compute $M^k_t$?

In [](etc-regret-analysis), we were able to compute this bound
using Hoeffding’s inequality, which assumes that the number of samples
is *fixed*. This was the case in ETC (where we pull each arm
$N_{\text{explore}}$ times), but in UCB, the number of times we pull
each arm depends on the agent’s actions, which in turn depend on the
random rewards and are therefore stochastic. So we *can’t* use
Hoeffding’s inequality directly.

Instead, we’ll apply the same trick we used in the ETC analysis: we’ll
use the **union bound** to compute a *looser* bound that holds
*uniformly* across all timesteps and arms. Let’s introduce some notation
to discuss this.

Let $N^k_t$ denote the (random) number of times arm $k$ has been pulled
within the first $t$ timesteps, and $\hat \mu^k_t$ denote the sample
average of those pulls. That is,

$$
\begin{aligned}
    N^k_t &:= \sum_{\tau=0}^{t-1} \mathbf{1} \{ a_\tau = k \} \\
    \hat \mu^k_t &:= \frac{1}{N^k_t} \sum_{\tau=0}^{t-1} \mathbf{1} \{ a_\tau = k \} r_\tau.
\end{aligned}
$$

To achieve the “fixed sample size” assumption, we’ll
need to shift our index from *time* to *number of samples from each
arm*. In particular, we’ll define $\tilde r^k_n$ to be the $n$th sample
from arm $k$, and $\tilde \mu^k_n$ to be the sample average of the first
$n$ samples from arm $k$. Then, for a fixed $n$, this satisfies the
“fixed sample size” assumption, and we can apply Hoeffding’s inequality
to get a bound on $\tilde \mu^k_n$.

So how can we extend our bound on $\tilde\mu^k_n$ to $\hat \mu^k_t$?
Well, we know $N^k_t \le t$ (where equality would be the case if and
only if we had pulled arm $k$ every time). So we can apply the same
trick as last time, where we uniform-ize across all possible values of
$N^k_t$:

$$
\begin{aligned}
    \P\left( \forall n \le t, |\tilde \mu^k_n - \mu^k | \le \sqrt{\frac{\ln(2/\delta)}{2n}} \right) &\ge 1-t\delta.
\end{aligned}
$$

In particular, since $N^k_t \le t$, and $\tilde \mu^k_{N^k_t} = \hat \mu^k_t$ by definition, we have

$$
\begin{aligned}
    \P\left( |\hat \mu^k_t - \mu^k | \le \sqrt{\frac{\ln(2t/\delta')}{2N^k_t}} \right) &\ge 1-\delta' \text{ where } \delta' := t \delta.
\end{aligned}
$$

This bound would then suffice for applying the UCB algorithm! That is, the upper confidence bound for arm $k$ would be

$$M^k_t := \hat \mu^k_t + \sqrt{\frac{\ln(2t/\delta')}{2N^k_t}},$$

where we can choose $\delta'$ depending on how tight we want the interval to be. A smaller $\delta'$ would give us a larger and higher-confidence interval, and vice versa. We can now use this to define the UCB algorithm.

```{code-cell}
class UCB(Agent):
    def __init__(self, K: int, T: int, delta: float):
        super().__init__(K, T)
        self.delta = delta

    def choose_arm(self):
        return solutions.ucb_choose_arm(self)
```

Intuitively, UCB prioritizes arms where:

1.  $\hat \mu^k_t$ is large, i.e. the arm has a high sample average, and
    we’d choose it for *exploitation*, and

2.  $\sqrt{\frac{\ln(2t/\delta')}{2N^k_t}}$ is large, i.e. we’re still
    uncertain about the arm, and we’d choose it for *exploration*.

As desired, this explores in a smarter, *adaptive* way compared to the
previous algorithms. Does it achieve lower regret?

```{code-cell}
agent = UCB(mab.K, mab.T, 0.05)
mab_loop(mab, agent)
plot_strategy(mab, agent)
```

### UCB regret analysis

First we’ll bound the regret incurred at each timestep. Then we’ll bound
the *total* regret across timesteps.

For the sake of analysis, we’ll use a slightly looser bound that applies
across the whole time horizon and across all arms. We’ll omit the
derivation since it’s very similar to the above (walk through it
yourself for practice).

$$
\begin{aligned}
    \P\left(\forall k \le K, t < T. |\hat \mu^k_t - \mu^k | \le B^k_t \right) &\ge 1-\delta'' \\
    \text{where} \quad B^k_t &:= \sqrt{\frac{\ln(2TK/\delta'')}{2N^k_t}}.
\end{aligned}
$$

Intuitively, $B^k_t$ denotes the *width* of the CI for arm $k$ at time
$t$. Then, assuming the above uniform bound holds (which occurs with
probability $1-\delta''$), we can bound the regret at each timestep as
follows:

$$
\begin{aligned}
    \mu^\star - \mu^{a_t} &\le \hat \mu^{k^*}_t + B_t^{k^*} - \mu^{a_t} && \text{applying UCB to arm } k^\star \\
    &\le \hat \mu^{a_t}_t + B^{a_t}_t - \mu^{a_t} && \text{since UCB chooses } a_t = \arg \max_{k \in [K]} \hat \mu^k_t + B_t^{k} \\
    &\le 2 B^{a_t}_t && \text{since } \hat \mu^{a_t}_t - \mu^{a_t} \le B^{a_t}_t \text{ by definition of } B^{a_t}_t \\
\end{aligned}
$$

Summing this across timesteps gives

$$
\begin{aligned}
    \text{Regret}_T &\le \sum_{t=0}^{T-1} 2 B^{a_t}_t \\
    &= \sqrt{2\ln(2TK/\delta'')} \sum_{t=0}^{T-1} (N^{a_t}_t)^{-1/2} \\
    \sum_{t=0}^{T-1} (N^{a_t}_t)^{-1/2} &= \sum_{t=0}^{T-1} \sum_{k=1}^K \mathbf{1}\{ a_t = k \} (N^k_t)^{-1/2} \\
    &= \sum_{k=1}^K \sum_{n=1}^{N_T^k} n^{-1/2} \\
    &\le K \sum_{n=1}^T n^{-1/2} \\
    \sum_{n=1}^T n^{-1/2} &\le 1 + \int_1^T x^{-1/2} \ \mathrm{d}x \\
    &= 1 + (2 \sqrt{x})_1^T \\
    &= 2 \sqrt{T} - 1 \\
    &\le 2 \sqrt{T} \\
\end{aligned}
$$

Putting everything together gives

$$
\begin{aligned}
    \text{Regret}_T &\le 2 K \sqrt{2T \ln(2TK/\delta'')} && \text{with probability } 1-\delta'' \\
    &= \tilde O(K\sqrt{T})
\end{aligned}
$$

In fact, we can do a more sophisticated analysis to trim off a factor of
$\sqrt{K}$ and show $\text{Regret}_T = \tilde O(\sqrt{TK})$.

+++

### Lower bound on regret (intuition)

Is it possible to do better than $\Omega(\sqrt{T})$ in general? In fact,
no! We can show that any algorithm must incur $\Omega(\sqrt{T})$ regret
in the worst case. We won’t rigorously prove this here, but the
intuition is as follows.

The Central Limit Theorem tells us that with $T$ i.i.d. samples from
some distribution, we can only learn the mean of the distribution to
within $\Omega(1/\sqrt{T})$ (the standard deviation). Then, since we get
$T$ samples spread out across the arms, we can only learn each arm’s
mean to an even looser degree.

That is, if two arms have means that are within about $1/\sqrt{T}$, we
won’t be able to confidently tell them apart, and will sample them about
equally. But then we’ll incur regret
$$\Omega((T/2) \cdot (1/\sqrt{T})) = \Omega(\sqrt{T}).$$

+++

(thompson_sampling)=
## Thompson sampling and Bayesian bandits

So far, we’ve treated the parameters $\mu^0, \dots, \mu^{K-1}$ of the
reward distributions as *fixed*. Instead, we can take a **Bayesian**
approach where we treat them as random variables from some **prior
distribution**. Then, upon pulling an arm and observing a reward, we can
simply *condition* on this observation to exactly describe the
**posterior distribution** over the parameters. This fully describes the
information we gain about the parameters from observing the reward.

From this Bayesian perspective, the **Thompson sampling** algorithm
follows naturally: just sample from the distribution of the optimal arm,
given the observations!

```{code-cell}
class Distribution(ABC):
    @abstractmethod
    def sample(self) -> Float[Array, " K"]: ...

    @abstractmethod
    def update(self, arm: int, reward: float): ...
```

```{code-cell}
class ThompsonSampling(Agent):
    def __init__(self, K: int, T: int, prior: Distribution):
        super().__init__(K, T)
        self.distribution = prior

    def choose_arm(self):
        means = self.distribution.sample()
        return random_argmax(means)

    def update_history(self, arm: int, reward: int):
        super().update_history(arm, reward)
        self.distribution.update(arm, reward)
```

In other words, we sample each arm proportionally to how likely we think
it is to be optimal, given the observations so far. This strikes a good
exploration-exploitation tradeoff: we explore more for arms that we’re
less certain about, and exploit more for arms that we’re more certain
about. Thompson sampling is a simple yet powerful algorithm that
achieves state-of-the-art performance in many settings.

:::{prf:example} Bayesian Bernoulli bandit
:label: bayesian_bernoulli

We’ve been working in the Bernoulli bandit setting, where arm $k$ yields a reward of $1$ with probability $\mu^k$ and no reward otherwise. The vector of success probabilities $\boldsymbol{\mu} = (\mu^1, \dots, \mu^K)$ thus describes the entire MAB.

Under the Bayesian perspective, we think of $\boldsymbol{\mu}$ as a *random* vector drawn from some prior distribution $\pi(\boldsymbol{\mu})$. For example, we might have $\pi$ be the Uniform distribution over the unit hypercube $[0, 1]^K$, that is,

$$\pi(\boldsymbol{\mu}) = \begin{cases}
    1 & \text{if } \boldsymbol{\mu}\in [0, 1]^K \\
    0 & \text{otherwise}
\end{cases}$$

In this case, upon viewing some reward, we can exactly calculate the **posterior** distribution of $\boldsymbol{\mu}$ using Bayes’s rule (i.e. the definition of conditional probability):

$$
\begin{aligned}
    \P(\boldsymbol{\mu} \mid a_0, r_0) &\propto \P(r_0 \mid a_0, \boldsymbol{\mu}) \P(a_0 \mid \boldsymbol{\mu}) \P(\boldsymbol{\mu}) \\
    &\propto (\mu^{a_0})^{r_0} (1 - \mu^{a_0})^{1-r_0}.
\end{aligned}
$$

This is the PDF of the
$\text{Beta}(1 + r_0, 1 + (1 - r_0))$ distribution, which is a conjugate
prior for the Bernoulli distribution. That is, if we start with a Beta
prior on $\mu^k$ (note that $\text{Unif}([0, 1]) = \text{Beta}(1, 1)$),
then the posterior, after conditioning on samples from
$\text{Bern}(\mu^k)$, will also be Beta. This is a very convenient
property, since it means we can simply update the parameters of the Beta
distribution upon observing a reward, rather than having to recompute
the entire posterior distribution from scratch.
:::

```{code-cell}
class Beta(Distribution):
    def __init__(self, K: int, alpha: int = 1, beta: int = 1):
        self.alphas = np.full(K, alpha)
        self.betas = np.full(K, beta)

    def sample(self):
        return np.random.beta(self.alphas, self.betas)

    def update(self, arm: int, reward: int):
        self.alphas[arm] += reward
        self.betas[arm] += 1 - reward
```

```{code-cell}
beta_distribution = Beta(mab.K)
agent = ThompsonSampling(mab.K, mab.T, beta_distribution)
mab_loop(mab, agent)
plot_strategy(mab, agent)
```

It turns out that asymptotically, Thompson sampling is optimal in the
following sense. {cite}`lai_asymptotically_1985` prove an
*instance-dependent* lower bound that says for *any* bandit algorithm,

$$\liminf_{T \to \infty} \frac{\E[N_T^k]}{\ln(T)} \ge \frac{1}{\text{KL}(\mu^k \parallel \mu^\star)}$$

where

$$\text{KL}(\mu^k \parallel \mu^\star) = \mu^k \ln \frac{\mu^k}{\mu^\star} + (1 - \mu^k) \ln \frac{1 - \mu^k}{1 - \mu^\star}$$

measures the **Kullback-Leibler divergence** from the Bernoulli
distribution with mean $\mu^k$ to the Bernoulli distribution with mean
$\mu^\star$. It turns out that Thompson sampling achieves this lower
bound with equality! That is, not only is the error *rate* optimal, but
the *constant factor* is optimal as well.

+++

## Contextual bandits

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
### Linear contextual bandits

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

+++

## Summary


