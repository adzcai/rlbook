---
jupytext:
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

(bandits)=
# Bandits

```{code-cell}
:tags: ["hide-input"]

from jaxtyping import Float
import numpy as np
from abc import ABC, abstractmethod  # "Abstract Base Class"
```

The **multi-armed bandits** (MAB) setting is a simple but powerful
setting for studying the basic challenges of RL. In this setting, an
agent repeatedly chooses from a fixed set of actions, called **arms**,
each of which has an associated reward distribution. The agent’s goal is
to maximize the total reward it receives over some time period.

In particular, we’ll spend a lot of time discussing the
**Exploration-Exploitation Tradeoff**: should the agent choose new
actions to learn more about the environment, or should it choose actions
that it already knows to be good?

::::{prf:example} Online advertising
:label: advertising

Let’s suppose you, the agent, are an advertising company. You have $K$
different ads that you can show to users; For concreteness, let’s
suppose there’s just a single user. You receive $1$ reward if the user
clicks the ad, and $0$ otherwise. Thus, the unknown *reward
distribution* associated to each ad is a Bernoulli distribution defined
by the probability that the user clicks on the ad. Your goal is to
maximize the total number of clicks by the user.
::::

::::{prf:example} Clinical trials
:label: clinical_trials

Suppose you’re a pharmaceutical company, and you’re testing a new drug.
You have $K$ different dosages of the drug that you can administer to
patients. You receive $1$ reward if the patient recovers, and $0$
otherwise. Thus, the unknown *reward distribution* associated to each
dosage is a Bernoulli distribution defined by the probability that the
patient recovers. Your goal is to maximize the total number of patients
that recover.
::::

In this chapter, we will introduce the multi-armed bandits setting, and
discuss some of the challenges that arise when trying to solve problems
in this setting. We will also introduce some of the key concepts that we
will use throughout the book, such as regret and
exploration-exploitation tradeoffs.

## Introduction

::::{prf:remark} Namesake
:label: multi-armed

The name “multi-armed bandits” comes from slot machines in casinos,
which are often called “one-armed bandits” since they have one arm (the
lever) and take money from the player.
::::

Let $K$ denote the number of arms. We’ll label them $0, \dots, K-1$ and
use *superscripts* to indicate the arm index; since we seldom need to
raise a number to a power, this hopefully won’t cause much confusion.
For simplicity, we’ll assume rewards are *bounded* between $0$ and $1$.
Then each arm has an unknown reward distribution
$\nu^k \in \Delta([0, 1])$ with mean $\mu^k = \E_{r \sim \nu^k} [r]$.

In pseudocode, the agent’s interaction with the MAB environment can be
described by the following process:

```{code-cell}
def mab_loop(mab: "MAB", agent: "Agent"):
    agent.init(mab.K, mab.T)
    for t in range(mab.T):
        arm = agent.choose_arm()  # in 0, ..., K-1
        reward = mab.pull(arm)
        agent.update_history(arm, reward)
```

where we define the `MAB` and `Agent` classes as follows:

```{code-cell}
class BernoulliMAB:
    def __init__(self, μ: Float[np.ndarray, "K"], T: int):
        """
        The Bernoulli multi-armed bandit environment.
        The mean (i.e. success probability) of the `k`-th arm is `μ[k]`.

        :param μ: the means of the reward distributions for each arm
        :param T: the time horizon
        """
        self.μ = μ
        self.T = T

    @property
    def K(self):
        return self.μ.size

    def pull(self, k: int) -> bool:
        """Pull the `k`-th arm and return the reward."""
        return np.random.rand() < self.μ[k]

class Agent(ABC):
    def __init__(self, K: int, T: int):
        """The MAB agent that decides how to choose an arm given the past history."""
        self.K = K
        self.T = T
        self.history = np.zeros((mab.K, 2))

    @abstractmethod
    def choose_arm(self) -> int:
        """Choose an arm of the MAB. Algorithm-specific."""
        ...

    @property
    def step(self):
        return self.history.sum()

    def update_history(self, arm: int, reward: bool):
        self.history[arm, +reward] += 1
```

What’s the *optimal* strategy for the agent, i.e. the one that achieves
the highest expected reward? Convince yourself that the agent should try
to always pull the arm with the highest expected reward
$\mu^\star := \max_{k \in [K]} \mu^k$.

The goal, then, can be rephrased as to minimize the **regret**, defined
below:

::::{prf:definition} Regret
:label: regret

The agent’s **regret** after $T$ timesteps is defined as

$$
\text{Regret}_T := \sum_{t=0}^{T-1} \mu^\star - \mu^{a_t}.
$$

Note that this depends on the *true means* of the pulled arms, *not* the actual
observed rewards.
We typically think of this as a random variable where
the randomness comes from the agent’s strategy (i.e. the sequence of
actions $a_0, \dots, a_{T-1}$).

Throughout the chapter, we will try to upper bound the regret of various
algorithms in two different senses:

1.  Upper bound the *expected* regret, i.e. show
    $\E[\text{Regret}_T] \le M_T$.

2.  Find a high-probability upper bound on the regret, i.e. show
    $\P(\text{Regret}_T \le M_{T, \delta}) \ge 1-\delta$.

Note that these two different approaches say very different things about
the regret. The first approach says that the *average* regret is at most
$M_T$. However, the agent might still achieve higher regret on many
runs. The second approach says that, *with high probability*, the agent
will achieve regret at most $M_{T, \delta}$. However, it doesn’t say
anything about the regret in the remaining $\delta$ fraction of runs,
which might be arbitrarily high.
::::

We’d like to achieve **sublinear regret** in expectation, i.e.
$\E[\text{Regret}_T] = o(T)$. That is, as we learn more about the
environment, we’d like to be able to exploit that knowledge to achieve
higher rewards.

The rest of the chapter comprises a series of increasingly sophisticated
MAB algorithms.

## Pure exploration (random guessing)

A trivial strategy is to always choose arms at random (i.e. "pure
exploration").

```{code-cell}
:label: pure_exploration

class PureExploration(Agent):
    def choose_arm(self):
        """Choose an arm uniformly at random."""
        return np.random.randint(self.step)
```

Note that

$$
\E_{a_t \sim \text{Unif}([K])}[\mu^{a_t}] = \bar \mu = \frac{1}{K} \sum_{k=1}^K \mu^k
$$

so the expected regret is simply

$$
\begin{align}
    \E[\text{Regret}_T] &= \sum_{t=0}^{T-1} \E[\mu^\star - \mu^{a_t}] \\
    &= T (\mu^\star - \bar \mu) > 0.
\end{align}
$$

This scales as $\Theta(T)$, i.e. *linear* in the number
of timesteps $T$. There’s no learning here: the agent doesn’t use any
information about the environment to improve its strategy.

## Pure greedy

How might we improve on pure exploration? Instead, we could try each arm
once, and then commit to the one with the highest observed reward. We’ll
call this the **pure greedy** strategy.

```{code-cell}
:label: pure_greedy

class PureGreedy(Agent):
    def choose_arm(self):
        """Choose the arm with the highest observed reward."""
        if self.step < self.K:
            # first K steps: choose each arm once
            return self.step

        if self.step == self.K:
            # after the first K steps: choose the arm with the highest observed reward
            self.greedy_arm = np.argmax(self.history[:, 1])

        return self.greedy_arm
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
\begin{align}
    \E[\text{Regret}_T] &= \P(r^0 < r^1) \cdot T(\mu^0 - \mu^1) + c \\
    &= (1 - \mu^0) \mu^1 \cdot T(\mu^0 - \mu^1) + c
\end{align}
$$

Which is still $\Theta(T)$, the same as pure exploration! Can we do
better?

## Explore-then-commit

We can improve the pure greedy algorithm as follows: let’s reduce the
variance of the reward estimates by pulling each arm
$N_{\text{explore}}> 1$ times before committing. This is called the
**explore-then-commit** strategy.

```{code-cell}
:label: etc

class ExploreThenCommit(Agent):
    def __init__(self, K: int, T: int, N_explore: int):
        super().__init__(K, T)
        self.N_explore = N_explore

    def choose_arm(self):
        if self.step < self.K * self.N_explore:
            # exploration phase: choose each arm N_explore times
            return self.step // self.N_explore

        # exploitation phase: choose the arm with the highest observed reward
        if self.step == self.K * self.N_explore:
            self.greedy_arm = np.argmax(self.history[:, 1])

        return self.greedy_arm
```

(Note that the “pure greedy” strategy is just the special case where
$N_{\text{explore}}= 1$.)

(etc-regret-analysis)=
### ETC regret analysis

Let’s analyze the expected regret of this strategy by splitting it up
into the exploration and exploitation phases.

#### Exploration phase.

This phase takes $N_{\text{explore}}K$ timesteps. Since at each step we
incur at most $1$ regret, the total regret is at most
$N_{\text{explore}}K$.

#### Exploitation phase.

This will take a bit more effort. We’ll prove that for any total time
$T$, we can choose $N_{\text{explore}}$ such that with arbitrarily high
probability, the regret is sublinear. We know the regret from the
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

(The proof of this inequality is beyond the scope of this book.) We can
apply this directly to the rewards for a given arm $k$, since the
rewards from that arm are i.i.d.:

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
\begin{align}
    \P\left( \forall k \in [K], |\Delta^k | \le \sqrt{\frac{\ln(2/\delta)}{2N_{\text{explore}}}} \right) &\ge 1-K\delta
\end{align}
$$

Then to apply this bound to $\hat k$ in particular, we
can apply the useful trick of “adding zero”:

$$
\begin{align}
    \mu^{k^\star} - \mu^{\hat k} &= \mu^{k^\star} - \mu^{\hat k} + (\hat \mu^{k^\star} - \hat \mu^{k^\star}) + (\hat \mu^{\hat k} - \hat \mu^{\hat k}) \\
    &= \Delta^{\hat k} - \Delta^{k^*} + \underbrace{(\hat \mu^{k^\star} - \hat \mu^{\hat k})}_{\le 0 \text{ by definition of } \hat k} \\
    &\le 2 \sqrt{\frac{\ln(2K/\delta')}{2N_{\text{explore}}}} \text{ with probability at least } 1-\delta'
\end{align}
$$

where we’ve set $\delta' = K\delta$. Putting this all
together, we’ve shown that, with probability $1 - \delta'$,

$$\text{Regret}_T \le N_{\text{explore}}K + T_{\text{exploit}} \cdot \sqrt{\frac{2\ln(2K/\delta')}{N_{\text{explore}}}}.$$

Note that it suffices for $N_{\text{explore}}$ to be on the order of
$\sqrt{T}$ to achieve sublinear regret. In particular, we can find the
optimal $N_{\text{explore}}$ by setting the derivative of the r.h.s. to
zero:

$$
\begin{align}
    0 &= K - T_{\text{exploit}} \cdot \frac{1}{2} \sqrt{\frac{2\ln(2K/\delta')}{N_{\text{explore}}^3}} \\
    N_{\text{explore}}&= \left( T_{\text{exploit}} \cdot \frac{\sqrt{\ln(2K/\delta')/2}}{K} \right)^{2/3}
\end{align}
$$

Plugging this into the expression for the regret, we
have (still with probability $1-\delta'$)

$$
\begin{align}
    \text{Regret}_T &\le 3 T^{2/3} \sqrt[3]{K \ln(2K/\delta') / 2} \\
    &= \tilde{O}(T^{2/3} K^{1/3}).
\end{align}
$$

The ETC algorithm is rather “abrupt” in that it switches from
exploration to exploitation after a fixed number of timesteps. In
practice, it’s often better to use a more gradual transition, which
brings us to the *epsilon-greedy* algorithm.

## Epsilon-greedy

Instead of doing all of the exploration and then all of the exploitation
separately – which additionally requires knowing the time horizon
beforehand – we can instead interleave exploration and exploitation by,
at each timestep, choosing a random action with some probability. We
call this the **epsilon-greedy** algorithm.

:::{prf:definition} Epsilon-greedy
:label: epsilon_greedy

**Input:** $\epsilon : \mathbb{N} \to [0, 1]$ $S^k \gets 0$ for each
$k \in [K]$ $N^k \gets 0$ for each $k \in [K]$ $k \sim \text{Unif}([K])$
$k \gets \arg \max_k \left(\frac{S^k}{N^k}\right)$ $r_t \sim \nu^k$
$S^k \gets S^k + r_t$ $N^k \gets N^k + 1$
:::

Note that we let $\epsilon$ vary over time. In particular we might want
to gradually *decrease* $\epsilon$ as we learn more about the reward
distributions over time.

It turns out that setting $\epsilon_t = \sqrt[3]{K \ln(t)/t}$ also
achieves a regret of $\tilde O(t^{2/3} K^{1/3})$ (ignoring the
logarithmic factors). (We will not prove this here.)

In ETC, we had to set $N_{\text{explore}}$ based on the total number of
timesteps $T$. But the epsilon-greedy algorithm actually handles the
exploration *automatically*: the regret rate holds for *any* $t$, and
doesn’t depend on the final horizon $T$.

But the way these algorithms explore is rather naive: we’ve been
exploring *uniformly* across all the arms. But what if we could be
smarter about it, and explore *more* for arms that we’re less certain
about?

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
\begin{align}
    N^k_t &:= \sum_{\tau=0}^{t-1} \mathbf{1} \{ a_\tau = k \} \\
    \hat \mu^k_t &:= \frac{1}{N^k_t} \sum_{\tau=0}^{t-1} \mathbf{1} \{ a_\tau = k \} r_\tau.
\end{align}
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
\begin{align}
    \P\left( \forall n \le t, |\tilde \mu^k_n - \mu^k | \le \sqrt{\frac{\ln(2/\delta)}{2n}} \right) &\ge 1-t\delta.
\end{align}
$$

In particular, since $N^k_t \le t$, and
$\tilde \mu^k_{N^k_t} = \hat \mu^k_t$ by definition, we have

$$
\begin{align}
    \P\left( |\hat \mu^k_t - \mu^k | \le \sqrt{\frac{\ln(2t/\delta')}{2N^k_t}} \right) &\ge 1-\delta' \text{ where } \delta' := t \delta.
\end{align}
$$

This bound would then suffice for applying the UCB
algorithm! That is, the upper confidence bound for arm $k$ would be
$$M^k_t := \hat \mu^k_t + \sqrt{\frac{\ln(2t/\delta')}{2N^k_t}},$$ where
we can choose $\delta'$ depending on how tight we want the interval to
be. A smaller $\delta'$ would give us a larger yet higher-confidence
interval, and vice versa. We can now use this to define the UCB
algorithm.

:::{prf:definition} Upper Confidence Bound (UCB)
:label: ucb

**Input:** $\delta' \in (0, 1)$
$k \gets \arg \max_{k' \in [K]} \frac{S^{k'}}{N^{k'}} + \sqrt{\frac{\ln(2t/\delta')}{2 N^{k'}}}$
$r_t \sim \nu^k$ $S^k \gets S^k + r_t$ $N^k \gets N^k + 1$

:::

**Exercise:** As written, this ignores the issue that we divide by
$N^k = 0$ for all arms at the beginning. How should we resolve this
issue?

Intuitively, UCB prioritizes arms where:

1.  $\hat \mu^k_t$ is large, i.e. the arm has a high sample average, and
    we’d choose it for *exploitation*, and

2.  $\sqrt{\frac{\ln(2t/\delta')}{2N^k_t}}$ is large, i.e. we’re still
    uncertain about the arm, and we’d choose it for *exploration*.

As desired, this explores in a smarter, *adaptive* way compared to the
previous algorithms. Does it achieve lower regret?

### UCB regret analysis

First we’ll bound the regret incurred at each timestep. Then we’ll bound
the *total* regret across timesteps.

For the sake of analysis, we’ll use a slightly looser bound that applies
across the whole time horizon and across all arms. We’ll omit the
derivation since it’s very similar to the above (walk through it
yourself for practice).

$$
\begin{align}
    \P\left(\forall k \le K, t < T. |\hat \mu^k_t - \mu^k | \le B^k_t \right) &\ge 1-\delta'' \\
    \text{where} \quad B^k_t &:= \sqrt{\frac{\ln(2TK/\delta'')}{2N^k_t}}.
\end{align}
$$

Intuitively, $B^k_t$ denotes the *width* of the CI for arm $k$ at time
$t$. Then, assuming the above uniform bound holds (which occurs with
probability $1-\delta''$), we can bound the regret at each timestep as
follows:

$$
\begin{align}
    \mu^\star - \mu^{a_t} &\le \hat \mu^{k^*}_t + B_t^{k^*} - \mu^{a_t} && \text{applying UCB to arm } k^\star \\
    &\le \hat \mu^{a_t}_t + B^{a_t}_t - \mu^{a_t} && \text{since UCB chooses } a_t = \arg \max_{k \in [K]} \hat \mu^k_t + B_t^{k} \\
    &\le 2 B^{a_t}_t && \text{since } \hat \mu^{a_t}_t - \mu^{a_t} \le B^{a_t}_t \text{ by definition of } B^{a_t}_t \\
\end{align}
$$

Summing this across timesteps gives

$$
\begin{align}
    \text{Regret}_T &\le \sum_{t=0}^{T-1} 2 B^{a_t}_t \\
    &= \sqrt{2\ln(2TK/\delta'')} \sum_{t=0}^{T-1} (N^{a_t}_t)^{-1/2} \\
    \sum_{t=0}^{T-1} (N^{a_t}_t)^{-1/2} &= \sum_{t=0}^{T-1} \sum_{k=1}^K \mathbf{1}\{ a_t = k \} (N^k_t)^{-1/2} \\
    &= \sum_{k=1}^K \sum_{n=1}^{N_T^k} n^{-1/2} \\
    &\le K \sum_{n=1}^T n^{-1/2} \\
    \sum_{n=1}^T n^{-1/2} &\le 1 + \int_1^T x^{-1/2} \ \mathrm{d}x \\
    &= 1 + (2 \sqrt{x})_1^T \\
    &= 2 \sqrt{T} - 1 \\
    &\le 2 \sqrt{T} \\
\end{align}
$$

Putting everything together gives

$$
\begin{align}
    \text{Regret}_T &\le 2 K \sqrt{2T \ln(2TK/\delta'')} && \text{with probability } 1-\delta'' \\
    &= \tilde O(K\sqrt{T})
\end{align}
$$

In fact, we can do a more sophisticated analysis to trim off a factor of
$\sqrt{K}$ and show $\text{Regret}_T = \tilde O(\sqrt{TK})$.

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

:::{prf:definition} Thompson sampling
:label: thompson_sampling

**Input:** the prior distribution $\pi \in \Delta([0, 1]^K)$
$\boldsymbol{\mu}\sim \pi(\cdot \mid a_0, r_0, \dots, a_{t-1}, r_{t-1})$
$a_t \gets \arg \max_{k \in [K]} \mu^k$ $r_t \sim \nu^{a_t}$

:::

In other words, we sample each arm proportionally to how likely we think
it is to be optimal, given the observations so far. This strikes a good
exploration-exploitation tradeoff: we explore more for arms that we’re
less certain about, and exploit more for arms that we’re more certain
about. Thompson sampling is a simple yet powerful algorithm that
achieves state-of-the-art performance in many settings.

:::{prf:example} Bayesian Bernoulli bandit
:label: bayesian_bernoulli

We’ve often been working in the Bernoulli bandit setting, where arm $k$
yields a reward of $1$ with probability $\mu^k$ and no reward otherwise.
The vector of success probabilities
$\boldsymbol{\mu}= (\mu^1, \dots, \mu^K)$ thus describes the entire MAB.

Under the Bayesian perspective, we think of $\boldsymbol{\mu}$ as a
*random* vector drawn from some prior distribution
$\pi(\boldsymbol{\mu})$. For example, we might have $\pi$ be the Uniform
distribution over the unit hypercube $[0, 1]^K$, that is,

$$\pi(\boldsymbol{\mu}) = \begin{cases}
    1 & \text{if } \boldsymbol{\mu}\in [0, 1]^K \\
    0 & \text{otherwise}
\end{cases}$$

Then, upon viewing some reward, we can exactly
calculate the **posterior** distribution of $\boldsymbol{\mu}$ using
Bayes’s rule (i.e. the definition of conditional probability):

$$
    \begin{align*}
        \P(\muv \mid a_0, r_0) &\propto \P(r_0 \mid a_0, \muv) \P(a_0 \mid \muv) \P(\muv) \\
        &\propto (\mu^{a_0})^{r_0} (1 - \mu^{a_0})^{1-r_0}.
    \end{align*}
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

```{bibliography}
```
