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
numbering:
  enumerator: 8.%s
---


# 8 Planning


## Introduction

Have you ever lost a strategy game against a skilled opponent?
It probably seemed like they were ahead of you at every turn.
They might have been _planning ahead_ and anticipating your actions,
then planning around them in order to win.
If this opponent was a computer,
they might have been using one of the strategies that we are about to explore.

## Deterministic, zero sum, fully observable two-player games

In this chapter, we will focus on games that are:

- _deterministic,_
- _zero sum_ (one player wins and the other loses),
- _fully observable,_ that is, the state of the game is perfectly known by both players,
- for _two players_ that alternate turns,

We can represent such a game as a _complete game tree._
Each possible state is a node in the tree,
and since we only consider deterministic games,
we can represent actions as edges leading from the current state to the next.
Each path through the tree, from root to leaf, represents a single game.

:::{figure} shared/tic_tac_toe.png
:align: center

The first two layers of the complete game tree of tic-tac-toe.
From Wikimedia.
:::

If you could store the complete game tree on a computer,
you would be able to win every potentially winnable game
by searching all paths from your current state and taking a winning move.
We will see an explicit algorithm for this in [the next section](#min-max-search).
However, as games become more complex,
it becomes computationally impossible to search every possible path.

For instance,
a chess player has roughly 30 actions to choose from at each turn,
and each game takes roughly 40 moves per player,
so trying to solve chess exactly using minimax
would take somewhere on the order of $30^{80} \approx 10^{118}$ operations.
That's 10 billion billion billion billion billion billion billion billion billion billion billion billion billion operations.
As of the time of writing,
the fastest processor can achieve almost 10 GHz (10 billion operations per second),
so to fully solve chess using minimax is many, many orders of magnitude out of reach.

It is thus intractable, in any realistic setting, to solve the complete game tree exactly.
Luckily, only a small fraction of those games ever occur in reality;
Later in this chapter,
we will explore ways to _prune away_ parts of the tree that we know we can safely ignore.
We can also _approximate_ the value of a state without fully evaluating it.
Using these approximations, we can no longer _guarantee_ winning the game,
but we can come up with strategies that will do well against most opponents.

### Notation

Let us now describe these games formally.
We'll call the first player Max and the second player Min.
Max seeks to maximize the final game score,
while Min seeks to minimize the final game score.

- We'll use $\mathcal{S}$ to denote the set of all possible game states.
- The game begins in some **initial state** $s_0 \in \mathcal{S}$.
- Max moves on even turn numbers $h = 2n$,
  and Min moves on odd turn numbers $h = 2n+1$,
  where $n$ is a natural number.
- The space of possible actions, $\mathcal{A}_h(s)$,
  depends on the state itself, as well as whose turn it is.
  (For example, in tic-tac-toe, Max can only play `X`s while Min can only play `O`s.)
- The game ends after $H$ total moves (which might be even or odd). We call the final state a **terminal state**.
- $P$ denotes the **state transitions**, that is,
  $P(s, a)$ denotes the resulting state when taking action $a \in \mathcal{A}(s)$ in state $s$.
- $r(s)$ denotes the **game score** of the terminal state $s$.
  Note that this is some positive or negative value seen by both players:
  A positive value indicates Max winning, a negative value indicates Min winning, and a value of $0$ indicates a tie.

We also call the sequence of states and actions a **trajectory**.

:::{attention}
Above, we suppose that the game ends after $H$ total moves.
But most real games have a _variable_ length.
How would you describe this?
:::

Let us frame tic-tac-toe in this setting.

- Each of the $9$ squares is either empty, marked X, or marked O.
  So there are $|\mathcal{S}| = 3^9$ potential states.
  Not all of these may be reachable!
- The initial state $s_0$ is the empty board.
- The set of possible actions for Max in state $s$, $\mathcal{A}_{2n}(s)$, is the set of tuples $(\text{``X''}, i)$ where $i$ refers to an empty square in $s$.
  Similarly, $\mathcal{A}_{2n+1}(s)$ is the set of tuples $(\text{``O''}, i)$ where $i$ refers to an empty square in $s$.
- We can take $H = 9$ as the longest possible game length.
- $P(s, a)$ for a *nonterminal* state $s$ is simply the board with the symbol and square specified by $a$ marked into $s$. Otherwise, if $s$ is a *terminal* state, i.e. it already has three symbols in a row, the state no longer changes.
- $r(s)$ at a *terminal* state is $+1$ if there are three Xs in a row, $-1$ if there are three Os in a row, and $0$ otherwise.

Our notation may remind you of [Markov decision processes](./mdps.md).
Given that these games also involve a sequence of states and actions,
can we formulate them as finite-horizon MDPs?
The two settings are not exactly analogous,
since in MDPs we only consider a _single_ policy,
while these games involve two distinct players with opposite objectives.
Since we want to analyze the behavior of _both_ players at the same time,
describing such a game as an MDP is more trouble than it's worth.

(min-max-search)=
## Min-max search *

:::{important}
The course (Fall 2024) does not cover min-max search.
This content is here to provide background on _optimally_ solving these deterministic, zero-sum, two-player games.
:::

In the introduction,
we claimed that we could win any potentially winnable game by looking ahead and predicting the opponent's actions.
This would mean that each _nonterminal_ state already has some predetermined game score,
that is, in each state,
it is already "obvious" which player is going to win.
Let $V_\hi^\star(s)$ denote the game score under optimal play starting in state $s$ at time $\hi$.
We can compute this by starting at the terminal states,
when the game's outcome is known,
and working backwards,
assuming that Max chooses the action that leads to the highest score
and Min chooses the action that leads to the lowest score.

:::{prf:algorithm} Min-max search algorithm
:label: min-max-value

$$
V_\hi^{\star}(s) = \begin{cases}
r(s) & \hi = \hor \\
\max_{a \in \mathcal{A}(s)} V_{\hi+1}^{\star}(P(s, a)) & h \text{ is even and } h < H \\
\min_{a \in \mathcal{A}(s)} V_{\hi+1}^{\star}(P(s, a)) & h \text{ is odd and } h < H \\
\end{cases}
$$
:::

This translates directly into a recursive depth-first search algorithm for searching the game tree.

```python
def minimax_search(s, player) -> Tuple["Action", "Value"]:
    """Return the value of the state (for Max) and the best action for Max to take."""
    if env.is_terminal(s):
        return None, env.winner(s)

    if player is max:
        a_max, v_max = None, None
        for a in actions:
            _, v = minimax_search(env.step(s, a), min)
            if v > v_max:
                a_max, v_max = a, v
        return a_max, v_max
    else:
        a_min, v_min = None, None
        for a in actions:
            _, v = minimax_search(env.step(s, a), max)
            if v < v_min:
                a_min, v_min = a, v
        return a_min, v_min
```

At each of the $\hor$ timesteps,
this algorithm iterates through the entire action space at that state,
and therefore has a time complexity of $\hor^{n_A}$
(where $n_A$ is the largest number of actions possibly available at once).
This makes the min-max algorithm impractical for even moderately sized games.

But do we need to compute the exact value of _every_ possible state?
Instead, is there some way we could "ignore" certain actions and their subtrees
if we already know of better options?
The **alpha-beta search** makes use of this intuition.

## Alpha-beta search

The intuition behind alpha-beta search is as follows:
Suppose Max is in state $s$,
and considering whether to take action $a$ or $a'$.
If at any point they finds out that action $a'$ is definitely worse than, or equal to, action $a$,
they don't need to evaluate action $a'$ any further.
Let us illustrate alpha-beta search with an example.

Concretely, we run min-max search as above, 
except now we keep track of two additional parameters $\alpha(s)$ and $\beta(s)$ while evaluating each state.
$\alpha(s)$ represents the _highest_ known game score Max can achieve from state $s$,
and $\beta(s)$ represents the _lowest_ known game score Min can achieve from state $s$.
So if Max is in state $s$, and evaluating a move that leads to state $s'$,
and they find that state $s'$ has some value *greater* than $\beta(s)$,
they can stop evaluating,
since they know Min would not choose an action that enters state $s$.

:::{prf:example} Alpha-beta search for a simple game
:label: alpha-beta-example

Consider a simple game that consists of just one move by Max and one move by Min. Each player has three available actions. Each pair of moves leads to a different integer outcome.
Max tries to find the optimal action using a depth-first search.
They imagine taking the first action,
and then imagine each of the actions that Min could take.
They know that Min will choose whichever option minimizes Max's score.
Thus the value of taking the first action is updated exactly:

![](./shared/alpha-beta-0.png)
![](./shared/alpha-beta-1.png)
![](./shared/alpha-beta-2.png)

Then Max imagines taking the second action.
Once again, they imagine each of the actions that Min could take,
in order.
They find that the first of Min's actions in this state leads to a _worse_ outcome (for Max):

![](./shared/alpha-beta-3.png)

Now Max doesn't need to explore Min's other actions;
they know that taking the second action will lead to an outcome at least as bad as the first outcome above,
so they would always prefer taking action one instead of action two.
So Max moves on to considering the third action:

![](./shared/alpha-beta-4.png)

There is still a chance that this action might outperform action one,
so they continue expanding:

![](./shared/alpha-beta-5.png)

Now they know taking action three leads to an outcome worse than action one,
so they do not need to consider any further states.
:::


```python
def alpha_beta_search(s, player, alpha, beta) -> Tuple["Action", "Value"]:
    """Return the value of the state (for Max) and the best action for Max to take."""
    if env.is_terminal(s):
        return None, env.winner(s)

    if player is max:
        a_max, v_max = None, None
        for a in actions:
            _, v = minimax_search(env.step(s, a), min, alpha, beta)
            if v > v_max:
                a_max, v_max = a, v
                alpha = max(alpha, v)
            if v_max >= beta:
                # we know Min will not choose the action that leads to this state
                return a_max, v_max
        return a_max, v_max

    else:
        a_min, v_min = None, None
        for a in actions:
            _, v = minimax_search(env.step(s, a), max)
            if v < v_min:
                a_min, v_min = a, v
                beta = min(beta, v)
            if v_min <= alpha:
                # we know Max will not choose the action that leads to this state
                return a_min, v_min
        return a_min, v_min
```

How do we choose what _order_ to explore the branches?
As you can tell, this significantly affects the efficiency of the pruning algorithm.
If Max explores the possible actions in order from worst to best,
they will not be able to prune any branches at all!
Additionally, to verify that an action is suboptimal,
we must run the search recursively from that action,
which ultimately requires traversing the tree all the way to a leaf node.
The longer the game might possibly last,
the more computation we have to run.

In practice, we can often use background information about the game to develop a **heuristic** for evaluating possible actions.
If a technique is based on background information or intuition,
especially if it isn't rigorously justified,
we call it a heuristic.

Can we develop _heuristic methods_ for tree exploration that works for all sorts of games?
<!-- Here's where we can incorporate the _reinforcement learning_ -->

## Monte Carlo Tree Search

The task of evaluating actions in a complex environment might seem familiar.
We've encountered this problem before in both the [multi-armed bandits](./bandits.md) setting and the [Markov decision process](./mdps.md) setting.
Now we'll see how to combine concepts from these to form a more general and efficient tree search heuristic called **Monte Carlo Tree Search** (MCTS).

When a problem is intractable to solve _exactly_,
we often turn to _approximate_ or _randomized_ algorithms that sacrifice some accuracy in exchange for computational efficiency.
MCTS also improves on alpha-beta search in this sense.
As the name suggests,
MCTS uses _Monte Carlo_ simulation, that is, collecting random samples and computing the sample statistics,
in order to _approximate_ the value of each action.

As before, we imagine a complete game tree in which each path represents an _entire game_.
The goal of MCTS is to assign values to only the game states that are _relevant_ to the _current game_;
We gradually expand the tree at each move.
For comparison, in alpha-beta search,
the entire tree only needs to be solved _once_,
and from then on,
choosing an action is as simple as taking a maximum over the previously computed values.

The crux of MCTS is approximating the win probability of a state by a _sample probability_.
In practice, MCTS is used for games with _binary outcomes_ where $r(s) \in \{ +1, -1 \}$,
and so this is equivalent to approximating the final game score.
To approximate the win probability from state $s$,
MCTS samples random games starting in $s$ and computes the sample proportion of those that the player wins.

Note that, for a given state $s$,
choosing the best action $a$ can be framed as a [multi-armed bandits](./bandits.md) problem,
where each action corresponds to an arm,
and the reward distribution of arm $k$ is the distribution of the game score over random games after choosing that arm.
The most commonly used bandit algorithm in practice for MCTS is the [{name}](#ucb) algorithm.

:::{note} Summary of UCB
Let us quickly review the UCB bandit algorithm.
For each arm $k$, we track the sample mean
$$\hat \mu^k_t = \frac{1}{N_t^k} \sum_{\tau=0}^{t-1} \ind{a_\tau = k} r_\tau$$
of all rewards from that arm up to time $t$.
Then we construct a _confidence interval_
$$C_t^k = [\hat \mu^k_t - B_t^k, \hat \mu^k_t + B_t^k],$$
where $B_t^k = \sqrt{\frac{\ln(2 t / \delta)}{2 N_t^k}}$ is given by Hoeffding's inequality,
so that with probability $\delta$ (some fixed parameter we choose),
the true mean $\mu^k$ lies within $C_t^k$.
Note that $B_t^k$ scales like $\sqrt{1/N^k_t}$,
i.e. the more we have visited that arm,
the more confident we get about it,
and the narrower the confidence interval.

To select an arm, we pick the arm with the highest _upper confidence bound_.
:::

This means that, for each edge (corresponding to a state-action pair $(s, a)$) in the game tree,
we keep track of the statistics required to compute its UCB:

- How many times it has been "visited" ($N_t^{s, a}$)
- How many of those visits resulted in victory ($\sum_{\tau=0}^{t-1} \ind{(s_\tau, a_\tau) = (s, a)} r_\tau$).
  Let us call this latter value $W^{s, a}_t$ (for number of "wins").

What does $t$ refer to in the above expressions?
Recall $t$ refers to the number of time steps elapsed in the _bandit environment_.
As mentioned above,
each state $s$ corresponds to its own bandit environment,
and so $t$ refers to $N^s$, that is,
how many actions have been taken from state $s$.
This term, $N^s$, gets incremented as the algorithm runs;
For simplicity, we won't introduce another index to track how it changes.

:::{prf:algorithm} Monte Carlo tree search algorithm
:label: mcts-algorithm

Inputs:
- $T$, the number of iterations per move
- $\pi_{\text{rollout}}$, the **rollout policy** for randomly sampling games
- $c$, a positive value that encourages exploration

To choose a single move starting at state $s_{\text{start}}$,
MCTS first tries to estimate the UCB values for each of the possible actions $\mathcal{A}(s_\text{start})$,
and then chooses the best one.
To estimate the UCB values,
it repeats the following four steps $T$ times:

1. **Selection**: We start at $s = s_{\text{start}}$. Let $\tau$ be an empty list that we will use to track states and actions.
   - Until $s$ has at least one action that hasn't been taken:
     - Choose $a \gets \argmax_k \text{UCB}^{s, k}$, where
       $$
       \text{UCB}^{s, a} = \frac{W^{s, a}}{N^s} + c \sqrt{\frac{\ln N^s}{N^{s, a}}}
       \label{ucb-tree}
       $$
     - Append $(s, a)$ to $\tau$
     - Set $s \gets P(s, a)$
2. **Expansion**: Let $s_\text{new}$ denote the final state in $\tau$ (that has at least one action that hasn't been taken). Choose one of these unexplored actions from $s_\text{new}$. Call it $a_{\text{new}}$. Add it to $\tau$.
3. **Simulation**: Simulate a complete game episode starting with the action $a_{\text{new}}$
   and then playing according to $\pi_\text{rollout}$.
   This results in the outcome $r \in \{ +1, -1 \}$.
4. **Backup**: For each $(s, a) \in \tau$:
     - Set $N^{s, a} \gets N^{s, a} + 1$
     - $W^{s, a} \gets W^{s, a} + r$
     - Set $N^s \gets N^s + 1$

After $T$ repeats of the above,
we return the action with the highest UCB value [](#ucb-tree).
Then play continues.

Between turns, we can keep the subtree whose statistics we have visited so far.
However, the rest of the tree for the actions we did _not_ end up taking gets discarded.
:::

The application which brought the MCTS algorithm to fame was DeepMind's **AlphaGo** {cite}`silver_mastering_2016`.
Since then, it has been used in numerous applications ranging from games to automated theorem proving.

How accurate is this Monte Carlo estimation?
It might depend heavily on the rollout policy $\pi_\text{rollout}$.
If the distribution it induces over games is very different from the distribution seen during real gameplay,
we might end up with a poor approximation to the actual value of a state.

### Value approximation

To remedy this,
we might make use of a value function $v : \mathcal{S} \to \mathbb{R}$ that more efficiently approximates the value of a state.
Then, we can replace the simulation step of [MCTS](#mcts-algorithm) with evaluating $r = v(P(s_\text{new}, a_\text{new}))$.

We might also make use of a _policy_ function $\pi : \mathcal{S} \to \triangle(\mathcal{A})$ that provides "intuition" as to which actions are more valuable in a given state.
We can scale the "exploration" term of [](#ucb-tree) according to the policy function's outputs.

Putting these together,
we can describe an updated version of MCTS that makes use of these value and policy functions:

:::{prf:algorithm} Monte Carlo tree search with policy and value functions
:label: mcts-policy-value

Inputs:
- $T$, the number of iterations per move
- $v$, a value function that evaluates how good a state is
- $\pi$, a policy function that encourages certain actions
- $c$, a positive value that encourages exploration

To select a move in state $s_\text{start}$, we repeat the following four steps $T$ times:

1. **Selection**: We start at $s = s_{\text{start}}$. Let $\tau$ be an empty list that we will use to track states and actions.
   - Until $s$ has at least one action that hasn't been taken:
     - Choose $a \gets \argmax_k \text{UCB}^{s, k}$, where
       $$
       \text{UCB}^{s, a} = \frac{W^{s, a}}{N^s} + c \pi(a \mid s) \sqrt{\frac{\ln N^s}{N^{s, a}}}
       \label{ucb-tree-policy}
       $$
     - Append $(s, a)$ to $\tau$
     - Set $s \gets P(s, a)$
2. **Expansion**: Let $s_\text{new}$ denote the final state in $\tau$ (that has at least one action that hasn't been taken). Choose one of these unexplored actions from $s_\text{new}$. Call it $a_{\text{new}}$. Add it to $\tau$.
3. **Simulation**: Evaluate $r = v(P(s_\text{new}, a_\text{new}))$. This approximates the value of the game after taking the action $a_\text{new}$.
4. **Backup**: For each $(s, a) \in \tau$:
     - Set $N^{s, a} \gets N^{s, a} + 1$
     - $W^{s, a} \gets W^{s, a} + r$
     - Set $N^s \gets N^s + 1$

We finally return the action with the highest UCB value [](#ucb-tree-policy).
Then play continues. As before, we can reuse the tree across timesteps.
:::

How do we actually compute a useful $\pi$ and $v$?
If we have some existing dataset of trajectories,
we could use [supervised learning](./imitation_learning.md) (that is, imitation learning)
to generate a policy $\pi$ via behavioral cloning
and learn $v$ by regressing the game outcomes onto states.
Then, plugging these into [the above algorithm](#mcts-policy-value)
results in a stronger policy by using tree search to "think ahead".

But we don't have to stop at just one improvement step;
we could iterate this process via **self-play**.

### Self-play

Recall the [policy iteration](#policy-iteration) algorithm from the [MDPs](./mdps.md) chapter.
Policy iteration alternates between **policy evaluation** (taking $\pi$ and computing $V^\pi$)
and **policy improvement** (setting $\pi$ to be greedy with respect to $V^\pi$).
Above, we saw how MCTS can be thought of as a "policy improvement" operation:
for a given policy $\pi^0$,
we can use it to influence MCTS.
The resulting algorithm is itself a policy $\pi^0_\text{MCTS}$ that maps from states to actions.
Now, we can use [behavioral cloning](./imitation_learning.md)
to obtain a new policy $\pi^1$ that imitates $\pi^0_\text{MCTS}$.
We can now use $\pi^1$ to influence MCTS,
and repeat.

:::{prf:algorithm} MCTS with self-play
:label: mcts-self-play

Input:

- A parameterized policy $\pi : \Theta \to \mathcal{S} \to \triangle(\mathcal{A})$
- A parameterized value function $v : \Theta \to \mathcal{S} \to \mathbb{R}$
- A number of trajectories $M$ to generate
- The initial parameters $\theta^0$

Initialize $\theta \gets \theta^0$.

For $t = 0, \dots, T-1$:

- **Policy improvement**: Use $\pi_{\theta}$ with MCTS to play against itself $M$ times. This generates $M$ trajectories $\tau_0, \dots, \tau_{M-1}$.
- **Policy evaluation**: Use behavioral cloning to mimic the behavior of the policy induced by MCTS. That is,
  $$\theta \gets \argmin_\theta - \sum_{m=0}^{M-1} \sum_{h=0}^{H-1} \log \pi_\theta(a_\hi \mid s_\hi)$$

:::


## References

Chapter 5 of {cite}`russell_artificial_2021` provides an excellent overview of search methods in games.


