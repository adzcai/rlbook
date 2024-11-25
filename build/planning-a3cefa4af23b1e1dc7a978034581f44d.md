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

# 8 Tree Search Methods


## Introduction

Have you ever lost a strategy game against a skilled opponent?
It probably seemed like they were ahead of you at every turn.
They might have been _planning ahead_ and anticipating your actions,
then planning around them in order to win.
If this opponent was a computer,
they might have been using one of the strategies that we are about to explore.

```{code-cell} ipython3
from utils import Int, Array
```

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

+++

:::{figure} shared/tic_tac_toe.png
:align: center

The first two layers of the complete game tree of tic-tac-toe.
From Wikimedia.
:::

+++

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

+++

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
  $P(s, a)$ denotes the resulting state when taking action $a \in \mathcal{A}(s)$ in state $s$. We'll assume that this function is time-homogeneous (a.k.a. stationary) and doesn't change across timesteps.
- $r(s)$ denotes the **game score** of the terminal state $s$.
  Note that this is some positive or negative value seen by both players:
  A positive value indicates Max winning, a negative value indicates Min winning, and a value of $0$ indicates a tie.

We also call the sequence of states and actions a **trajectory**.

:::{attention}
Above, we suppose that the game ends after $H$ total moves.
But most real games have a _variable_ length.
How would you describe this?
:::

:::{prf:example} Tic-tac-toe
:label: tic-tac-toe

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
:::

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

In the introduction,
we claimed that we could win any potentially winnable game by looking ahead and predicting the opponent's actions.
This would mean that each _nonterminal_ state already has some predetermined game score,
that is, in each state,
it is already "obvious" which player is going to win.

Let $V_\hi^\star(s)$ denote the game score under optimal play from both players starting in state $s$ at time $\hi$.

:::{prf:definition} Min-max search algorithm
:label: min-max-value

$$
V_\hi^{\star}(s) = \begin{cases}
r(s) & \hi = \hor \\
\max_{a \in \mathcal{A}_\hi(s)} V_{\hi+1}^{\star}(P(s, a)) & \hi \text{ is even and } \hi < H \\
\min_{a \in \mathcal{A}_\hi(s)} V_{\hi+1}^{\star}(P(s, a)) & \hi \text{ is odd and } \hi < H \\
\end{cases}
$$
:::

We can compute this by starting at the terminal states,
when the game's outcome is known,
and working backwards,
assuming that Max chooses the action that leads to the highest score
and Min chooses the action that leads to the lowest score.

This translates directly into a recursive depth-first search algorithm for searching the complete game tree.

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

latex(minimax_search)
```

:::{prf:example} Min-max search for a simple game
:label: min-max-example

Consider a simple game with just two steps: Max chooses one of three possible actions (A, B, C),
and then Min chooses one of three possible actions (D, E, F).
The combination leads to a certain integer outcome,
shown in the table below:

|   | D  | E  | F  |
| - | -- | -- | -- |
| A | 4  | -2 | 5  |
| B | -3 | 3  | 1  |
| C | 0  | 3  | -1 |

We can visualize this as the following complete game tree,
where each box contains the value $V_\hi^\star(s)$ of that node.
The min-max values of the terminal states are already known:

![](./shared/minmax.png)

We begin min-max search at the root,
exploring each of Max's actions.
Suppose Max chooses action A.
Then Min will choose action E to minimize the game score,
making the value of this game node $\min(4, -2, 5) = -2$.

![](./shared/minmax-2.png)

Similarly, if Max chooses action B,
then Min will choose action D,
and if Max chooses action C,
then Min will choose action F.
We can fill in the values of these nodes accordingly:

![](./shared/minmax-3.png)

Thus, Max's best move is to take action C,
resulting in a game score of $\max(-2, -3, -1) = -1$.

![](./shared/minmax-4.png)
:::

### Complexity of min-max search

At each of the $\hor$ timesteps,
this algorithm iterates through the entire action space at that state,
and therefore has a time complexity of $\hor^{n_A}$
(where $n_A$ is the largest number of actions possibly available at once).
This makes the min-max algorithm impractical for even moderately sized games.

But do we need to compute the exact value of _every_ possible state?
Instead, is there some way we could "ignore" certain actions and their subtrees
if we already know of better options?
The **alpha-beta search** makes use of this intuition.

(alpha-beta-search)=
## Alpha-beta search

The intuition behind alpha-beta search is as follows:
Suppose Max is in state $s$,
and considering whether to take action $a$ or $a'$.
If at any point they find out that action $a'$ is definitely worse than (or equal to) action $a$,
they don't need to evaluate action $a'$ any further.

Concretely, we run min-max search as above,
except now we keep track of two additional parameters $\alpha(s)$ and $\beta(s)$ while evaluating each state:

- Starting in state $s$, Max can achieve a game score of _at least_ $\alpha(s)$ assuming Min plays optimally. That is, $V^\star_\hi(s) \ge \alpha(s)$ at all points.
- Analogously, starting in state $s$, Min can ensure a game score of _at most_ $\beta(s)$ assuming Max plays optimally. That is, $V^\star_\hi(s) \le \beta(s)$ at all points.

Suppose we are evaluating $V^\star_\hi(s)$,
where it is Max's turn ($\hi$ is even).
We update $\alpha(s)$ to be the _highest_ minimax value achievable from $s$ so far.
That is, the value of $s$ is _at least_ $\alpha(s)$.
Suppose Max chooses action $a$, which leads to state $s'$, in which it is Min's turn.
If any of Min's actions in $s'$ achieve a value $V^\star_{\hi+1}(s') \le \alpha(s)$,
we know that Max would not choose action $a$,
since they know that it is _worse_ than whichever action gave the value $\alpha(s)$.
Similarly, to evaluate a state on Min's turn,
we update $\beta(s)$ to be the _lowest_ value achievable from $s$ so far.
That is, the value of $s$ is _at most_ $\beta(s)$.
Suppose Min chooses action $a$,
which leads to state $s'$ for Max.
If Max has any actions that do _better_ than $\beta(s)$,
they would take it,
making action $a$ a suboptimal choice for Min.

:::{prf:example} Alpha-beta search for a simple game
:label: alpha-beta-example

Let us use the same simple game from [](#min-max-example).
We list the values of $\alpha(s), \beta(s)$ in each node throughout the algorithm.
These values are initialized to $-\infty, +\infty$ respectively.
We shade any squares that have not been visited by the algorithm,
and we assume that actions are evaluated from left to right.

![](./shared/alpha-beta-0.png)

Suppose Max takes action A. Let $s'$ be the resulting game state.
The values of $\alpha(s')$ and $\beta(s')$
are initialized at the same values as the root state,
since we want to prune a subtree if there exists a better action at any step higher in the tree.

![](./shared/alpha-beta-1.png)

Then we iterate through Min's possible actions,
updating the value of $\beta(s')$ as we go.

![](./shared/alpha-beta-2.png)
![](./shared/alpha-beta-3.png)

Once the value of state $s'$ is fully evaluated,
we know that Max can achieve a value of _at least_ $-2$ starting from the root,
and so we update $\alpha(s)$, where $s$ is the root state:

![](./shared/alpha-beta-4.png)

Then Max imagines taking action B. Again, let $s'$ denote the resulting game state.
We initialize $\alpha(s')$ and $\beta(s')$ from the root:

![](./shared/alpha-beta-5.png)

Now suppose Min takes action D, resulting in a value of $-3$.
We see that $V^\star_\hi(s') = \min(-3, x, y)$,
where $x$ and $y$ are the values of the remaining two actions.
But since $\min(-3, x, y) \le -3$,
we know that the value of $s'$ is at most $-3$.
But Max can achieve a better value of $\alpha(s') = -2$ by taking action A,
and so Max will never take action B,
and we can prune the search here.
We will use dotted lines to indicate states that have been ruled out from the search:

![](./shared/alpha-beta-6.png)

Finally, suppose Max takes action C.
For Min's actions D and E,
there is still a chance that action C might outperform action A,
so we continue expanding:

![](./shared/alpha-beta-7.png)
![](./shared/alpha-beta-8.png)

Finally, we see that Min taking action F achieves the minimum value at this state.
This shows that optimal play is for Max to take action C,
and Min to take action F.

![](./shared/alpha-beta-9.png)
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

(monte-carlo-tree-search)=
## Monte Carlo Tree Search

The task of evaluating actions in a complex environment might seem familiar.
We've encountered this problem before in both the [multi-armed bandits](./bandits.md) setting and the [Markov decision process](./mdps.md) setting.
Now we'll see how to combine concepts from these to form a more general and efficient tree search heuristic called **Monte Carlo Tree Search** (MCTS).

When a problem is intractable to solve _exactly_,
we often turn to _approximate_ algorithms that sacrifice some accuracy in exchange for computational efficiency.
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
for simplicity, we won't introduce another index to track how it changes.

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
       \text{UCB}^{s, a} = \frac{W^{s, a}}{N^{s, a}} + c \sqrt{\frac{\ln N^s}{N^{s, a}}}
       \label{ucb-tree}
       $$
     - Append $(s, a)$ to $\tau$
     - Set $s \gets P(s, a)$
2. **Expansion**: Let $s_\text{new}$ denote the final state in $\tau$ (that has at least one action that hasn't been taken). Choose one of these unexplored actions from $s_\text{new}$. Call it $a_{\text{new}}$. Add it to $\tau$.
3. **Simulation**: Simulate a complete game episode by starting with the action $a_{\text{new}}$
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
It depends heavily on the rollout policy $\pi_\text{rollout}$.
If the distribution $\pi_\text{rollout}$ induces over games is very different from the distribution seen during real gameplay,
we might end up with a poor value approximation.

### Incorporating value functions and policies

To remedy this,
we might make use of a value function $v : \mathcal{S} \to \mathbb{R}$ that more efficiently approximates the value of a state.
Then, we can replace the simulation step of [MCTS](#mcts-algorithm) with evaluating $r = v(s_\text{next})$, where $s_\text{next} = P(s_\text{new}, a_\text{new})$.

We might also make use of a **"guiding" policy** $\pi_\text{guide} : \mathcal{S} \to \triangle(\mathcal{A})$ that provides "intuition" as to which actions are more valuable in a given state.
We can scale the exploration term of [](#ucb-tree) according to the policy's outputs.

Putting these together,
we can describe an updated version of MCTS that makes use of these value functions and policy:

:::{prf:algorithm} Monte Carlo tree search with policy and value functions
:label: mcts-policy-value

Inputs:
- $T$, the number of iterations per move
- $v$, a value function that evaluates how good a state is
- $\pi_\text{guide}$, a guiding policy that encourages certain actions
- $c$, a positive value that encourages exploration

To select a move in state $s_\text{start}$, we repeat the following four steps $T$ times:

1. **Selection**: We start at $s = s_{\text{start}}$. Let $\tau$ be an empty list that we will use to track states and actions.
   - Until $s$ has at least one action that hasn't been taken:
     - Choose $a \gets \argmax_k \text{UCB}^{s, k}$, where
       $$
       \text{UCB}^{s, a} = \frac{W^{s, a}}{N^s} + c \cdot \pi_\text{guide}(a \mid s) \sqrt{\frac{\ln N^s}{N^{s, a}}}
       \label{ucb-tree-policy}
       $$
     - Append $(s, a)$ to $\tau$
     - Set $s \gets P(s, a)$
2. **Expansion**: Let $s_\text{new}$ denote the final state in $\tau$ (that has at least one action that hasn't been taken). Choose one of these unexplored actions from $s_\text{new}$. Call it $a_{\text{new}}$. Add it to $\tau$.
3. **Simulation**: Let $s_\text{next} = P(s_\text{new}, a_\text{new})$. Evaluate $r = v(s_\text{next})$. This approximates the value of the game after taking the action $a_\text{new}$.
4. **Backup**: For each $(s, a) \in \tau$:
   - $N^{s, a} \gets N^{s, a} + 1$
   - $W^{s, a} \gets W^{s, a} + r$
   - $N^s \gets N^s + 1$

We finally return the action with the highest UCB value [](#ucb-tree-policy).
Then play continues. As before, we can reuse the tree across timesteps.
:::

How do we actually compute a useful $\pi_\text{guide}$ and $v$?
If we have some existing dataset of trajectories,
we could use [supervised learning](./imitation_learning.md) (that is, imitation learning)
to generate a policy $\pi_\text{guide}$ via behavioral cloning
and learn $v$ by regressing the game outcomes onto states.
Then, plugging these into [the above algorithm](#mcts-policy-value)
results in a stronger policy by using tree search to "think ahead".

But we don't have to stop at just one improvement step;
we could iterate this process via **self-play**.

### Self-play

Recall the [policy iteration](#policy_iteration) algorithm from the [MDPs](./mdps.md) chapter.
Policy iteration alternates between **policy evaluation** (taking $\pi$ and computing $V^\pi$)
and **policy improvement** (setting $\pi$ to be greedy with respect to $V^\pi$).
Above, we saw how MCTS can be thought of as a "policy improvement" operation:
for a given policy $\pi^0$,
we can use it to guide MCTS,
resulting in an algorithm that is itself a policy $\pi^0_\text{MCTS}$ that maps from states to actions.
Now, we can use [behavioral cloning](./imitation_learning.md)
to obtain a new policy $\pi^1$ that imitates $\pi^0_\text{MCTS}$.
We can now use $\pi^1$ to guide MCTS,
and repeat.

:::{prf:algorithm} MCTS with self-play
:label: mcts-self-play

Input:

- A parameterized policy class $\pi_\theta : \mathcal{S} \to \triangle(\mathcal{A})$
- A parameterized value function class $v_\lambda : \mathcal{S} \to \mathbb{R}$
- A number of trajectories $M$ to generate
- The initial parameters $\theta^0, \lambda^0$

For $t = 0, \dots, T-1$:

- **Policy improvement**: Let $\pi^t_\text{MCTS}$ denote the policy obtained by [](#mcts-policy-value) with $\pi_{\theta^t}$ and $v_{\lambda^t}$. We use $\pi^t_\text{MCTS}$ to play against itself $M$ times. This generates $M$ trajectories $\tau_0, \dots, \tau_{M-1}$.
- **Policy evaluation**: Use behavioral cloning to find a set of policy parameters $\theta^{t+1}$ that mimic the behavior of $\pi^t_\text{MCTS}$ and a set of value function parameters $\lambda^{t+1}$ that approximate its value function. That is,
  \begin{align*}
  \theta^{t+1} &\gets \argmin_\theta \sum_{m=0}^{M-1} \sum_{\hi=0}^{H-1} - \log \pi_\theta(a^m_\hi \mid s^m_\hi) \\
  \lambda^{t+1} &\gets \argmin_\lambda \sum_{m=0}^{M-1} \sum_{\hi=0}^{H-1} (v_\lambda(s^m_\hi) - R(\tau_m))^2
  \end{align*}

Note that in implementation,
the policy and value are typically both returned by a single deep neural network,
that is, with a single set of parameters,
and the two loss functions are added together.
:::

This algorithm was brought to fame by AlphaGo Zero {cite}`silver_mastering_2017`.

## Summary

In this chapter,
we explored tree search-based algorithms for deterministic, zero sum, fully observable two-player games.
We began with [min-max search](#min-max-search),
an algorithm for exactly solving the game value of every possible state.
However, this is impossible to execute in practice,
and so we must resort to various ways to reduce the number of states and actions that we must explore.
[Alpha-beta search](#alpha-beta-search) does this by _pruning_ away states that we already know to be suboptimal,
and [Monte Carlo Tree Search](#monte-carlo-tree-search) _approximates_ the value of states instead of evaluating them exactly.


## References

Chapter 5 of {cite}`russell_artificial_2021` provides an excellent overview of search methods in games.
The original AlphaGo paper {cite}`silver_mastering_2016` was a groundbreaking application of these technologies.
{cite}`silver_mastering_2017` removed the imitation learning phase,
learning from scratch.
AlphaZero {cite}`silver_general_2018` then extended to other games beyond Go,
namely shogi and chess,
also learning from scratch.
In MuZero {cite}`schrittwieser_mastering_2020`,
this was further extended by learning a model of the game dynamics.
