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


:::{warning}
This chapter is a WORK IN PROGRESS.
:::

## Introduction

Have you ever lost a strategy game against a skilled opponent?
It probably seemed like they were ahead of you at every turn.
They might have been _planning ahead_ and anticipating your actions,
then planning around them in order to win.
If this opponent was a computer,
they might have been using one of the strategies in this chapter.

This chapter is heavily inspired by Chapter 5 of {cite}`russell_artificial_2021`.
We will focus on games that are:

- for _two players_ that alternate turns,
- _deterministic,_
- _zero sum_ (one player wins and the other loses),
- and _fully observable,_ that is, the state of the game is perfectly known by both players.

We can represent such a game as a _complete game tree._
Each path through the tree, from root to leaf, represents a single game.

:::{figure} shared/tic_tac_toe.png
:align: center

The first two layers of the complete game tree of tic-tac-toe.
From Wikimedia.
:::

If you could store the entire game tree and perfectly predict the outcome of every move,
you would be able to win every game (if possible).
However, as games become more complex,
it becomes computationally impossible to search every possible path.
Later in this chapter,
we will explore ways to _prune away_ parts of the tree that we know are suboptimal,
as well as ways to _approximate_ the value of a state.

We'll follow the convention of naming the two players Max and Min.
Max seeks to maximize the final game score,
while Min seeks to minimize the final game score.
We'll borrow most of the notation from the [](./mdps.md) chapter,
treating such games as finite-horizon MDPs.
The reward $r(s)$ comes only at the end of the game ($\hi = \hor$)
and is $+1$ if Max wins and $-1$ if Min wins.
The state transitions are _deterministic_ i.e. $s_{\hi+1} = P(s_\hi, a_\hi)$.
We assume the state indicates which player's turn it is.


## Min-max search

The optimal value function of a given state at a given timestep,
_playing as Max,_
$V_\hi^{\star}(s)$,
is also called its _minimax value._
This assumes that Min always acts optimally,
that is,
each player assumes the other is acting in their own best interest.

$$
V_\hi^{\star}(s) = \begin{cases}
r(s) & \hi = \hor \\
\max_{a \in \mathcal{A}} V_{\hi+1}^{\star}(P(s, a)) & \text{it is Max's turn} \\
\min_{a \in \mathcal{A}} V_{\hi+1}^{\star}(P(s, a)) & \text{it is Min's turn} \\
\end{cases}
$$

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
this algorithm iterates through the entire action space,
and therefore has a time complexity of $\hor^{|\mathcal{A}|}$.
This makes it impractical for even moderately sized games.
For instance,
a chess player has roughly 35 actions to choose from,
and each game takes roughly 40 moves per player,
so trying to solve chess exactly using minimax
would take somewhere on the order of $35^{80} \approx 10^{123}$ operations.
Suffice it to say that computing this would be an impressive feat.

How can we start to trim down the search tree,
that is,
the subgraph that we actually need to traverse?
We might not need to expand actions if we already know of better options.
The **alpha-beta search** formalizes this intuition.

## Alpha-beta search

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




## Monte Carlo Tree Search

