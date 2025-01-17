# rlbook

This is the course textbook for the Harvard undergraduate course **CS 1840: Introduction to Reinforcement Learning** (also offered as STAT 184).

This project is rendered using [Quarto](https://quarto.org).
Run `quarto render` to build the project and `quarto publish` to deploy it to GitHub pages.

Please leave an issue on GitHub if you have any suggestions or improvements!

## Differences from the course

### MDPs

I have found that students struggle with the concept of an optimal policy and related concepts.
Since there are many equivalent definitions,
it can be confusing which definition to reach for in particular proof;
when asking a student to prove something,
to some, it may seem trivial or unclear what we are trying to prove,
since they saw the desired characterization as the original definition.

- Satisfies Bellman optimality equations
- Dominates all other policies
- Dominates state-dependent, deterministic policies

We spend more time explaining contraction mappings,
which serve as the cornerstone of proofs in the infinite-horizon setting.

We also prove the Bellman consistency equations before using them in the policy evaluation setting.
The new setting does feel more well motivated.

We define the Bellman operator earlier on,
since the notation is shared between the finite and infinite horizon settings.

I also implemented the small tidying MDP example,
since it has an intuitive optimal policy.

For value functions,
since we only define value functions for state-dependent policies,
we make this more explicit and defer the proof to the AJKS book.

We also clarify "Bellman _consistency_ equations" and "Bellman _optimality_ equations".

We also add some intuition for why policy iteration converges more quickly
than value iteration.

