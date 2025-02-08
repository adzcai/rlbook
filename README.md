# rlbook

This is the course textbook for the Harvard undergraduate course **CS 1840: Introduction to Reinforcement Learning** (also offered as STAT 184).

This project is rendered using [Quarto](https://quarto.org).
Run `quarto render` to build the project and `quarto publish` to deploy it to GitHub pages.
Run `quarto preview` to preview the project in a web browser.
You may need to first set the environment variable

```bash
QUARTO_CHROMIUM_HEADLESS_MODE=new
```

(See https://github.com/quarto-dev/quarto-cli/issues/10532)

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

This textbook takes a slightly different pass.

1. We first define what it means for a policy to be optimal:
   It achieves higher expected remaining reward than all other policies.
   This definition works for state-dependent as well as history-dependent policies.
2. We give an example of an optimal policy in a simple MDP.
3. We claim that every MDP has a state-dependent, deterministic optimal policy.
4. Then it makes sense to define the optimal value function.

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


I also thought the motivation behind infinite-horizon setting could have been made more explicit,
instead of treating it as simply another set of notation to work with.

### Learning

In the supervised learning chapter,
I provide a bit more detailed of an example of a supervised learning task
(image classification of handwritten digits).

I add an example illustrating parameterized function classes.
Students without prior experience in machine learning might not be familiar with the concept of parameters.
They are often introduced in terms of "knobs that you can turn to improve the model",
but this doesn't really accurately describe them at all.

