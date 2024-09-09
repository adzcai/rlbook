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
math:
  '\st': 'x'
  '\act': 'u'
numbering:
  enumerator: 2.%s
---

# 2 Linear Quadratic Regulators

## Introduction

Up to this point, we have considered decision problems with finitely
many states and actions. However, in many applications, states and
actions may take on continuous values. For example, consider autonomous
driving, controlling a robot's joints, and automated manufacturing. How
can we teach computers to solve these kinds of problems? This is the
task of **continuous control**.

:::{figure} shared/rubiks_cube.jpg
:name: control_examples

Solving a Rubik’s Cube with a robot hand.
:::

:::{figure} shared/boston_dynamics.jpg
:name: robot_hand

Boston Dynamics’s Spot robot.
:::

Aside from the change in the state and action spaces, the general
problem setup remains the same: we seek to construct an *optimal policy*
that outputs actions to solve the desired task. We will see that many
key ideas and algorithms, in particular dynamic programming algorithms,
carry over to this new setting.

This chapter introduces a fundamental tool to solve a simple class of
continuous control problems: the **linear quadratic regulator**. We will
then extend this basic method to more complex settings.

::::{prf:example} CartPole
:label: cart_pole

Try to balance a pencil on its point on a flat surface. It's much more
difficult than it may first seem: the position of the pencil varies
continuously, and the state transitions governing the system, i.e. the
laws of physics, are highly complex. This task is equivalent to the
classic control problem known as *CartPole*:

:::{image} shared/cart_pole.png
:width: 200px
:::

The state $\st \in \mathbb{R}^4$ can be described by:

1.  the position of the cart;

2.  the velocity of the cart;

3.  the angle of the pole;

4.  the angular velocity of the pole.

We can *control* the cart by applying a horizontal force $\act \in \mathbb{R}$.

**Goal:** Stabilize the cart around an ideal state and action
$(\st^\star, \act^\star)$.
::::

## Optimal control

Recall that an MDP is defined by its state space $\mathcal{S}$, action space
$\mathcal{A}$, state transitions $P$, reward function $r$, and discount factor
$\gamma$ or time horizon $\hor$. These have equivalents in the control
setting:

-   The state and action spaces are *continuous* rather than finite.
    That is, $\mathcal{S} \subseteq \mathbb{R}^{n_\st}$ and $\mathcal{A} \subseteq \mathbb{R}^{n_\act}$,
    where $n_\st$ and $n_\act$ are the corresponding dimensions of these
    spaces, i.e. the number of coordinates to specify a single state or
    action respectively.

-   We call the state transitions the **dynamics** of the system. In the
    most general case, these might change across timesteps and also
    include some stochastic **noise** $w_\hi$ at each timestep. We
    denote these dynamics as the function $f_\hi$ such that
    $\st_{\hi+1} = f_\hi(\st_\hi, \act_\hi, w_\hi)$. Of course, we can
    simplify to cases where the dynamics are *deterministic/noise-free*
    (no $w_\hi$ term) and/or *time-homogeneous* (the same function $f$
    across timesteps).

-   Instead of maximizing the reward function, we seek to minimize the
    **cost function** $c_\hi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$. Often, the cost
    function describes *how far away* we are from a **target
    state-action pair** $(\st^\star, \act^\star)$. An important special
    case is when the cost is *time-homogeneous*; that is, it remains the
    same function $c$ at each timestep $h$.

-   We seek to minimize the *undiscounted* cost within a *finite time
    horizon* $\hor$. Note that we end an episode at the final state
    $\st_\hor$ -- there is no $\act_\hor$, and so we denote the cost for
    the final state as $c_\hor(\st_\hor)$.

With all of these components, we can now formulate the **optimal control
problem:** *compute a policy to minimize the expected undiscounted cost
over $\hor$ timesteps.* In this chapter, we will only consider
*deterministic, time-dependent* policies
$\pi = (\pi_0, \dots, \pi_{H-1})$ where $\pi_h : \mathcal{S} \to \mathcal{A}$ for each
$\hi \in [\hor]$.

:::{prf:definition} General optimal control problem
:label: optimal_control



$$
\begin{split}
            \min_{\pi_0, \dots, \pi_{\hor-1} : \mathcal{S} \to \mathcal{A}} \quad & \E \left[
                \left( \sum_{\hi=0}^{\hor-1} c_\hi(\st_\hi, \act_\hi) \right) + c_\hor(\st_\hor)
                \right] \\
            \text{where} \quad & \st_{\hi+1} = f_\hi(\st_\hi, \act_\hi, w_\hi), \\
            & \act_\hi = \pi_\hi(\st_\hi) \\
            & \st_0 \sim \mu_0 \\
            & w_\hi \sim \text{noise}
        \end{split}
$$


:::

### A first attempt: Discretization

Can we solve this problem using tools from the finite MDP setting? If
$\mathcal{S}$ and $\mathcal{A}$ were finite, then we'd be able to work backwards using the DP algorithm for computing the optimal policy in an MDP ([](#pi_star_dp)).
This inspires us to try *discretizing* the
problem.

Suppose $\mathcal{S}$ and $\mathcal{A}$ are bounded, that is,
$\max_{\st \in \mathcal{S}} \|\st\| \le B_\st$ and
$\max_{\act \in \mathcal{A}} \|\act\| \le B_\act$. To make $\mathcal{S}$ and $\mathcal{A}$ finite,
let's choose some small positive $\epsilon$, and simply round each
coordinate to the nearest multiple of $\epsilon$. For example, if
$\epsilon = 0.01$, then we round each element of $\st$ and $\act$ to two
decimal spaces.

However, the discretized $\widetilde{\mathcal{S}}$ and $\widetilde{\mathcal{A}}$ may be finite, but
they may be infeasibly large: we must divide *each dimension* into
intervals of length $\varepsilon$, resulting in
$|\widetilde{\mathcal{S}}| = (B_\st/\varepsilon)^{n_\st}$ and
$|\widetilde{\mathcal{A}}| = (B_\act/\varepsilon)^{n_\act}$. To get a sense of how
quickly this grows, consider $\varepsilon = 0.01, n_\st = n_\act = 10$.
Then the number of elements in the transition matrix would be
$|\widetilde{\mathcal{S}}|^2 |\widetilde{\mathcal{A}}| = (100^{10})^2 (100^{10}) = 10^{60}$! (That's
a trillion trillion trillion trillion trillion.)

What properties of the problem could we instead make use of? Note that
by discretizing the state and action spaces, we implicitly assumed that
rounding each state or action vector by some tiny amount $\varepsilon$
wouldn't change the behavior of the system by much; namely, that the
cost and dynamics were relatively *continuous*. Can we use this
continuous structure in other ways? This leads us to the **linear
quadratic regulator**.

(lqr)=
## The Linear Quadratic Regulator

The optimal control problem
{prf:ref}`optimal_control` seems highly complex in its general
case. Is there a relevant simplification that we can analyze?

Let us consider *linear dynamics* and an *upward-curved quadratic cost
function* (in both arguments). We will also consider a time-homogenous
cost function that targets $(s^\star, a^\star) = (0, 0)$. This model is
called the **linear quadratic regulator** (LQR) and is a fundamental
tool in control theory. Solving the LQR problem will additionally enable
us to *locally approximate* more complex setups using *Taylor
approximations*.

:::{prf:definition} The linear quadratic regulator
:label: lqr_definition

**Linear, time-homogeneous dynamics**: for each timestep $h \in [H]$,


$$
\begin{aligned}
        \st_{\hi+1} &= f(\st_\hi, \act_\hi, w_\hi) = A \st_\hi + B \act_\hi + w_\hi \\
        \text{where } w_\hi &\sim \mathcal{N}(0, \sigma^2 I).
\end{aligned}
$$

 Here, $w_\hi$ is a spherical Gaussian **noise term**
that makes the state transitions random. Setting $\sigma = 0$ gives us
**deterministic** state transitions. We will find that the optimal
policy actually *does not depend on the noise*, although the optimal
value function and Q-function do.

**Upward-curved quadratic, time-homogeneous cost function**:


$$
c(\st_\hi, \act_\hi) = \begin{cases}
            \st_\hi^\top Q \st_\hi + \act_\hi^\top R \act_\hi & \hi < \hor \\
            \st_\hi^\top Q \st_\hi                            & \hi = \hor
        \end{cases}.
$$
        
         We require $Q$ and $R$ to both be positive
definite matrices so that $c$ has a well-defined unique minimum. We can
furthermore assume without loss of generality that they are both
symmetric (see exercise below).

This results in the LQR optimization problem: 

$$
\begin{aligned}
        \min_{\pi_0, \dots, \pi_{\hor-1} : \mathcal{S} \to \mathcal{A}} \quad & \E \left[ \left( \sum_{\hi=0}^{\hor-1} \st_\hi^\top Q \st_\hi + \act_\hi^\top R \act_\hi \right) + \st_\hor^\top Q \st_\hor \right] \\
        \textrm{where} \quad                                & \st_{\hi+1} = A \st_\hi + B \act_\hi + w_\hi                                                                                        \\
                                                            & \act_\hi = \pi_\hi (\st_\hi)                                                                                                        \\
                                                            & w_\hi \sim \mathcal{N}(0, \sigma^2 I)                                                                                               \\
                                                            & \st_0 \sim \mu_0.
\end{aligned}
$$


:::

::: exercise
We've set $Q$ and $R$ to be *symmetric* positive definite (SPD)
matrices. Here we'll show that the symmetry condition can be imposed
without loss of generality. Show that replacing $Q$ with
$(Q + Q^\top) / 2$ (which is symmetric) yields the same cost function.
:::

It will be helpful to reintroduce the *value function* notation for a
policy to denote the average cost it incurs. These will be instrumental
in constructing the optimal policy via **dynamic programming**.

:::{prf:definition} Value functions for LQR
:label: value_lqr

Given a policy $\mathbf{\pi} = (\pi_0, \dots, \pi_{\hor-1})$, we can
define its value function $V^\pi_\hi : \mathcal{S} \to \mathbb{R}$ at time
$\hi \in [\hor]$ as the average **cost-to-go** incurred by that policy:


$$
\begin{split}
            V^\pi_\hi (\st) &= \E \left[ \left( \sum_{i=\hi}^{\hor-1} c(\st_i, \act_i) \right) + c(\st_\hor) \mid \st_\hi = \st,  \act_i = \pi_i(\st_i) \quad \forall \hi \le i < H \right] \\
            &= \E \left[ \left( \sum_{i=\hi}^{\hor-1} \st_i^\top Q \st_i + \act_i^\top R \act_i \right) + \st_\hor^\top Q \st_\hor \mid \st_\hi = \st, \act_i = \pi_i(\st_i) \quad \forall \hi \le i < H \right] \\
        \end{split}
$$



The Q-function additionally conditions on the first action we take:


$$
\begin{split}
            Q^\pi_\hi (\st, \act) &= \E \bigg[ \left( \sum_{i=\hi}^{\hor-1} c(\st_i, \act_i) \right) + c(\st_\hor) \\
                &\qquad\qquad \mid  (\st_\hi, \act_\hi) = (\st, \act), \act_i = \pi_i(\st_i) \quad \forall \hi \le i < H \bigg] \\
            &= \E \bigg[ \left( \sum_{i=\hi}^{\hor-1} \st_i^\top Q \st_i + \act_i^\top R \act_i \right) + \st_\hor^\top Q \st_\hor \\
                &\qquad\qquad \mid (\st_\hi, \act_\hi) = (\st, \act), \act_i = \pi_i(\st_i) \quad \forall \hi \le i < H \bigg] \\
        \end{split}
$$


:::

(optimal_lqr)=
## Optimality and the Riccati Equation

In this section, we'll compute the optimal value function $V^\star_h$,
Q-function $Q^\star_h$, and policy $\pi^\star_h$ in the LQR setting
{prf:ref}`lqr_definition` using
**dynamic programming** in a very similar way to the DP algorithms [in the MDP setting](#eval_dp):

1.  We'll compute $V_H^\star$ (at the end of the horizon) as our base
    case.

2.  Then we'll work backwards in time, using $V_{h+1}^\star$ to compute
    $Q_h^\star$, $\pi_{h}^\star$, and $V_h^\star$.

Along the way, we will prove the striking fact that the solution has
very simple structure: $V_h^\star$ and $Q^\star_h$ are *upward-curved
quadratics* and $\pi_h^\star$ is *linear* and furthermore does not
depend on the noise!

:::{prf:definition} Optimal value functions for LQR
:label: optimal_value_lqr

The **optimal value
function** is the one that, at any time and in any state, achieves
*minimum cost* across *all policies*: 

$$
\begin{split}
    V^\star_\hi(\st) &= \min_{\pi_\hi, \dots, \pi_{\hor-1}} V^\pi_\hi(\st) \\
    &= \min_{\pi_{\hi}, \dots, \pi_{\hor-1}} \E \bigg[ \left( \sum_{i=\hi}^{\hor-1} \st_\hi^\top Q \st_\hi + \act_\hi^\top R \act_\hi \right) + \st_\hor^\top Q \st_\hor \\
        &\hspace{8em} \mid \st_\hi = \st, \act_i = \pi_i(\st_i) \quad \forall \hi \le i < H \bigg] \\
\end{split}
$$


:::

:::{prf:theorem} Optimal value function in LQR is a upward-curved quadratic
:label: optimal_value_lqr_quadratic

At each timestep $h \in [H]$,


$$
V^\star_\hi(\st) = \st^\top P_\hi \st + p_\hi
$$

 for some symmetric
positive definite matrix $P_\hi \in \mathbb{R}^{n_\st \times n_\st}$ and vector
$p_\hi \in \mathbb{R}^{n_\st}$.
:::

:::{prf:theorem} Optimal policy in LQR is linear
:label: optimal_policy_lqr_linear

At each timestep $h \in [H]$, 

$$
\pi^\star_\hi (\st) = - K_\hi \st
$$

 for
some $K_\hi \in \mathbb{R}^{n_\act \times n_\st}$. (The negative is due to
convention.)
:::

**Base case:** At the final timestep, there are no possible actions to
take, and so $V^\star_\hor(\st) = c(\st) = \st^\top Q \st$. Thus
$V_\hor^\star(\st) = \st^\top P_\hi \st + p_\hi$ where $P_\hor = Q$ and
$p_\hor$ is the zero vector.

**Inductive hypothesis:** We seek to show that the inductive step holds
for both theorems: If $V^\star_{\hi+1}(\st)$ is a upward-curved
quadratic, then $V^\star_\hi(\st)$ must also be a upward-curved
quadratic, and $\pi^\star_\hi(\st)$ must be linear. We'll break this
down into the following steps:

::: steps
Show that $Q^\star_\hi(\st, \act)$ is a upward-curved quadratic (in both
$\st$ and $\act$).

Derive the optimal policy
$\pi^\star_\hi(\st) = \arg \min_\act Q^\star_\hi(\st, \act)$ and show
that it's linear.

Show that $V^\star_\hi(\st)$ is a upward-curved quadratic.
:::

We first assume the inductive hypothesis that our theorems are true at
time $\hi+1$. That is,


$$
V^\star_{\hi+1}(\st) = \st^\top P_{\hi+1} \st + p_{\hi+1} \quad \forall \st \in \mathcal{S}.
$$



**Step 1.** We aim to show that $Q^\star_\hi(\st)$ is a upward-curved
quadratic. Recall that the definition of
$Q^\star_\hi : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is


$$
Q^\star_\hi(\st, \act) = c(\st, \act) + \E_{\st' \sim f(\st, \act, w_{\hi+1})} [V^\star_{\hi+1}(\st')].
$$


Recall $c(\st, \act) = \st^\top Q \st + \act^\top R \act$. Let's
consider the average value over the next timestep. The only randomness
in the dynamics comes from the noise
$w_{\hi+1} \sim \mathcal{N}(0, \sigma^2 I)$, so we can write out this expected
value as: 

$$
\begin{aligned}
            & \E_{\st'} [V^\star_{\hi+1}(\st')]                                                                                                         \\
    {} = {} & \E_{w_{\hi+1}} [V^\star_{\hi+1}(A \st + B \act + w_{\hi+1})]                                             &  & \text{definition of } f     \\
    {} = {} & \E_{w_{\hi+1}} [ (A \st + B \act + w_{\hi+1})^\top P_{\hi+1} (A \st + B \act + w_{\hi+1}) + p_{\hi+1} ]. &  & \text{inductive hypothesis}
\end{aligned}
$$

 Summing and combining like terms, we get


$$
\begin{aligned}
    Q^\star_\hi(\st, \act) & = \st^\top Q \st + \act^\top R \act + \E_{w_{\hi+1}} [(A \st + B \act + w_{\hi+1})^\top P_{\hi+1} (A \st + B \act + w_{\hi+1}) + p_{\hi+1}] \\
                           & = \st^\top (Q + A^\top P_{\hi+1} A)\st + \act^\top (R + B^\top P_{\hi+1} B) \act + 2 \st^\top A^\top P_{\hi+1} B \act                       \\
                           & \qquad + \E_{w_{\hi+1}} [w_{\hi+1}^\top P_{\hi+1} w_{\hi+1}] + p_{\hi+1}.
\end{aligned}
$$

 Note that the terms that are linear in $w_\hi$ have mean
zero and vanish. Now consider the remaining expectation over the noise.
By expanding out the product and using linearity of expectation, we can
write this out as 

$$
\begin{aligned}
    \E_{w_{\hi+1}} [w_{\hi+1}^\top P_{\hi+1} w_{\hi+1}] & = \sum_{i=1}^d \sum_{j=1}^d (P_{\hi+1})_{ij} \E_{w_{\hi+1}} [(w_{\hi+1})_i (w_{\hi+1})_j].
\end{aligned}
$$

 When dealing with these *quadratic forms*, it's often
helpful to consider the terms on the diagonal ($i = j$) separately from
those off the diagonal. On the diagonal, the expectation becomes


$$

(P_{\hi+1})_{ii} \E (w_{\hi+1})_i^2 = \sigma^2 (P_{\hi+1})_{ii}.

$$

 Off
the diagonal, since the elements of $w_{\hi+1}$ are independent, the
expectation factors, and since each element has mean zero, the term
disappears:


$$

(P_{\hi+1})_{ij} \E [(w_{\hi+1})_i] \E [(w_{\hi+1})_j] = 0.

$$

 Thus,
the only terms left are the ones on the diagonal, so the sum of these
can be expressed as the trace of $\sigma^2 P_{\hi+1}$:


$$

\E_{w_{\hi+1}} [w_{\hi+1}^\top P_{\hi+1} w_{\hi+1}] = \mathrm{Tr}(\sigma^2 P_{\hi+1}).

$$


Substituting this back into the expression for $Q^\star_\hi$, we have:



:::{math}
:label: q_star_lqr
\begin{aligned}
    Q^\star_\hi(\st, \act) & = \st^\top (Q + A^\top P_{\hi+1} A) \st + \act^\top (R + B^\top P_{\hi+1} B) \act
    + 2\st^\top A^\top P_{\hi+1} B \act                                                                        \\
                            & \qquad + \mathrm{Tr}(\sigma^2 P_{\hi+1}) + p_{\hi+1}.
\end{aligned}
:::



As we hoped, this expression is quadratic in $\st$ and $\act$.
Furthermore, we'd like to show that it also has *positive curvature*
with respect to $\act$ so that its minimum with respect to $\act$ is
well-defined. We can do this by proving that the **Hessian matrix** of
second derivatives is positive definite:


$$

\nabla_{\act \act} Q_\hi^\star(\st, \act) = R + B^\top P_{\hi+1} B

$$

 This
is fairly straightforward: recall that in our definition of LQR, we
assumed that $R$ is SPD (see
{prf:ref}`lqr_definition`).
Also note that since $P_{\hi+1}$ is SPD (by the inductive hypothesis),
so too must be $B^\top P_{\hi+1} B$. (If this isn't clear, try proving
it as an exercise.) Since the sum of two SPD matrices is also SPD, we
have that $R + B^\top P_{\hi+1} B$ is SPD, and so $Q^\star_\hi$ is
indeed a upward-curved quadratic with respect to $\act$.

**Step 2.** Now we aim to show that $\pi^\star_\hi$ is linear. Since
$Q^\star_\hi$ is a upward-curved quadratic, finding its minimum over
$\act$ is easy: we simply set the gradient with respect to $\act$ equal
to zero and solve for $\act$. First, we calculate the gradient:


$$
\begin{aligned}
    \nabla_\act Q^\star_\hi(\st, \act) & = \nabla_\act [ \act^\top (R + B^\top P_{\hi+1} B) \act + 2 \st^\top A^\top P_{\hi+1} B \act ] \\
                                       & = 2 (R + B^\top P_{\hi+1} B) \act + 2 (\st^\top A^\top P_{\hi+1} B)^\top
\end{aligned}
$$

Setting this to zero, we get 

$$
\begin{aligned}
    0                  & = (R + B^\top P_{\hi+1} B) \pi^\star_\hi(\st) + B^\top P_{\hi+1} A \st \nonumber \\
    \pi^\star_\hi(\st) & = (R + B^\top P_{\hi+1} B)^{-1} (-B^\top P_{\hi+1} A \st) \nonumber              \\
                       & = - K_\hi \st,
\end{aligned}
$$

 where
$K_\hi = (R + B^\top P_{\hi+1} B)^{-1} B^\top P_{\hi+1} A$. Note that
this optimal policy doesn't depend on the starting distribution $\mu_0$.
It's also fully **deterministic** and isn't affected by the noise terms
$w_0, \dots, w_{\hor-1}$.

**Step 3.** To complete our inductive proof, we must show that the
inductive hypothesis is true at time $\hi$; that is, we must prove that
$V^\star_\hi(\st)$ is a upward-curved quadratic. Using the identity
$V^\star_\hi(\st) = Q^\star_\hi(\st, \pi^\star(\st))$, we have:


$$
\begin{aligned}
    V^\star_\hi(\st) & = Q^\star_\hi(\st, \pi^\star(\st))                                                                \\
                     & = \st^\top (Q + A^\top P_{\hi+1} A) \st + (-K_\hi \st)^\top (R + B^\top P_{\hi+1} B) (-K_\hi \st)
    + 2\st^\top A^\top P_{\hi+1} B (-K_\hi \st)                                                                          \\
                     & \qquad + \mathrm{Tr}(\sigma^2 P_{\hi+1}) + p_{\hi+1}
\end{aligned}
$$

 Note that with respect to $\st$, this is the sum of a
quadratic term and a constant, which is exactly what we were aiming for!
The constant term is clearly
$p_\hi = \mathrm{Tr}(\sigma^2 P_{\hi+1}) + p_{\hi+1}$. We can simplify the
quadratic term by substituting in $K_\hi$. Notice that when we do this,
the $(R+B^\top P_{\hi+1} B)$ term in the expression is cancelled out by
its inverse, and the remaining terms combine to give the **Riccati
equation**:

:::{prf:definition} Riccati equation
:label: riccati



$$

P_\hi = Q + A^\top P_{\hi+1} A - A^\top P_{\hi+1} B (R + B^\top P_{\hi+1} B)^{-1} B^\top P_{\hi+1} A.

$$


:::

There are several nice properties to note about the Riccati equation:

1.  It's defined **recursively.** Given the dynamics defined by $A$ and
    $B$, and the state cost matrix $Q$, we can recursively calculate
    $P_\hi$ across all timesteps starting from $P_\hor = Q$.

2.  $P_\hi$ often appears in calculations surrounding optimality, such
    as $V^\star_\hi, Q^\star_\hi$, and $\pi^\star_\hi$.

3.  Together with the dynamics given by $A$ and $B$, and the action
    coefficients $R$, it fully defines the optimal policy.

Now we've shown that $V^\star_\hi(\st) = \st^\top P_\hi \st + p_\hi$,
which is a upward-curved quadratic, and this concludes our proof.

In summary, we just demonstrated that at each timestep $\hi \in \hor$,
the optimal value function $V^\star_\hi$ and optimal Q-function
$Q^\star_\hi$ are both upward-curved quadratics and the optimal policy
$\pi^\star_\hi$ is linear. We also showed that all of these quantities
can be calculated using a sequence of symmetric matrices
$P_0, \dots, P_H$ that can be defined recursively using the Riccati
equation {prf:ref}`riccati`.

Before we move on to some extensions of LQR, let's consider how the
state at time $\hi$ behaves when we act according to this optimal
policy.

### Expected state at time $\hi$

How can we compute the expected state at time $\hi$ when acting
according to the optimal policy? Let's first express $\st_\hi$ in a
cleaner way in terms of the history. Note that having linear dynamics
makes it easy to expand terms backwards in time: 

$$
\begin{aligned}
    \st_\hi & = A \st_{\hi-1} + B \act_{\hi-1} + w_{\hi-1}                                 \\
            & = A (A\st_{\hi-2} + B \act_{\hi-2} + w_{\hi-2}) + B \act_{\hi-1} + w_{\hi-1} \\
            & = \cdots                                                                     \\
            & = A^\hi \st_0 + \sum_{i=0}^{\hi-1} A^i (B \act_{\hi-i-1} + w_{\hi-i-1}).
\end{aligned}
$$



Let's consider the *average state* at this time, given all the past
states and actions. Since we assume that $\E [w_\hi] = 0$ (this is the
zero vector in $d$ dimensions), when we take an expectation, the $w_\hi$
term vanishes due to linearity, and so we're left with


$$
\E [\st_\hi \mid \st_{0:(\hi-1)}, \act_{0:(\hi-1)}] = A^\hi \st_0 + \sum_{i=0}^{\hi-1} A^i B \act_{\hi-i-1}.
$$


If we choose actions according to our optimal policy, this becomes


$$
\E [\st_\hi \mid \st_0, \act_i = - K_i \st_i \quad \forall i \le \hi] = \left( \prod_{i=0}^{\hi-1} (A - B K_i) \right) \st_0.
$$


**Exercise:** Verify this.

This introdces the quantity $A - B K_i$, which shows up frequently in
control theory. For example, one important question is: will $\st_\hi$
remain bounded, or will it go to infinity as time goes on? To answer
this, let's imagine for simplicity that these $K_i$s are equal (call
this matrix $K$). Then the expression above becomes $(A-BK)^\hi \st_0$.
Now consider the maximum eigenvalue $\lambda_{\max}$ of $A - BK$. If
$|\lambda_{\max}| > 1$, then there's some nonzero initial state
$\bar \st_0$, the corresponding eigenvector, for which


$$
\lim_{\hi \to \infty} (A - BK)^\hi \bar \st_0
    = \lim_{\hi \to \infty} \lambda_{\max}^\hi \bar \st_0
    = \infty.
$$
    
Otherwise, if $|\lambda_{\max}| < 1$, then it's impossible for your original state to explode as dramatically.

## Extensions

We've now formulated an optimal solution for the time-homogeneous LQR
and computed the expected state under the optimal policy. However, real
world tasks rarely have such simple dynamics, and we may wish to design
more complex cost functions. In this section, we'll consider more
general extensions of LQR where some of the assumptions we made above
are relaxed. Specifically, we'll consider:

1.  **Time-dependency**, where the dynamics and cost function might
    change depending on the timestep.

2.  **General quadratic cost**, where we allow for linear terms and a
    constant term.

3.  **Tracking a goal trajectory** rather than aiming for a single goal
    state-action pair.

Combining these will allow us to use the LQR solution to solve more
complex setups by taking *Taylor approximations* of the dynamics and
cost functions.

(time_dep_lqr)=
### Time-dependent dynamics and cost function

So far, we've considered the *time-homogeneous* case, where the dynamics
and cost function stay the same at every timestep. However, this might
not always be the case. As an example, in many sports, the rules and
scoring system might change during an overtime period. To address these
sorts of problems, we can loosen the time-homogeneous restriction, and
consider the case where the dynamics and cost function are
*time-dependent.* Our analysis remains almost identical; in fact, we can
simply add a time index to the matrices $A$ and $B$ that determine the
dynamics and the matrices $Q$ and $R$ that determine the cost.

::: exercise
Walk through the above derivation to verify this claim.
:::

The modified problem is now defined as follows:

:::{prf:definition} Time-dependent LQR
:label: time_dependent_lqr



$$
\begin{aligned}
        \min_{\pi_{0}, \dots, \pi_{\hor-1}} \quad & \E \left[ \left( \sum_{\hi=0}^{\hor-1} (\st_\hi^\top Q_\hi \st_\hi) + \act_\hi^\top R_\hi \act_\hi \right) + \st_\hor^\top Q_\hor \st_\hor \right] \\
        \textrm{where} \quad                      & \st_{\hi+1} = f_\hi(\st_\hi, \act_\hi, w_\hi) = A_\hi \st_\hi + B_\hi \act_\hi + w_\hi                                                             \\
                                                  & \st_0 \sim \mu_0                                                                                                                                   \\
                                                  & \act_\hi = \pi_\hi (\st_\hi)                                                                                                                       \\
                                                  & w_\hi \sim \mathcal{N}(0, \sigma^2 I).
\end{aligned}
$$


:::

The derivation of the optimal value functions and the optimal policy
remains almost exactly the same, and we can modify the Riccati equation
accordingly:

:::{prf:definition} Time-dependent Riccati Equation
:label: riccati_time_dependent



$$

P_\hi = Q_\hi + A_\hi^\top P_{\hi+1} A_\hi - A_\hi^\top P_{\hi+1} B_\hi (R_\hi + B_\hi^\top P_{\hi+1} B_\hi)^{-1} B_\hi^\top P_{\hi+1} A_\hi.

$$


Note that this is just the time-homogeneous Riccati equation
({prf:ref}`riccati`), but with the time index added to each of the
relevant matrices.
:::

Additionally, by allowing the dynamics to vary across time, we gain the
ability to *locally approximate* nonlinear dynamics at each timestep.
We'll discuss this later in the chapter.

### More general quadratic cost functions

Our original cost function had only second-order terms with respect to
the state and action, incentivizing staying as close as possible to
$(\st^\star, \act^\star) = (0, 0)$. We can also consider more general
quadratic cost functions that also have first-order terms and a constant
term. Combining this with time-dependent dynamics results in the
following expression, where we introduce a new matrix $M_\hi$ for the
cross term, linear coefficients $q_\hi$ and $r_\hi$ for the state and
action respectively, and a constant term $c_\hi$:

:::{math}
:label: general_quadratic_cost

c_\hi(\st_\hi, \act_\hi) = ( \st_\hi^\top Q_\hi \st_\hi + \st_\hi^\top M_\hi \act_\hi + \act_\hi^\top R_\hi \act_\hi ) + (\st_\hi^\top q_\hi + \act_\hi^\top r_\hi) + c_\hi.
:::

Similarly, we can also include a
constant term $v_\hi \in \mathbb{R}^{n_\st}$ in the dynamics (note that this is
*deterministic* at each timestep, unlike the stochastic noise $w_\hi$):


$$

\st_{\hi+1} = f_\hi(\st_\hi, \act_\hi, w_\hi) = A_\hi \st_\hi + B_\hi \act_\hi + v_\hi + w_\hi.

$$



::: exercise
Derive the optimal solution. (You will need to slightly modify the above
proof.)
:::

### Tracking a predefined trajectory

Consider applying LQR to a task like autonomous driving, where the
target state-action pair changes over time. We might want the vehicle to
follow a predefined *trajectory* of states and actions
$(\st_\hi^\star, \act_\hi^\star)_{\hi=0}^{\hor-1}$. To express this as a
control problem, we'll need a corresponding time-dependent cost
function:


$$

c_\hi(\st_\hi, \act_\hi) = (\st_\hi - \st^\star_\hi)^\top Q (\st_\hi - \st^\star_\hi) + (\act_\hi - \act^\star_\hi)^\top R (\act_\hi - \act^\star_\hi).

$$


Note that this punishes states and actions that are far from the
intended trajectory. By expanding out these multiplications, we can see
that this is actually a special case of the more general quadratic cost
function above
({eq}`general_quadratic_cost`):


$$

M_\hi = 0, \qquad q_\hi = -2Q \st^\star_\hi, \qquad r_\hi = -2R \act^\star_\hi, \qquad c_\hi = (\st^\star_\hi)^\top Q (\st^\star_\hi) + (\act^\star_\hi)^\top R (\act^\star_\hi).

$$



(approx_nonlinear)=
## Approximating nonlinear dynamics

The LQR algorithm solves for the optimal policy when the dynamics are
*linear* and the cost function is an *upward-curved quadratic*. However,
real settings are rarely this simple! Let's return to the CartPole
example from the start of the chapter
({prf:ref}`cart_pole`). The dynamics (physics) aren't linear. How
can we approximate this by an LQR problem?

Concretely, let's consider a *noise-free* problem since, as we saw, the
noise doesn't factor into the optimal policy. Let's assume the dynamics
and cost function are stationary, and ignore the terminal state for
simplicity:

:::{prf:definition} Nonlinear control problem
:label: nonlinear_control



$$
\begin{aligned}
        \min_{\pi_0, \dots, \pi_{\hor-1} : \mathcal{S} \to \mathcal{A}} \quad & \E_{\st_0} \left[ \sum_{\hi=0}^{\hor-1} c(\st_\hi, \act_\hi) \right] \\
        \text{where} \quad                                  & \st_{\hi+1} = f(\st_\hi, \act_\hi)                                   \\
                                                            & \act_\hi = \pi_\hi(\st_\hi)                                          \\
                                                            & \st_0 \sim \mu_0                                                     \\
                                                            & c(\st, \act) = d(\st, \st^\star) + d(\act, \act^\star).
\end{aligned}
$$

 Here, $d$ denotes a function that measures the
"distance" between its two arguments.
:::

This is now only slightly simplified from the general optimal control
problem (see
{prf:ref}`optimal_control`). Here, we don't know an analytical form
for the dynamics $f$ or the cost function $c$, but we assume that we're
able to *query/sample/simulate* them to get their values at a given
state and action. To clarify, consider the case where the dynamics are
given by real world physics. We can't (yet) write down an expression for
the dynamics that we can differentiate or integrate analytically.
However, we can still *simulate* the dynamics and cost function by
running a real-world experiment and measuring the resulting states and
costs. How can we adapt LQR to this more general nonlinear case?

### Local linearization

How can we apply LQR when the dynamics are nonlinear or the cost
function is more complex? We'll exploit the useful fact that we can take
a function that's *locally continuous* around $(s^\star, a^\star)$ and
approximate it nearby with low-order polynomials (i.e. its Taylor
approximation). In particular, as long as the dynamics $f$ are
differentiable around $(\st^\star, \act^\star)$ and the cost function
$c$ is twice differentiable at $(\st^\star, \act^\star)$, we can take a
linear approximation of $f$ and a quadratic approximation of $c$ to
bring us back to the regime of LQR.

Linearizing the dynamics around $(\st^\star, \act^\star)$ gives:


$$
\begin{gathered}
    f(\st, \act) \approx f(\st^\star, \act^\star) + \nabla_\st f(\st^\star, \act^\star) (\st - \st^\star) + \nabla_\act f(\st^\star, \act^\star) (\act - \act^\star) \\
    (\nabla_\st f(\st, \act))_{ij} = \frac{d f_i(\st, \act)}{d \st_j}, \quad i, j \le n_\st \qquad (\nabla_\act f(\st, \act))_{ij} = \frac{d f_i(\st, \act)}{d \act_j}, \quad i \le n_\st, j \le n_\act
\end{gathered}
$$

 and quadratizing the cost function around
$(\st^\star, \act^\star)$ gives: 

$$
\begin{aligned}
    c(\st, \act) & \approx c(\st^\star, \act^\star) \quad \text{constant term}                                                                                      \\
                 & \qquad + \nabla_\st c(\st^\star, \act^\star) (\st - \st^\star) + \nabla_\act c(\st^\star, \act^\star) (a - \act^\star) \quad \text{linear terms} \\
                 & \left. \begin{aligned}
                               & \qquad + \frac{1}{2} (\st - \st^\star)^\top \nabla_{\st \st} c(\st^\star, \act^\star) (\st - \st^\star)       \\
                               & \qquad + \frac{1}{2} (\act - \act^\star)^\top \nabla_{\act \act} c(\st^\star, \act^\star) (\act - \act^\star) \\
                               & \qquad + (\st - \st^\star)^\top \nabla_{\st \act} c(\st^\star, \act^\star) (\act - \act^\star)
                          \end{aligned} \right\} \text{quadratic terms}
\end{aligned}
$$

 where the gradients and Hessians are defined as


$$
\begin{aligned}
    (\nabla_\st c(\st, \act))_{i}         & = \frac{d c(\st, \act)}{d \st_i}, \quad i \le n_\st
                                          & (\nabla_\act c(\st, \act))_{i}                                               & = \frac{d c(\st, \act)}{d \act_i}, \quad i \le n_\act               \\
    (\nabla_{\st \st} c(\st, \act))_{ij}  & = \frac{d^2 c(\st, \act)}{d \st_i d \st_j}, \quad i, j \le n_\st
                                          & (\nabla_{\act \act} c(\st, \act))_{ij}                                       & = \frac{d^2 c(\st, \act)}{d \act_i d \act_j}, \quad i, j \le n_\act \\
    (\nabla_{\st \act} c(\st, \act))_{ij} & = \frac{d^2 c(\st, \act)}{d \st_i d \act_j}. \quad i \le n_\st, j \le n_\act
\end{aligned}
$$



**Exercise:** Note that this cost can be expressed in the general
quadratic form seen in
{eq}`general_quadratic_cost`. Derive the corresponding
quantities $Q, R, M, q, r, c$.

### Finite differencing

To calculate these gradients and Hessians in practice, we use a method
known as **finite differencing** for numerically computing derivatives.
Namely, we can simply use the limit definition of the derivative, and
see how the function changes as we add or subtract a tiny $\delta$ to
the input.


$$

\frac{d}{dx} f(x) = \lim_{\delta \to 0} \frac{f(x + \delta) - f(x)}{\delta}

$$


Note that this only requires us to be able to *query* the function, not
to have an analytical expression for it, which is why it's so useful in
practice.

### Local convexification

However, simply taking the second-order approximation of the cost
function is insufficient, since for the LQR setup we required that the
$Q$ and $R$ matrices were positive definite, i.e. that all of their
eigenvalues were positive.

One way to naively *force* some symmetric matrix $D$ to be positive
definite is to set any non-positive eigenvalues to some small positive
value $\varepsilon > 0$. Recall that any real symmetric matrix
$D \in \mathbb{R}^{n \times n}$ has an basis of eigenvectors $u_1, \dots, u_n$
with corresponding eigenvalues $\lambda_1, \dots, \lambda_n$ such that
$D u_i = \lambda_i u_i$. Then we can construct the positive definite
approximation by


$$

\widetilde{D} = \left( \sum_{i=1, \dots, n \mid \lambda_i > 0} \lambda_i u_i u_i^\top \right) + \varepsilon I.

$$



**Exercise:** Convince yourself that $\widetilde{D}$ is indeed positive
definite.

Note that Hessian matrices are generally symmetric, so we can apply this
process to $Q$ and $R$ to obtain the positive definite approximations
$\widetilde{Q}$ and $\widetilde{R}$.
Now that we have a upward-curved
quadratic approximation to the cost function, and a linear approximation
to the state transitions, we can simply apply the time-homogenous LQR
methods from [](#optimal_lqr).

But what happens when we enter states far away from $\st^\star$ or want
to use actions far from $\act^\star$? A Taylor approximation is only
accurate in a *local* region around the point of linearization, so the
performance of our LQR controller will degrade as we move further away.
We'll see how to address this in the next section using the **iterative LQR** algorithm.

:::{figure} shared/log_taylor.png
:name: local_linearization

Local linearization might only be accurate in a small region around the
point of linearization.
:::

(iterative_lqr)=
### Iterative LQR

To address these issues with local linearization, we'll use an iterative
approach, where we repeatedly linearize around different points to
create a *time-dependent* approximation of the dynamics, and then solve
the resulting time-dependent LQR problem to obtain a better policy. This
is known as **iterative LQR** or **iLQR**:

:::{prf:definition} Iterative LQR (high-level)
:label: ilqr

For each iteration of the algorithm:

::: steps
Form a time-dependent LQR problem around the current candidate
trajectory using local linearization.

Compute the optimal policy using [](time_dep_lqr).

Generate a new series of actions using this policy.

Compute a better candidate trajectory by interpolating between the
current and proposed actions.
:::
:::

Now let's go through the details of each step. We'll use superscripts to
denote the iteration of the algorithm. We'll also denote
$\bar \st_0 = \E_{\st_0 \sim \mu_0} [\st_0]$ as the expected initial
state.

At iteration $i$ of the algorithm, we begin with a **candidate**
trajectory
$\bar \tau^i = (\bar \st^i_0, \bar \act^i_0, \dots, \bar \st^i_{\hor-1}, \bar \act^i_{\hor-1})$.

**Step 1: Form a time-dependent LQR problem.** At each timestep
$\hi \in [\hor]$, we use the techniques from
[](approx_nonlinear) to linearize the dynamics and
quadratize the cost function around $(\bar \st^i_\hi, \bar \act^i_\hi)$:


$$
\begin{aligned}
    f_\hi(\st, \act) & \approx f(\bar {\st}^i_\hi, \bar {\act}^i_\hi) + \nabla_{\st } f(\bar {\st}^i_\hi, \bar {\act}^i_\hi)(\st - \bar {\st}^i_\hi) + \nabla_{\act } f(\bar {\st}^i_\hi, \bar {\act}^i_\hi)(\act - \bar {\act}^i_\hi)                         \\
    c_\hi(\st, \act) & \approx c(\bar {\st}^i_\hi, \bar {\act}^i_\hi) + \begin{bmatrix}
                                                              \st - \bar {\st }^i_\hi& \act - \bar {\act}^i_\hi
                                                          \end{bmatrix} \begin{bmatrix}
                                                                            \nabla_{\st } c(\bar {\st}^i_\hi, \bar {\act}^i_\hi)\\
                                                                            \nabla_{\act} c(\bar {\st}^i_\hi, \bar {\act}^i_\hi)
                                                                        \end{bmatrix}                                                      \\
                     & \qquad + \frac{1}{2} \begin{bmatrix}
                                                \st - \bar {\st }^i_\hi& \act - \bar {\act}^i_\hi
                                            \end{bmatrix} \begin{bmatrix}
                                                              \nabla_{\st \st} c(\bar {\st}^i_\hi, \bar {\act}^i_\hi)  & \nabla_{\st \act} c(\bar {\st}^i_\hi, \bar {\act}^i_\hi)  \\
                                                              \nabla_{\act \st} c(\bar {\st}^i_\hi, \bar {\act}^i_\hi) & \nabla_{\act \act} c(\bar {\st}^i_\hi, \bar {\act}^i_\hi)
                                                          \end{bmatrix}
    \begin{bmatrix}
        \st - \bar {\st }^i_\hi\\
        \act - \bar {\act}^i_\hi
    \end{bmatrix}.
\end{aligned}
$$



**Step 2: Compute the optimal policy.** We can now solve the
time-dependent LQR problem using the Riccati equation from
[](time_dep_lqr) to compute the optimal policy
$\pi^i_0, \dots, \pi^i_{\hor-1}$.

**Step 3: Generate a new series of actions.** We can then generate a new
sample trajectory by taking actions according to this optimal policy:


$$

\bar \st^{i+1}_0 = \bar \st_0, \qquad \widetilde \act_\hi = \pi^i_\hi(\bar \st^{i+1}_\hi), \qquad \bar \st^{i+1}_{\hi+1} = f(\bar \st^{i+1}_\hi, \widetilde \act_\hi).

$$


Note that the states are sampled according to the *true* dynamics, which
we assume we have query access to.

**Step 4: Compute a better candidate trajectory.**, Note that we've
denoted these actions as $\widetilde \act_\hi$ and aren't directly using
them for the next iteration $\bar \act^{i+1}_\hi$. Rather, we want to
*interpolate* between them and the actions from the previous iteration
$\bar \act^i_0, \dots, \bar \act^i_{\hor-1}$. This is so that the cost
will *increase monotonically,* since if the new policy turns out to
actually be worse, we can stay closer to the previous trajectory. (Can
you think of an intuitive example where this might happen?)

Formally, we want to find $\alpha \in [0, 1]$ to generate the next
iteration of actions
$\bar \act^{i+1}_0, \dots, \bar \act^{i+1}_{\hor-1}$ such that the cost
is minimized: 

$$
\begin{aligned}
    \min_{\alpha \in [0, 1]} \quad & \sum_{\hi=0}^{\hor-1} c(\st_\hi, \bar \act^{i+1}_\hi)                     \\
    \text{where} \quad             & \st_{\hi+1} = f(\st_\hi, \bar \act^{i+1}_\hi)                             \\
                                   & \bar \act^{i+1}_\hi = \alpha \bar \act^i_\hi + (1-\alpha) \widetilde \act_\hi \\
                                   & \st_0 = \bar \st_0.
\end{aligned}
$$

 Note that this optimizes over the closed interval
$[0, 1]$, so by the Extreme Value Theorem, it's guaranteed to have a
global maximum.

The final output of this algorithm is a policy $\pi^{n_\text{steps}}$
derived after $n_\text{steps}$ of the algorithm. Though the proof is
somewhat complex, one can show that for many nonlinear control problems,
this solution converges to a locally optimal solution (in the policy
space).

## Summary

This chapter introduced some approaches to solving different variants of
the optimal control problem
{prf:ref}`optimal_control`. We began with the simple case of linear
dynamics and an upward-curved quadratic cost. This model is called the
LQR and we solved for the optimal policy using dynamic programming. We
then extended these results to the more general nonlinear case via local
linearization. We finally saw the iterative LQR algorithm for solving
nonlinear control problems.
