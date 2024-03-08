# Supervised learning

Supervised learning is a subfield of machine learning that focuses on learning from *labeled* data.

::::{prf:theorem} Conditional expectation minimizes mean squared error

We draw a random variable $X$ from a distribution $P(X)$ and a random variable $Y$ from a conditional distribution $P(Y|X)$. The conditional expectation of $Y$ given $X$ minimizes the mean squared error:

$$
E[Y \mid X] = \arg\min_{(X, Y)} E[(Y - f(X))^2]
$$

where $f$ is a function of $X$.
::::