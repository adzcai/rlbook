"""Some exports to be used in the book."""

import matplotlib.pyplot as plt

# convenient class builder
from typing import NamedTuple

# function typings
from collections.abc import Callable

# array typings
from jaxtyping import Array, Float, Int

# convenient function composition
from functools import partial

# numerical computing and linear algebra
import jax
import jax.numpy as jnp

import gymnasium as gym

# print functions as latex
import latexify

latex = partial(latexify.algorithmic, use_math_symbols=True, to_file="./pseudocode")

plt.style.use("fivethirtyeight")

