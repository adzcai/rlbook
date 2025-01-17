"""Some exports to be used in the book."""

import matplotlib.pyplot as plt

from collections.abc import Callable

# convenient class builder
from typing import NamedTuple, Optional

# function typings
from collections.abc import Callable

# array typings
from jaxtyping import Array, Float, Int

# convenient function composition
from functools import partial

# numerical computing and linear algebra
import jax
import jax.numpy as jnp
import jax.random as rand
import numpy as np

# reinforcement learning environments
import gymnasium as gym

# progress bars
from tqdm import tqdm

# print functions as latex
import latexify
from latexify.plugins.sum_prod import SumProdPlugin
from latexify.plugins.jaxtyping import JaxTypingPlugin
from latexify.plugins.numpy import NumpyPlugin


latex = partial(
    latexify.algorithmic,
    use_math_symbols=True,
    plugins=[
        SumProdPlugin(),
        JaxTypingPlugin(),
        NumpyPlugin(),
    ]
)

plt.style.use("fivethirtyeight")

def rng(seed: int):
    """An iterator of JAX PRNG keys."""
    key = rand.PRNGKey(seed)
    while True:
        key, subkey = rand.split(key)
        yield subkey
