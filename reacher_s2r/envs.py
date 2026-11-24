"""Reacher environment factory."""
from __future__ import annotations

import gymnasium as gym

ENV_ID = "Reacher-v5"



def make_env(kind: str = "nominal", seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """kind: 'nominal' (default sim)."""
    if kind != "nominal":
        raise ValueError(f"unknown env kind: {kind}")
    return gym.make(ENV_ID, render_mode=render_mode)
