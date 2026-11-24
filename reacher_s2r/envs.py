"""Reacher environments: nominal sim, perturbed "real robot" and domain randomization."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

ENV_ID = "Reacher-v5"

# Indices of the arm in the MuJoCo model (the target has its own slide joints/body).
ARM_DOFS = [0, 1]          # joint0 (shoulder), joint1 (elbow)
ARM_BODIES = [1, 2, 3]     # body0, body1, fingertip

# "Real robot" = nominal model with these fixed changes (sim-to-real gap).
REAL_ROBOT = {"damping": 2.0, "mass": 1.5, "gear": 0.8}
# Harder variant: same, but the motors deliver only half of the nominal torque.
REAL_ROBOT_HARD = {"damping": 2.0, "mass": 1.5, "gear": 0.5}
REAL_OBS_NOISE_STD = 0.01

# Domain randomization range: each factor sampled from U[1 - DR_RANGE, 1 + DR_RANGE].
DR_RANGE = 0.5


class PhysicsScaler(gym.Wrapper):
    """Keeps nominal physics parameters and lets you scale them."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        m = self.env.unwrapped.model
        self._nom_damping = m.dof_damping.copy()
        self._nom_mass = m.body_mass.copy()
        self._nom_gear = m.actuator_gear.copy()

    def set_scales(self, damping: float = 1.0, mass: float = 1.0, gear: float = 1.0) -> None:
        m = self.env.unwrapped.model
        m.dof_damping[:] = self._nom_damping
        m.body_mass[:] = self._nom_mass
        m.actuator_gear[:] = self._nom_gear
        m.dof_damping[ARM_DOFS] *= damping
        m.body_mass[ARM_BODIES] *= mass
        m.actuator_gear[:, 0] *= gear
        self.scales = {"damping": damping, "mass": mass, "gear": gear}


class FixedPhysics(PhysicsScaler):
    """Constant physics change, e.g. the 'real robot'."""

    def __init__(self, env: gym.Env, damping=1.0, mass=1.0, gear=1.0):
        super().__init__(env)
        self.set_scales(damping, mass, gear)


class DomainRandomization(PhysicsScaler):
    """Samples new damping / mass / gear factors on every reset()."""

    def __init__(self, env: gym.Env, rng_range: float = DR_RANGE, seed: int | None = None):
        super().__init__(env)
        self.rng_range = rng_range
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        lo, hi = 1.0 - self.rng_range, 1.0 + self.rng_range
        self.set_scales(*self.rng.uniform(lo, hi, size=3))
        return self.env.reset(**kwargs)


class ObsNoise(gym.ObservationWrapper):
    """Gaussian sensor noise on observations (the reward/info still use the true state)."""

    def __init__(self, env: gym.Env, std: float, seed: int | None = None):
        super().__init__(env)
        self.std = std
        self.rng = np.random.default_rng(seed)

    def observation(self, obs):
        return (obs + self.rng.normal(0.0, self.std, size=obs.shape)).astype(obs.dtype)


def make_env(kind: str = "nominal", seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """kind: 'nominal' (default sim), 'real' / 'real_hard' (perturbed robot), 'dr' (domain randomization)."""
    env = gym.make(ENV_ID, render_mode=render_mode)
    if kind == "nominal":
        return env
    if kind in ("real", "real_hard"):
        env = FixedPhysics(env, **(REAL_ROBOT if kind == "real" else REAL_ROBOT_HARD))
        return ObsNoise(env, REAL_OBS_NOISE_STD, seed=seed)
    if kind == "dr":
        env = DomainRandomization(env, seed=seed)
        return ObsNoise(env, REAL_OBS_NOISE_STD, seed=None if seed is None else seed + 1)
    raise ValueError(f"unknown env kind: {kind}")
