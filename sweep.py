"""Robustness sweep: change one physics parameter at a time and measure final distance.

  python sweep.py        # writes docs/robustness_sweep.png and results/sweep.json
"""
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from reacher_s2r.envs import ENV_ID, FixedPhysics
from reacher_s2r.policy import Policy

POLICIES = {"PPO": "models/ppo_nominal", "PPO + DR": "models/ppo_dr"}
PARAMS = {"damping": "joint damping", "mass": "link mass", "gear": "actuator gear"}
SCALES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
EPISODES = 100


def final_dist(model, scales, seed=777):
    env = FixedPhysics(gym.make(ENV_ID), **scales)
    d = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            a = model(obs)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        d.append(-info["reward_dist"] * 100)
    return float(np.mean(d))


def main():
    models = {n: Policy(p) for n, p in POLICIES.items()}
    res = {n: {p: [final_dist(m, {p: s}) for s in SCALES] for p in PARAMS} for n, m in models.items()}
    Path("results").mkdir(exist_ok=True)
    Path("results/sweep.json").write_text(json.dumps({"scales": SCALES, "results": res}, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    colors = {"PPO": "#2a6fdb", "PPO + DR": "#e07a1f"}
    for ax, (p, label) in zip(axes, PARAMS.items()):
        for n in models:
            ax.plot(SCALES, res[n][p], marker="o", label=n, color=colors[n])
        ax.axvspan(0.5, 1.5, color="0.92", zorder=0)
        ax.axhline(2.0, color="0.5", ls="--", lw=1)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(SCALES, [str(s) for s in SCALES])
        ax.set_title(f"Scale: {label}")
        ax.set_xlabel("× nominal value")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean final distance [cm]")
    axes[0].legend(frameon=False)
    fig.suptitle("Robustness to a single parameter change (grey = DR range, dashed = 2 cm)")
    fig.tight_layout()
    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/robustness_sweep.png", dpi=130)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
