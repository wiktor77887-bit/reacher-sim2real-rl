"""Learning curves from EvalCallback logs (models/*/evaluations.npz) -> docs/learning_curves.png"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = {"ppo_nominal": ("PPO (nominal model)", "#2a6fdb"),
          "ppo_dr": ("PPO + DR (evaluated on DR)", "#e07a1f"),
          "sac_nominal": ("SAC (nominal model)", "#2e9e5b")}


def main():
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (label, color) in LABELS.items():
        f = Path("models") / name / "evaluations.npz"
        if not f.exists():
            continue
        d = np.load(f)
        mean, std = d["results"].mean(1), d["results"].std(1)
        ax.plot(d["timesteps"], mean, label=label, color=color)
        ax.fill_between(d["timesteps"], mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episode reward (20 episodes)")
    ax.set_title("Learning curves – Reacher-v5")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/learning_curves.png", dpi=130)
    print("docs/learning_curves.png")


if __name__ == "__main__":
    main()
