"""Evaluate policies on the nominal model and on the perturbed "real robot".

  python evaluate.py                    # 100 episodes, all available policies
Writes results/results.json and results/results.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reacher_s2r.envs import REAL_OBS_NOISE_STD, REAL_ROBOT, REAL_ROBOT_HARD, make_env
from reacher_s2r.policy import Policy

SUCCESS_RADIUS = 0.02  # 2 cm

POLICIES = {
    "Random": None,
    "PPO": ("ppo", "models/ppo_nominal"),
    "PPO + DR": ("ppo", "models/ppo_dr"),
    "SAC": ("sac", "models/sac_nominal"),
}
ENVS = {"nominal": "Nominal model", "real": "\"Real robot\"", "real_hard": "\"Real robot\", weak motors"}


def load(spec):
    if spec is None:
        return None
    algo, path = spec
    if not (Path(path) / "best_model.zip").exists():
        return "missing"
    return Policy(path, algo)


def run_episodes(model, kind, n_episodes, seed=12345):
    env = make_env(kind, seed=seed)
    env.action_space.seed(seed)
    rewards, final_dists = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action = model(obs)
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
        rewards.append(total)
        final_dists.append(-info["reward_dist"])  # reward_dist = -||fingertip - target||
    env.close()
    d = np.array(final_dists)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_final_dist_cm": float(d.mean() * 100),
        "sem_final_dist_cm": float(d.std(ddof=1) / np.sqrt(len(d)) * 100),
        "median_final_dist_cm": float(np.median(d) * 100),
        "success_rate": float((d < SUCCESS_RADIUS).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    args = p.parse_args()

    results = {}
    for name, spec in POLICIES.items():
        model = load(spec)
        if model == "missing":
            continue
        results[name] = {}
        for kind in ENVS:
            results[name][kind] = run_episodes(model, kind, args.episodes)
            r = results[name][kind]
            print(f"{name:10s} {kind:8s} reward {r['mean_reward']:7.2f}  "
                  f"dist {r['mean_final_dist_cm']:5.2f} cm  success {r['success_rate']:.0%}")

    out = Path("results")
    out.mkdir(exist_ok=True)
    meta = {"episodes": args.episodes, "success_radius_m": SUCCESS_RADIUS,
            "real_robot": REAL_ROBOT, "real_robot_hard": REAL_ROBOT_HARD, "real_obs_noise_std": REAL_OBS_NOISE_STD}
    (out / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))

    lines = ["| Policy | Model | Mean reward | Mean final distance ± SE [cm] | Median [cm] | Success (< 2 cm) |",
             "|---|---|---:|---:|---:|---:|"]
    for name, per_env in results.items():
        for kind, r in per_env.items():
            lines.append(f"| {name} | {ENVS[kind]} | {r['mean_reward']:.2f} ± {r['std_reward']:.2f} | "
                         f"{r['mean_final_dist_cm']:.2f} ± {r['sem_final_dist_cm']:.2f} | {r['median_final_dist_cm']:.2f} | "
                         f"{r['success_rate']:.0%} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
