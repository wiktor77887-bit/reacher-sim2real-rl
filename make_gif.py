"""Record a GIF of the trained policy: python make_gif.py [--model path] [--env nominal|real]"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering on Linux; use "glfw" on Windows

import imageio.v2 as imageio

from reacher_s2r.envs import make_env
from reacher_s2r.policy import Policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ppo_nominal")
    p.add_argument("--env", default="nominal")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--out", default="docs/reacher.gif")
    args = p.parse_args()

    model = Policy(args.model)
    env = make_env(args.env, seed=0, render_mode="rgb_array")
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=100 + ep)
        done = False
        while not done:
            a = model(obs)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
            frames.append(env.render()[::2, ::2])  # 240x240 keeps the GIF small
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=1000 / 30, loop=0)
    print(f"{len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
