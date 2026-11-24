"""Train a policy for Reacher-v5.

  python train.py --env nominal            # policy A: default model
  python train.py --env dr                 # policy B: domain randomization
  python train.py --env nominal --algo sac --timesteps 150000
"""
import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from reacher_s2r.envs import make_env

# PPO hyperparameters for Reacher from RL Baselines3 Zoo (rl-baselines3-zoo/hyperparams/ppo.yml).
PPO_KWARGS = dict(n_steps=512, batch_size=32, gamma=0.9, learning_rate=1.04e-4, ent_coef=7.5e-8,
                  clip_range=0.3, n_epochs=5, gae_lambda=1.0, max_grad_norm=0.9, vf_coef=0.95)


class SaveVecNormalize(BaseCallback):
    """Saves observation-normalization statistics together with each new best model."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def _on_step(self) -> bool:
        if self.model.get_vec_normalize_env() is not None:
            self.model.get_vec_normalize_env().save(str(self.path / "vecnormalize.pkl"))
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["nominal", "dr"], default="nominal")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--eval-freq", type=int, default=20_000, help="in total env steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    name = f"{args.algo}_{args.env}"
    out = Path("models") / name
    out.mkdir(parents=True, exist_ok=True)

    ppo = args.algo == "ppo"
    n_envs = args.n_envs if ppo else 1
    env = make_vec_env(make_env, n_envs=n_envs, seed=args.seed, env_kwargs={"kind": args.env})
    # Best model is chosen on the training distribution, never on the "real robot".
    eval_env = make_vec_env(make_env, n_envs=1, seed=args.seed + 1000, env_kwargs={"kind": args.env})
    if ppo:  # observation normalization; EvalCallback syncs the statistics to eval_env
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, tensorboard_log="runs/", **PPO_KWARGS)
    else:
        model = SAC("MlpPolicy", env, seed=args.seed, verbose=1, tensorboard_log="runs/")

    cb = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=max(args.eval_freq // n_envs, 1),
                      best_model_save_path=str(out), log_path=str(out), deterministic=True,
                      callback_on_new_best=SaveVecNormalize(out))
    model.learn(total_timesteps=args.timesteps, callback=cb, tb_log_name=name)
    model.save(out / "final_model")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
