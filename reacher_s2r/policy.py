"""Load a trained model (+ optional VecNormalize statistics) as a simple obs -> action function."""
import pickle
from pathlib import Path

from stable_baselines3 import PPO, SAC


class Policy:
    def __init__(self, model_dir: str | Path, algo: str = "ppo"):
        model_dir = Path(model_dir)
        self.model = (PPO if algo == "ppo" else SAC).load(model_dir / "best_model.zip", device="cpu")
        self.vecnorm = None
        stats = model_dir / "vecnormalize.pkl"
        if stats.exists():
            with open(stats, "rb") as f:
                self.vecnorm = pickle.load(f)  # only obs_rms / clip_obs are used

    def __call__(self, obs):
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return action
