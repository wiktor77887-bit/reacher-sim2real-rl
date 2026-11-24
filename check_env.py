"""Smoke test: Reacher-v5 with random actions in a window (render_mode="human")."""
import gymnasium as gym

try:
    env = gym.make("Reacher-v5", render_mode="human")
except gym.error.Error:
    env = gym.make("Reacher-v4", render_mode="human")

obs, info = env.reset(seed=0)
print("obs:", obs.shape, "action:", env.action_space)
for _ in range(500):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
