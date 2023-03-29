import numpy as np
import gymnasium as gym
import torch


def trajectory(trajectory: np.ndarray, env: gym.Env) -> float:
    
    observation, info = env.reset(seed=0)
    
    t = 0
    cum_reward = 0
    terminated, truncated = False, False
    while t < len(trajectory) and not terminated and not truncated:
        action = trajectory[t]
        observation, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
        t += 1
    return cum_reward / len(trajectory)


def trajectories(trajectories: torch.Tensor, env: gym.Env, verbose: bool = False) -> torch.Tensor:
    
    x = trajectories.cpu().numpy()
    y = torch.zeros(len(x))
    for i, trj in enumerate(x):
        y[i] = eval_trajectory(trajectory=trj, env=env)
    if verbose:
        print(f"Avg reward: {y.mean().item():.3f} +/- {y.std().item():.3f}; Range = {y.min().item():.3f}, {y.max().item():.3f}")
    return y 