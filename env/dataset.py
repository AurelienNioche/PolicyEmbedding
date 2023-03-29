import torch
from torch.utils.data import Dataset
import env.eval as env_eval
import gymnasium as gym


class TrajectoryDataset(Dataset): 
    def __init__(self, x, y):
        self.x = x
        self.y = y
        super().__init__()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make(
        env: gym.Env,
        time_horizon: int,
        device: torch.device,
        seed: int = 12345,
        n_sample: int = 2000,
        upper_bound: float = None,
        verbose: bool = True):
    
    print("Sampling trajectories and evaluating them...")
    
    torch.manual_seed(seed)
    x = torch.rand((n_sample, time_horizon))
    y = env_eval.trajectories(trajectories=x, env=env)

    if upper_bound is not None:
        # Exclude best ones
        not_too_good = y <= 0.8
        y = y[not_too_good]
        x = x[not_too_good]
    
    dataset = TrajectoryDataset(x=x.to(device), y=y.to(device))

    if verbose:
        print(f"n={n_sample}; avg = {y.mean():.3f} +/- {y.std():.3f}; range  = {y.min().item():.3f}, {y.max().item():.3f}")
    return dataset