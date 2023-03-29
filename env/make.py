import gymnasium as gym
import numpy as np


def mujoco_cartpole():
    
    def new_step(self, action):
        # Action given between 0 and 1
        # Need to be scaled between - 3 and 3
        action = action*6 - 3
        action = np.array([action, ])
        return self.__original_step(action)
        
    env = gym.make('InvertedPendulum-v4')
    env.__original_step = env.step
    env.step = new_step.__get__(env, env.__class__)
    return env