import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

def choice_env(index):
    switcher = {
        0: {
            0: "Thien"
        },
        1: "MountainCar-v0",
        2: "Acrobot-v1",
        3: "Pendulum-v1",
        4: "MountainCarContinuous-v0",
        5: "FrozenLake8x8-v1",
        6: "Assault-ram-v0",
        7: "Breakout-v0",
        8: "Skiing-ram-v0"
    }
    return switcher[index][index]

print(choice_env(0))

