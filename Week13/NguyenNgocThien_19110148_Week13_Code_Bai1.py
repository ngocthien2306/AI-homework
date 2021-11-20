import random as rd
import gym 
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from collections import namedtuple  # create tuple

# 11 - HIDDEN_SIZE
# 12 - BATCH_SIZE
# 13 - PERCENTILE


model = 2
if model == 1:
    HIDDEN_SIZE = 128
    BATCH_SIZE = 16
    PERCENTILE = 70

if model == 2:
    HIDDEN_SIZE = 50
    BATCH_SIZE = 10
    PERCENTILE = 80

if model == 3:
    HIDDEN_SIZE = 15
    BATCH_SIZE = 10
    PERCENTILE = 90

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    def forward(self, input):
        return self.net(input)


Episode = namedtuple("Episole", field_names=['reward', "steps"])


Episode_Step = namedtuple("EpisoleStep", field_names=['observation', "action"])


def get_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0  # Tổng reward của episole
    episole_steps = []

    obs = env.reset()
    sm = nn.Softmax(dim=1)  # activation of output layer, distributed probability
    while True:
        # Chạy vòng lặp để lấy các batch
        obs_v = torch.FloatTensor([obs])
        # Biến đổi observation thành tensor ví dụ mảng 1 chiều, 2, 3 chiều...
        action_probs_v = sm(net(obs_v))
        action_probs = action_probs_v.data.numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)

        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        step = Episode_Step(observation=obs, action=action)
        episole_steps.append(step)

        if done:
            e = Episode(reward=episode_reward, steps=episole_steps)
            batch.append(e)
            episode_reward = 0.0
            episole_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        # nếu reward đạt chuẩn ta đưa vô train_obs, train_act
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    train_mode = False

    if train_mode:
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
        net = Net(obs_size, HIDDEN_SIZE, n_actions)
        obj = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=net.parameters(), lr=0.01)
        for iter_no, batch in enumerate(get_batches(env, net, BATCH_SIZE)):

            obs_v, act_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)

            optimizer.zero_grad()
            action_score_v = net(obs_v)
            loss_v = obj(action_score_v, act_v)

            loss_v.backward()
            optimizer.step()

            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f"
                  % (iter_no, loss_v.item(), reward_m, reward_b))
            if reward_m > 499:
                torch.save(net, "trained_net" + str(HIDDEN_SIZE) + "neurons.pt")
                print("Solved! Saved the model.")
                break
    else:
        env = gym.wrappers.Monitor(env, directory="solved_cartpole", force=True)
        obs = env.reset()

        net = torch.load("trained_net" + str(HIDDEN_SIZE) + "neurons.pt")
        sm = nn.Softmax(dim=1)
        total_reward = 0

        while True:

            obs_v = torch.FloatTensor([obs])
            act_probs_v = sm(net(obs_v))
            act_probs = act_probs_v.data.numpy()[0]
            action = np.random.choice(len(act_probs), p=act_probs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.01)
            print(obs)

            if done:
                print(total_reward)
                break
