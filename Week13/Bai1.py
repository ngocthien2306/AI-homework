''' NOTE: The following code is INCOMPLETE and CAN'T RUN. You need to add code here and there to make it work. '''
import timeit

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import namedtuple

# Khởi tạo các Hidden_size (số lượng neural), Batch_size (số lượng episole)
# và Percentile (bao nhiêu phần trăm nhỏ hơn n episole)
# MISSING CODE HERE
model = 3

if model == 1:
    HIDDEN_SIZE = 128
    BATCH_SIZE = 16
    PERCENTILE = 70

if model == 2:
    HIDDEN_SIZE = 80
    BATCH_SIZE = 10
    PERCENTILE = 90

if model == 3:
    HIDDEN_SIZE = 100
    BATCH_SIZE = 20
    PERCENTILE = 75

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)

        )

    def forward(self, input):
        return self.net(input);


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def get_batches(env, net, batch_size):
    batch = []

    episode_reward = 0.0
    episode_steps = []

    obs = env.reset()

    sm = nn.Softmax(dim=1)  # activation of output layer

    while True:
        obs_v = torch.FloatTensor([obs])

        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)

        # MISSING CODE HERE: create next observation, reward over each time we iterate to find episole suitable

        next_obs, reward, is_done, info = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)

        if is_done:
            # MISSING CODE HERE: we need declare episole to add them to batch
            episode = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(episode)

            episode_reward = 0.0
            episode_steps = []
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
        if reward < reward_bound:  # MISSING CODE HERE: compare reward to select goad observation it mean (the figure we can find good solution)
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean  # MISSING CODE HERE: return observation of good reward

def train_agent(env):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = Net(obs_size, HIDDEN_SIZE, n_actions)

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    for iter_no, batch in enumerate(get_batches(env, net, BATCH_SIZE)):

        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)  # MISSING CODE HERE: without filter_batch method to fliter the good batch

        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)

        loss_v.backward()
        optimizer.step()

        file = open("log.txt", "a")
        file.write(
            "%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b) + "\n")
        # print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))

        # if reward_m > 199: # for CartPole-v0
        if reward_m > 200:  # for LunarLander-v2
            torch.save(net, 'trained_net_' + str(HIDDEN_SIZE) + 'neurons.pt')  # qNOTE: save ONLY 1 var/file
            file.close()
            print("Solved! Saved the model.")
            break

# use the trained network
def use_trained(env):
    env = gym.wrappers.Monitor(env, directory="lunar_lander_2", force=True)
    # MISSING CODE HERE: without observation
    obs = env.reset()

    net = torch.load('trained_net_' + str(HIDDEN_SIZE) + 'neurons.pt')
    sm = nn.Softmax(dim=1)

    total_reward = 0

    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)

        # MISSING CODE HERE
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.01)
        # print the good observation that we are training for agent
        print(obs)
        total_reward += reward

        # print total_reward để xem agent đã train thực sự hiểu quả, bởi vì trong vài trường hợp khi train reward đạt tới 500
        # nhưng khi chạy thì khoảng 300-400 nó đã ngã, do model ta thiết lập chưa hiệu quả
        print(total_reward)

        if done:
            env.close()
            break

    print("Model %d (%d neurons). Total reward: %.2f" % (model, HIDDEN_SIZE, total_reward))

if __name__ == "__main__":
    # MISSING CODE HERE: lack variable init the environment name

    env = gym.make("LunarLander-v2")
    start_time = time.time()
    train_agent(env)
    #use_trained(env)
    end_time = time.time()
    print("time: ", str(end_time - start_time))



