import gym
import time
import gym.envs.box2d
import random
import numpy as np
from gym import envs


class Agent:
    def __init__(self, env, choice):
        self.check_type = None
        # print(type(env.action_space)) để lấy kiểu của biến
        # ví dụ Type: Box() or Discrete()
        self.env = env
        if str(type(env.action_space)) == "<class 'gym.spaces.discrete.Discrete'>":
            self.check_type = True
        else:
            self.check_type = False

        if self.check_type:
            self.action_size = env.action_space.n

        else:
            # search with "how to get action gym.spaces.box.Box"
            # https://github.com/openai/gym/blob/master/gym/spaces/box.py
            # em đã xem link trên để hiểu được các action của type Box là high, low và shape tại line 25

            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape

        self.choice = choice
        self.final_reward = 0

    def get_action(self, observation):
        # print(self.check_type)
        # print(type(env.action_space))

        if self.check_type:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low, self.action_high, self.action_shape)
        # Ta có thể sử dụng cách tạo action bên dưới, cách dưới sẽ linh hoạt hơn vì
        # ta sẽ cho nó những điều kiện dựa vào observation để làm cho reward cao hơn
        '''
        if self.choice == 0:
            action = observation[2]
            if action < 0:
                action = 0
            else:
                action = 1

        elif self.choice == 1:
            action = random.choice([0, 1, 2])

        elif self.choice == 2:
            action = random.choice([0, 1, 2])

        elif self.choice == 3 or 4:
            action = np.random.uniform(self.action_low, self.action_high, self.action_shape)
            
        else:
            action = random.choice([0, 1, 2, 3])
        '''

        return action

def run_agent(agent, env):

    observation = env.reset()
    while True:
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        agent.final_reward += reward
        print(observation)
        time.sleep(0.1)
        env.render()
        if done:
            env.close()
            break
    print("Finished")
    print(agent.final_reward)

def choice_env(index):
    switcher = {
        0: "CartPole-v0",
        1: "MountainCar-v0",
        2: "Acrobot-v1",
        3: "Pendulum-v1",
        4: "MountainCarContinuous-v0",
        5: "FrozenLake8x8-v1",
        6: "Assault-ram-v0",
        7: "Breakout-v0",
        8: "Skiing-ram-v0"
    }
    return switcher[index]

if __name__ == "__main__":
    print("""
    0 - CartPole-v0
    1 - MountainCar-v0
    2 - Acrobot-v1
    3 - Pendulum-v1
    4 - MountainCarContinuous-v0
    5 - FrozenLake8x8-v1
    6 - Assault-ram-v0
    7 - Breakout-v0
    8 - "Skiing-ram-v0"
    """)

    choice = int(input("Please choose the environment you want to test: "))

    name_env = choice_env(choice)
    env = gym.make(name_env)
    agent = Agent(env, choice)
    run_agent(agent, env)

