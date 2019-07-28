import torch
import torch.nn as nn
from torch.optim import adam
from torch.optim import sgd
import torch.nn.functional as F
import gym
from collections import namedtuple
import numpy as np
import random


env = gym.make("MountainCar-v0")
env = env.unwrapped
env.seed(1)
RENDER = False
state_space = 2
action_space = 3


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_space, 32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,action_space)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output


class Agent():
    def __init__(self):
        super(Agent,self).__init__()
        self.online_net = Net()
        self.target_net = Net()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.experience = namedtuple("agent_experience", ['state', 'action', 'reward', 'next_state'])
        self.experience_pool = []
        self.epsilon = 0.1
        self.gamma = 0.99
        self.batch_size = 64
        self.learning_rate = 0.0025
        self.optimizer = adam.Adam(self.online_net.parameters(), self.learning_rate)
        # self.optimizer = sgd.SGD(self.online_net.parameters(), self.learning_rate)
        self.loss = nn.MSELoss()

    def select_action(self, state):
        if np.random.random() > self.epsilon:    # epsilon-greedy
            action_Q_value = self.online_net.forward(torch.FloatTensor(state))
            action = action_Q_value.argmax().item()
        else:
            action = np.random.randint(3)
        return action

    def save_experience(self, exp):
        if len(self.experience_pool) >= 20000:
            self.experience_pool = self.experience_pool[1: ]  # 保持经验池经验个数恒定
        self.experience_pool.append(exp)

    def update_online_net(self):
        if len(self.experience_pool) > 100:
            batch_experience = random.choices(self.experience_pool, k=self.batch_size)   # 对经验池进行批处理
            batch_state = [x[0] for x in batch_experience]
            batch_state = torch.FloatTensor(batch_state)
            batch_action = [[x[1]] for x in batch_experience]
            batch_action = torch.LongTensor(batch_action)
            batch_reward = [[x[2]] for x in batch_experience]
            batch_reward = torch.FloatTensor(batch_reward)
            batch_next_state = [x[3] for x in batch_experience]
            batch_next_state = torch.FloatTensor(batch_next_state)
            batch_online_Q = self.online_net.forward(batch_state).gather(dim=1, index=batch_action)
            batch_target_Q = self.gamma*self.target_net.forward(batch_next_state).detach().max(dim=1)[0].view(self.batch_size, 1) \
                             + batch_reward
            batch_loss = self.loss(batch_online_Q, batch_target_Q)
            batch_loss = batch_loss / self.batch_size
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

        else:
            pass

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())   # 将online_net同步到target_net


def main():
    agent = Agent()
    for episode in range(500):
        state = env.reset()
        for time_slot in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if RENDER:
                env.render()
            if done:
                reward = 1
            reward = reward * 100 if reward >0 else reward * 1
            agent.save_experience(agent.experience(state,action,reward,next_state))
            agent.update_online_net()
            if time_slot % 100 == 0:
                agent.update_target_net()
            if done:
                print("My DQN episode {}, the time is {}".format(episode,time_slot))
                break
            if time_slot >= 9999:
                print("My DQN episode {}, the time is {}".format(episode, time_slot))
                break
            state = next_state

if __name__ == '__main__':
    main()
